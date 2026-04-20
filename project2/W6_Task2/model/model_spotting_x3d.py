"""
File containing the main model.
"""

#Standard imports
import os
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from thop import profile, clever_format

#Local imports
from model.modules import BaseRGBModel, FCLayers, FocalLoss, TemporalTransformer, TemporalGRU, step
from util.loss_weigths import build_class_weights
from util.io import load_json

from pytorchvideo.models.hub import x3d_s, x3d_m, x3d_l

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self.args = args
            self._feature_arch = self.args.feature_arch

            if self._feature_arch == "x3d_s":
                features = x3d_s(pretrained=True)
            elif self._feature_arch == "x3d_m":
                features = x3d_m(pretrained=True)
            elif self._feature_arch == "x3d_l":
                features = x3d_l(pretrained=True)
            else:
                raise NotImplementedError(self._feature_arch)
            
            feat_dim = 192  # X3D-S/M/L output feature dimension after block 4
            self._backbone_dim = feat_dim
            self._features = features

            # Temporal transformer
            if self.args.temporal_model == "transformer":
                self._temporal_model = TemporalTransformer(
                    clip_len = self.args.clip_len,
                    embed_dim = self._backbone_dim,
                    num_heads=args.attention_heads,
                    depth=args.transformer_depth,
                    attn_dropout=args.transformer_dropout,
                    mlp_dim=args.transformer_mlp_dim,
                    proj_dropout=args.proj_dropout
                )
                self._head_dim = feat_dim

            elif self.args.temporal_model == "gru":
                self._temporal_model = TemporalGRU(
                    embed_dim=self._backbone_dim,
                    hidden_dim=args.gru_hidden_dim,
                    num_layers=args.gru_layers,
                    dropout=args.gru_dropout,
                    bidirectional=args.gru_bidirectional
                )
                self._head_dim = self._temporal_model.out_dim

            # MLP for classification
            self._fc = FCLayers(self._head_dim, self.args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.43216, 0.394666, 0.37645), std = (0.22803, 0.22145, 0.216989)) #Kinetics 400 
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats

            # X3D expects (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)

            # Ignore X3D classification head: use only blocks 0..4
            for i, block in enumerate(self._features.blocks):
                if i == 5:
                    break
                x = block(x) # Shape: [B, D, T, H', W']

            # Pool only spatial dims, keep temporal dim
            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # [B, D, T, 1, 1]
            x = x.squeeze(-1).squeeze(-1)                    # [B, D, T]
            im_feat = x.permute(0, 2, 1)                     # [B, T, D]

            # Apply model to encode temporality
            if self.args.temporal_model in ["transformer", "gru"]:
                im_feat = self._temporal_model(im_feat)

            #MLP
            im_feat = self._fc(im_feat) #B, T, num_classes+1

            return im_feat 
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

        if args.class_weights_type == "stats":
            stats = load_json(os.path.join(args.save_dir, 'dataset_statistics.json'))
            train_counts = stats["train"]["counts"]
            self.class_weights = build_class_weights(train_counts).to(self.device)
        elif args.class_weights_type == "hardcoded":
            self.class_weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)
        else:
            self.class_weights = torch.tensor([1.0] + [1.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        if args.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=self.class_weights,
                gamma=args.gamma
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                reduction="mean"
            )

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T

                    if self._args.oversample_actions:
                        pos_mask = label != 0
                        neg_mask = label == 0

                        pos_idx = torch.where(pos_mask)[0]
                        neg_idx = torch.where(neg_mask)[0]

                        num_pos = pos_idx.numel()

                        if num_pos > 0:
                            neg_ratio = self._args.oversampling_ratio
                            num_neg_keep = min(neg_ratio * num_pos, neg_idx.numel())
                            perm = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg_keep]
                            neg_idx = neg_idx[perm]
                            keep_idx = torch.cat([pos_idx, neg_idx], dim=0)
                        else:
                            # fallback
                            num_neg_keep = min(256, neg_idx.numel())
                            perm = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg_keep]
                            keep_idx = neg_idx[perm]

                        pred = pred[keep_idx]
                        label = label[keep_idx]

                    loss = self.criterion(pred, label)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()

    def get_stats(self):
        # Dummy input (shape from arguments)
        resolution_str = self._args.frame_dir.rstrip('/').split('/')[-1]
        width, height = map(int, resolution_str.split('x'))
        dummy_input = torch.randn(1, self._args.clip_len, 3, height, width).to(self.device)
        
        # Model parameters and complexity
        return clever_format(profile(self._model, inputs=(dummy_input, ), verbose=False), "%.2f")