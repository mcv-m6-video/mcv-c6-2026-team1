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
from model.modules import BaseRGBModel, TemporalDETR, step
from util.loss import DETRLoss

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
            
            if args.maintain_temporal:
                print(">> Using default X3D to maintain full temporal resolution (L)")
            else:
                print(">> Added 1D conv as neck to reduce temporal reduction (L')")

            self._features = features
            self.embed_dim = 192  # X3D-S/M/L output feature dimension after block 4

            # Define a 1D Convolutional neck for L -> L' reduction if not maintaining temporal resolution
            self.temporal_aggregation = nn.Sequential(
                nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU()
            )

            # Temporal DETR model
            self._detr = TemporalDETR(
                embed_dim=self.embed_dim,
                num_classes=args.num_classes,
                num_encoder_layers=args.transformer_depth,
                num_decoder_layers=args.transformer_depth,
                num_heads=args.transformer_attention_heads,
                dim_feedforward=args.transformer_mlp_dim,
                dropout=args.transformer_dropout,
                max_len=args.clip_len
            )

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip()
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.43216, 0.394666, 0.37645), std = (0.22803, 0.22145, 0.216989)) #Kinetics 400 
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization kinetics 400 stats

            # X3D expects (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)

            # Ignore X3D classification head: use only blocks 0..4
            for i, block in enumerate(self._features.blocks):
                if i == 5:
                    break
                x = block(x) # Shape: [B, D, L', H', W']

            # Pool only spatial dims, keep temporal dim
            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # [B, D, L', 1, 1]
            x = x.squeeze(-1).squeeze(-1)                    # [B, D, L']

            if not self.args.maintain_temporal:
                # Apply learnable 1D convolution to reduce temporal dimension
                x = self.temporal_aggregation(x)

            video_feat = x.permute(0, 2, 1)                  # [B, L', D]

            # Pass through DETR ('pred_logits' + 'pred_time')
            outputs = self._detr(video_feat)

            return outputs
        
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

        # Initialize DETR loss
        self.criterion = DETRLoss(args.num_classes, args.time_weight, args.background_weight, args.use_focal_loss).to(self.device)

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
                label = [l.to(self.device) for l in batch['label']]
                timestamp = [t.to(self.device) for t in batch['timestamp']]

                with torch.cuda.amp.autocast():
                    output = self._model(frame)
                    loss = self.criterion(output, label, timestamp)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

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
                output = self._model(seq)

            # Apply softmax to logits
            probs = torch.softmax(output['pred_logits'], dim=-1)
            
            return {
                "probs": probs.cpu().numpy(),
                "timestamps": output['pred_time'].cpu().numpy()
            }

    def get_stats(self):
        # Dummy input (shape from arguments)
        resolution_str = self._args.frame_dir.rstrip('/').split('/')[-1]
        width, height = map(int, resolution_str.split('x'))
        dummy_input = torch.randn(1, self._args.clip_len, 3, height, width).to(self.device)
        
        # Model parameters and complexity
        return clever_format(profile(self._model, inputs=(dummy_input, ), verbose=False), "%.2f")