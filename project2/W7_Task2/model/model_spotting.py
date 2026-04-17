"""
File containing the main model.
"""

# Standard imports
import os
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from thop import profile, clever_format

# Local imports
from model.modules import (
    BaseRGBModel,
    FCLayers,
    FocalLoss,
    TemporalTransformer,
    TemporalGRU,
    TemporalDETR,
    HungarianMatcher,
    TemporalSetCriterion,
    step
)
from util.loss_weigths import build_class_weights
from util.io import load_json


class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args=None):
            super().__init__()
            self.args = args
            self._feature_arch = self.args.feature_arch

            # Backbone
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

            elif self._feature_arch.startswith('vitbase'):
                features = timm.create_model('vit_base_patch16_224', pretrained=True)
                feat_dim = features.head.in_features
                features.head = nn.Identity()

            else:
                raise NotImplementedError(self.args.feature_arch)

            self._features = features
            self._backbone_dim = feat_dim
            self._head_dim = feat_dim

            # Optional temporal feature model BEFORE the head
            if self.args.temporal_model == "transformer":
                self._temporal_model = TemporalTransformer(
                    clip_len=self.args.clip_len,
                    embed_dim=self._backbone_dim,
                    num_heads=args.attention_heads,
                    depth=args.transformer_depth,
                    attn_dropout=args.transformer_dropout,
                    mlp_dim=args.transformer_mlp_dim,
                    proj_dropout=args.proj_dropout
                )

            elif self.args.temporal_model == "gru":
                self._temporal_model = TemporalGRU(
                    embed_dim=self._backbone_dim,
                    hidden_dim=args.gru_hidden_dim,
                    num_layers=args.gru_layers,
                    dropout=args.gru_dropout,
                    bidirectional=args.gru_bidirectional
                )
                self._head_dim = self._temporal_model.out_dim

            elif self.args.temporal_model == "none":
                pass

            else:
                raise ValueError(f"Unknown temporal_model: {self.args.temporal_model}")

            # Head
            # head_model:
            #   - "frame_fc" : original dense per-frame classifier
            #   - "detr"     : temporal DETR head
            if self.args.head_model == "frame_fc":
                self._fc = FCLayers(self._head_dim, self.args.num_classes + 1)

            elif self.args.head_model == "detr":
                self._detr = TemporalDETR(
                    embed_dim=self._head_dim,
                    num_classes=self.args.num_classes,
                    num_queries=args.num_queries,
                    num_encoder_layers=args.detr_encoder_layers,
                    num_decoder_layers=args.detr_decoder_layers,
                    num_heads=args.detr_num_heads,
                    dim_feedforward=args.detr_ffn_dim,
                    dropout=args.detr_dropout,
                    max_len=args.clip_len
                )
            else:
                raise ValueError(f"Unknown head_model: {self.args.head_model}")

            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            self.standardization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x):
            x = self.normalize(x)
            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = self.augment(x)

            x = self.standardize(x)

            if self._feature_arch.startswith('vitbase'):
                x = F.interpolate(
                    x.view(-1, channels, height, width),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                x = x.view(-1, channels, height, width)

            # backbone per-frame features
            im_feat = self._features(x).reshape(batch_size, clip_len, self._backbone_dim)

            # optional temporal model before the final head
            if self.args.temporal_model in ["transformer", "gru"]:
                im_feat = self._temporal_model(im_feat)

            if self.args.head_model == "frame_fc":
                out = self._fc(im_feat)

            elif self.args.head_model == "detr":
                out = self._detr(im_feat)

            else:
                raise ValueError(f"Unknown head_model: {self.args.head_model}")

            return out

        def normalize(self, x):
            return x / 255.

        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standardize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standardization(x[i])
            return x

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._args = args

        self._model.to(self.device)
 
        # Loss setup
        if args.head_model == "detr":
            self.matcher = HungarianMatcher(
                cost_class=args.set_cost_class,
                cost_time=args.set_cost_time
            )
            self.criterion = TemporalSetCriterion(
                num_classes=self._num_classes,
                matcher=self.matcher,
                eos_coef=args.eos_coef,
                loss_time_weight=args.loss_time_weight
            )

        else:
            # Original dense frame classification loss
            if args.class_weights_type == "stats":
                stats = load_json(os.path.join(args.save_dir, 'dataset_statistics.json'))
                train_counts = stats["train"]["counts"]
                self.class_weights = build_class_weights(train_counts).to(self.device)

            elif args.class_weights_type == "hardcoded":
                self.class_weights = torch.tensor(
                    [1.0] + [5.0] * self._num_classes,
                    dtype=torch.float32
                ).to(self.device)

            else:
                self.class_weights = torch.tensor(
                    [1.0] + [1.0] * self._num_classes,
                    dtype=torch.float32
                ).to(self.device)

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
        """
        TRAIN / VALIDATION LOOP

        Dense head expects batch:
            batch["frame"]: [B, T, C, H, W]
            batch["label"]: [B, T] with frame-wise labels

        DETR head expects batch:
            batch["frame"]: [B, T, C, H, W]
            batch["targets"]: list of length B, where each item is:
                {
                    "labels": LongTensor [N_i],  values in [1..num_classes]
                    "times":  FloatTensor [N_i], normalized in [0,1]
                }

        Notes for the future dataset:
            - each clip can have a variable number of actions N_i
            - a clip with no actions is valid:
                {
                    "labels": tensor([], dtype=torch.long),
                    "times": tensor([], dtype=torch.float32)
                }
        """
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.0

        amp_ctx = torch.cuda.amp.autocast if self.device == "cuda" else nullcontext

        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()

                with amp_ctx():
                    outputs = self._model(frame)

                    if self._args.head_model == "detr":
                        targets = []
                        for tgt in batch["targets"]:
                            targets.append({
                                "labels": tgt["labels"].to(self.device).long(),
                                "times": tgt["times"].to(self.device).float()
                            })

                        loss_dict = self.criterion(outputs, targets)
                        loss = loss_dict["loss"]

                    else:
                        label = batch['label'].to(self.device).long()

                        pred = outputs
                        pred = pred.view(-1, self._num_classes + 1)  # [B*T, C+1]
                        label = label.view(-1)                       # [B*T]

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
                                num_neg_keep = min(256, neg_idx.numel())
                                perm = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg_keep]
                                keep_idx = neg_idx[perm]

                            pred = pred[keep_idx]
                            label = label[keep_idx]

                        loss = self.criterion(pred, label)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        """
        Inference.

        Dense head output:
            softmax probabilities over frames:
                [B, T, num_classes+1]

        DETR head output:
            dict with
                scores:      [B, Q]   best action prob (excluding no-object)
                labels:      [B, Q]   predicted action labels in [1..num_classes]
                times:       [B, Q]   normalized event centers in [0,1]
                bg_scores:   [B, Q]   probability of no-object class
        """
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)

        if len(seq.shape) == 4:  # [T, C, H, W]
            seq = seq.unsqueeze(0)

        if seq.device != self.device:
            seq = seq.to(self.device)

        seq = seq.float()

        self._model.eval()
        amp_ctx = torch.cuda.amp.autocast if self.device == "cuda" else nullcontext

        with torch.no_grad():
            with amp_ctx():
                outputs = self._model(seq)

            if self._args.head_model == "detr":
                pred_logits = outputs["pred_logits"]    # [B, Q, C+1]
                pred_time = outputs["pred_time"]        # [B, Q, 1]

                probs = torch.softmax(pred_logits, dim=-1)

                bg_scores = probs[..., 0]        # no-object score
                action_probs = probs[..., 1:]    # action classes only

                scores, labels = action_probs.max(dim=-1)
                labels = labels + 1

                return {
                    "scores": scores.cpu().numpy(),
                    "labels": labels.cpu().numpy(),
                    "times": pred_time.squeeze(-1).cpu().numpy(),
                    "bg_scores": bg_scores.cpu().numpy()
                }

            else:
                pred = torch.softmax(outputs, dim=-1)
                return pred.cpu().numpy()

    def get_stats(self):
        resolution_str = self._args.frame_dir.rstrip('/').split('/')[-1]
        width, height = map(int, resolution_str.split('x'))
        dummy_input = torch.randn(1, self._args.clip_len, 3, height, width).to(self.device)

        return clever_format(profile(self._model, inputs=(dummy_input,), verbose=False), "%.2f")