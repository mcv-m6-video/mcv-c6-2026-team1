"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

# Computational reporting
from thop import profile, clever_format

#Local imports
from model.modules import BaseRGBModel, FCLayers, step


# Temporal operations
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout, mlp_dim):
        super().__init__()
        self.multi_head = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.multi_head(
            self.norm1(x), self.norm1(x), self.norm1(x)
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, depth, attn_dropout, mlp_dim):
        super().__init__()
        self.temporal_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer_blocks = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, attn_dropout, mlp_dim)
            for _ in range(depth)
        ])

    def forward(self, x):
        B, T, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, T+1, D)
        x = x + self.temporal_embed[:, :T+1, :]

        for block in self.transformer_blocks:
            x = block(x)

        return x[:, 0]
    

class TemporalMaxPool(nn.Module):
    def forward(self, x):
        return torch.max(x, dim=1)[0]
    

def build_temporal_module(args, embed_dim):
    if args.temp_aggregation.lower() == "max":
        return TemporalMaxPool()
    if args.temp_aggregation.lower() == "transformer":
        return TemporalTransformer(
            seq_len=args.clip_len,
            embed_dim=embed_dim,
            num_heads=args.attention_heads,
            depth=args.transformer_depth,
            attn_dropout=args.transformer_dropout,
            mlp_dim=args.transformer_mlp_dim
        )
    
    raise NotImplementedError(args.temp_aggregation)

# Backbone
def build_backbone(feature_arch):
    if feature_arch.startswith(('rny002', 'rny004', 'rny008')):
        model_name = {
            'rny002': 'regnety_002',
            'rny004': 'regnety_004',
            'rny008': 'regnety_008',
        }[feature_arch.rsplit('_', 1)[0]]

        features = timm.create_model(model_name, pretrained=True)
        feat_dim = features.head.fc.in_features
        features.head.fc = nn.Identity()
        return features, feat_dim

    raise NotImplementedError(feature_arch)


class Model(BaseRGBModel):
    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self.args = args
            self._feature_arch = self.args.feature_arch

            # Build backone
            self._features, self._d = build_backbone(self._feature_arch)

            # How to aggregate latents
            self.temporal_module = build_temporal_module(
                args=self.args,
                embed_dim=self._d
            )

            # MLP for classification
            self._fc = FCLayers(self._d, self.args.num_classes)

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
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

            self.use_auxiliary = getattr(self.args, 'use_auxiliary', False)
            self.aux_weight = getattr(self.args, 'aux_weight', 0.1)

            if self.use_auxiliary:
                self._aux_event_head = nn.Linear(self._d, 1)
            else:
                self._aux_event_head = None

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
                        
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d) #B, T, D


            aux_event_logits = None
            if self.use_auxiliary:
                aux_event_logits = self._aux_event_head(im_feat).squeeze(-1)  # (B, T, 1) to (B, T) 

            # Main task
            clip_feat = self.temporal_module(im_feat)   # B, D
            main_logits = self._fc(clip_feat) 

            if self.use_auxiliary:
                return {
                    'main_logits': main_logits,
                    'aux_event_logits': aux_event_logits
                }

            return main_logits 
        
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

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).float()

                if self._args.use_auxiliary:
                    eventness = batch['eventness'].to(self.device).float()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)

                    if self._args.use_auxiliary:
                        main_logits = pred['main_logits']
                        aux_event_logits = pred['aux_event_logits']

                        main_loss = F.binary_cross_entropy_with_logits(
                            main_logits, label
                        )
                        aux_loss = F.binary_cross_entropy_with_logits(
                            aux_event_logits, eventness
                        )
                        loss = main_loss + self._args.aux_weight * aux_loss
                    else:
                        loss = F.binary_cross_entropy_with_logits(pred, label)

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

                if self._args.use_auxiliary:
                    pred = pred['main_logits']

            # apply sigmoid
            pred = torch.sigmoid(pred)
            
            return pred.cpu().numpy()

    def get_stats(self):
        # Dummy input (shape from arguments)
        resolution_str = self._args.frame_dir.rstrip('/').split('/')[-1]
        width, height = map(int, resolution_str.split('x'))
        dummy_input = torch.randn(1, self._args.clip_len, 3, height, width).to(self.device)
        
        # Model parameters and complexity
        return clever_format(profile(self._model, inputs=(dummy_input, ), verbose=False), "%.2f")
