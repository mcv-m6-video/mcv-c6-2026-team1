"""
File containing the different modules related to the model: T-DEED.
"""

#Standard imports
import abc
import torch
import torch.nn as nn
import math

#Local imports

class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args):
        raise NotImplementedError()

    @abc.abstractmethod
    def epoch(self, loader, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, seq):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, state_dict):
        raise NotImplementedError()

class BaseRGBModel(ABCModel):

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(self._get_params(), **opt_args), \
            torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    """ Assume there is a self._model """

    def _get_params(self):
        return [p for p in self._model.parameters() if p.requires_grad]

    def state_dict(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module.state_dict()
        return self._model.state_dict()

    def load(self, state_dict):
        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)    

class FCLayers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        if len(x.shape) == 3:
            b, t, d = x.shape
            x = x.reshape(b * t, d)
            return self._fc_out(self.dropout(x)).reshape(b, t, -1)
        elif len(x.shape) == 2:
            return self._fc_out(self.dropout(x))

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

def step(optimizer, scaler, loss, lr_scheduler=None):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    if lr_scheduler is not None:
        lr_scheduler.step()
    optimizer.zero_grad()

def set_trainable_backbone(backbone, train_last_n_blocks=-1):
    """
    Controls how much of the backbone is trainable.

    train_last_n_blocks:
        -1 : train full backbone
         0 : freeze full backbone
        >0 : train last N logical blocks/stages

    ViT:
        blocks = transformer blocks (+ final norm)

    RegNet/CNN:
        blocks = [stem, s1, s2, s3, s4, head]
    """
    # Train full backbone
    if train_last_n_blocks == -1:
        for p in backbone.parameters():
            p.requires_grad = True
        return

    # Freeze full backbone
    for p in backbone.parameters():
        p.requires_grad = False

    if train_last_n_blocks == 0:
        return

    # ViT-style backbones
    if hasattr(backbone, 'blocks'):
        trainable_groups = list(backbone.blocks)

        if hasattr(backbone, 'norm'):
            trainable_groups.append(backbone.norm)

        n = min(train_last_n_blocks, len(trainable_groups))

        for module in trainable_groups[-n:]:
            for p in module.parameters():
                p.requires_grad = True
        return

    # RegNet / CNN-style backbones
    trainable_groups = []

    # Order: early -> late
    for name in ['stem', 's1', 's2', 's3', 's4', 'head']:
        if hasattr(backbone, name):
            trainable_groups.append(getattr(backbone, name))

    if len(trainable_groups) == 0:
        return

    n = min(train_last_n_blocks, len(trainable_groups))

    for module in trainable_groups[-n:]:
        for p in module.parameters():
            p.requires_grad = True