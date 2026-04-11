"""
File containing the different modules related to the model: T-DEED.
"""

#Standard imports
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return list(self._model.parameters())

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.long()

        log_probs = F.log_softmax(logits, dim=1)                     
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1) 
        pt = log_pt.exp()                                             

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            loss = alpha_t * loss

        return loss.mean()
    
class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_dropout,
        mlp_dim,
        proj_dropout
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout=proj_dropout)

        self.drop_path_attn = nn.Dropout(proj_dropout)
        self.drop_path_mlp = nn.Dropout(proj_dropout)

    def forward(self, x):
        z = self.norm1(x)
        y, _ = self.attn(z,z,z)
        x = x + self.drop_path_attn(y)
        x = x + self.drop_path_mlp(self.mlp(self.norm2(x)))
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, clip_len, embed_dim, num_heads, depth, attn_dropout, mlp_dim, proj_dropout):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.temporal_embed = nn.Parameter(torch.randn(1, clip_len, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, attn_dropout, mlp_dim, proj_dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        B, T, D = x.shape
        x = x + self.temporal_embed[:, :T, :]

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return x