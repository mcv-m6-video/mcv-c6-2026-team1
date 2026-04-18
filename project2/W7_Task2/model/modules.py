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

    def get_optimizer(self):
        param_groups = [
            {
                "params": self._model._features.parameters(),
                "lr": self._args.backbone_learning_rate
            },
            {
                "params": self._model._detr.parameters(),
                "lr": self._args.head_learning_rate
            }
        ]

        optimizer = torch.optim.AdamW(param_groups)
        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        return optimizer, scaler

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

    # Clip gradients to prevent explosion
    if scaler is not None: scaler.unscale_(optimizer) # Unscale before clipping when using AMP
    torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], max_norm=0.1)

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

class TemporalGRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, dropout=0.0, bidirectional=False):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        # x: [B, T, D]
        y, _ = self.gru(x)   # [B, T, out_dim]
        return y
    
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x):
        T = x.size(1)
        return self.pos_embed[:, :T, :]

class TemporalDETR(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_classes,
        num_queries=10,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=250,
    ):
        super().__init__()

        self.pos_encoding = TemporalPositionalEncoding(max_len=max_len, embed_dim=embed_dim)

        # Encoder (Processes the full video encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder (Processes the learned queries)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Learned event queries (The Q potential actions)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # FFN Prediction heads
        self.class_head = nn.Linear(embed_dim, num_classes + 1)  # +1 for BACKGROUND
        self.time_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid() # Forces output to be a normalized timestamp [0, 1]
        )

    def forward(self, x):
        B = x.shape[0]

        # Encoder pass
        pos = self.pos_encoding(x)     
        memory = self.encoder(x + pos) # Add positional info before encoder

        # Decoder pass
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) 
        hs = self.decoder(tgt=query_embed, memory=memory) 

        # Heads
        pred_logits = self.class_head(hs)  
        pred_time = self.time_head(hs)    

        return {
            "pred_logits": pred_logits,
            "pred_time": pred_time
        }