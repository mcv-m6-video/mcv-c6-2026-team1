"""
File containing the different modules related to the model.
"""
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment



class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args=None):
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
    """
    Base class with optimizer + checkpoint helpers.

    Assumes self._model exists.
    """

    def get_optimizer(self):
        param_groups = [
            {
                "params": self._model._features.parameters(),
                "lr": self._args.backbone_learning_rate
            }
        ]

        if hasattr(self._model, "_temporal_model"):
            param_groups.append({
                "params": self._model._temporal_model.parameters(),
                "lr": self._args.temporal_learning_rate
            })

        if hasattr(self._model, "_fc"):
            param_groups.append({
                "params": self._model._fc.parameters(),
                "lr": self._args.head_learning_rate
            })

        if hasattr(self._model, "_detr"):
            param_groups.append({
                "params": self._model._detr.parameters(),
                "lr": self._args.head_learning_rate # TODO: CHANGE THIS TO DETR LR?
            })

        optimizer = torch.optim.AdamW(param_groups)
        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        return optimizer, scaler

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


class FCLayers(nn.Module):
    """
    Input:
        x: [B, T, D] or [B, D]
    Output:
        [B, T, num_classes] or [B, num_classes]
    """

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
        else:
            raise ValueError(f"Unexpected input shape for FCLayers: {x.shape}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits:  [N, C]
        targets: [N]
        """
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
        y, _ = self.attn(z, z, z)
        x = x + self.drop_path_attn(y)
        x = x + self.drop_path_mlp(self.mlp(self.norm2(x)))
        return x


class TemporalTransformer(nn.Module):
    """
    Input:
        x: [B, T, D]
    Output:
        x: [B, T, D]
    """

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
    """
    Input:
        x: [B, T, D]
    Output:
        y: [B, T, out_dim]
    """

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
        y, _ = self.gru(x)
        return y


# DETR temporal head

class TemporalPositionalEncoding(nn.Module):
    """
    Input:
        x: [B, T, D]
    Output:
        pos: [1, T, D]
    """
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x):
        T = x.size(1)
        return self.pos_embed[:, :T, :]


class TemporalDETR(nn.Module):
    """
    Temporal DETR head for action spotting.

    Input:
        x: [B, T, D]
    Output:
        dict:
            pred_logits: [B, Q, C+1]
                - class logits for each query
                - class 0 = "no-object" 
                - classes 1..C = action classes

            pred_time: [B, Q, 1]
                - normalized event center in [0,1]
                - one prediction per query
    """

    def __init__(
        self,
        embed_dim,
        num_classes,
        num_queries=15,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=250,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.pos_encoding = TemporalPositionalEncoding(
            max_len=max_len,
            embed_dim=embed_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Learned event queries
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # Prediction heads
        self.class_head = nn.Linear(embed_dim, num_classes + 1)  # +1 for no-object
        self.time_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, D = x.shape

        pos = self.pos_encoding(x)     # [1, T, D]
        memory = self.encoder(x + pos) # [B, T, D]

        # learned queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, D]

        hs = self.decoder(
            tgt=query_embed, # updated over time
            memory=memory
        )  # [B, Q, D]

        pred_logits = self.class_head(hs)  # [B, Q, C+1]
        pred_time = self.time_head(hs)     # [B, Q, 1]

        return {
            "pred_logits": pred_logits,
            "pred_time": pred_time
        }


# Hungarian matching
class HungarianMatcher(nn.Module):
    """
    Matches DETR predictions to ground-truth events using Hungarian assignment.

    Input:
        pred_logits: [B, Q, C+1]
        pred_time:   [B, Q, 1]
    Returns:
        indices: list of length B
        each item is:
            (pred_indices, target_indices)
    """

    def __init__(self, cost_class=1.0, cost_time=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_time = cost_time

    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]          # [B, Q, C+1]
        pred_time = outputs["pred_time"].squeeze(-1)  # [B, Q]

        out_prob = pred_logits.softmax(-1)
        B, Q, _ = pred_logits.shape

        indices = []

        for b in range(B):
            tgt_labels = targets[b]["labels"]
            tgt_times = targets[b]["times"]

            # no events in this clip
            if tgt_labels.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64)
                ))
                continue

            # Classification cost
            # lower cost if the prediction gives high prob to the GT class
            cost_class = -out_prob[b][:, tgt_labels]  # [Q, N]

            # Time regression cost
            # L1 distance between predicted event centers and GT event centers
            cost_time = torch.cdist(
                pred_time[b].unsqueeze(-1),   # [Q, 1]
                tgt_times.unsqueeze(-1),      # [N, 1]
                p=1
            )  # [Q, N]

            C = self.cost_class * cost_class + self.cost_time * cost_time
            C = C.cpu()

            pred_ind, tgt_ind = linear_sum_assignment(C)

            indices.append((
                torch.as_tensor(pred_ind, dtype=torch.int64),
                torch.as_tensor(tgt_ind, dtype=torch.int64)
            ))

        return indices


# Set-based loss
class TemporalSetCriterion(nn.Module):
    """
    DETR-style set loss for temporal point spotting.

    Loss components:
        1) classification loss over ALL queries
           - matched queries -> GT label
           - unmatched queries -> no-object (class 0)

        2) time regression loss only on matched queries
           - L1 on event center
    """

    def __init__(self, num_classes, matcher, eos_coef=0.1, loss_time_weight=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_time_weight = loss_time_weight

        # Weight for the "no-object" class in CE loss
        # Lower weight helps because most queries are unmatched
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[0] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices):
        pred_logits = outputs["pred_logits"]   # [B, Q, C+1]
        B, Q, _ = pred_logits.shape

        # Default every query to class 0 = no-object
        target_classes = torch.zeros((B, Q), dtype=torch.long, device=pred_logits.device)

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]

        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),   # [B, C+1, Q]
            target_classes,
            weight=self.empty_weight
        )
        return loss_ce

    def loss_time(self, outputs, targets, indices):
        pred_time = outputs["pred_time"].squeeze(-1)  # [B, Q]

        src_times = []
        tgt_times = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_times.append(pred_time[b, src_idx])
                tgt_times.append(targets[b]["times"][tgt_idx].to(pred_time.device))

        if len(src_times) == 0:
            return pred_time.sum() * 0.0

        src_times = torch.cat(src_times)
        tgt_times = torch.cat(tgt_times)

        return F.l1_loss(src_times, tgt_times, reduction="mean")

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_time = self.loss_time(outputs, targets, indices)

        total_loss = loss_ce + self.loss_time_weight * loss_time

        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_time": loss_time,
            "indices": indices
        }