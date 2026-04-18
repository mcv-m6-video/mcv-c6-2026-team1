import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def build_class_weights(train_counts, max_weight=10.0):
    counts = []
    for class_name in train_counts.keys():
        counts.append(train_counts[class_name])

    counts = np.array(counts, dtype=np.float32)
    counts = np.maximum(counts, 1.0)

    raw_weights = 1.0 / np.sqrt(counts)

    # normalize so background weight becomes 1 (background is first class)
    raw_weights = raw_weights / raw_weights[0]

    # clip to avoid huge weights
    raw_weights = np.clip(raw_weights, 1.0, max_weight)

    return torch.tensor(raw_weights, dtype=torch.float32)


class DETRLoss(nn.Module):
    def __init__(self, num_classes, time_weight, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.time_weight = time_weight
        self.eos_coef = eos_coef # Weight for the BACKGROUND class
        
        class_weights = torch.ones(self.num_classes + 1)
        class_weights[0] = self.eos_coef 
        self.register_buffer('ce_weight', class_weights)

    def forward(self, outputs, labels, timestamps):
        """
        outputs: dict with 'pred_logits' [B, Q, C+1] and 'pred_time' [B, Q, 1]
        labels: list of B tensors (each shape [N_actions])
        timestamps: list of B tensors (each shape [N_actions, 1])
        """
        pred_logits = outputs['pred_logits'].float()
        pred_time = outputs['pred_time'].float()
        B, Q = pred_logits.shape[:2]

        # 1. HUNGARIAN MATCHING
        indices = []
        for i in range(B):
            if len(labels[i]) == 0:
                # No actions in this clip
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            # Extract data for the single batch item
            out_prob = pred_logits[i].softmax(-1)
            out_timestamp = pred_time[i]
            tgt_labels = labels[i]
            tgt_timestamps = timestamps[i]

            # Compute Cost: -Probability of correct class + L1 distance of time
            cost_class = -out_prob[:, tgt_labels]
            cost_time = torch.cdist(out_timestamp, tgt_timestamps, p=1)
            
            # Total cost matrix
            C = cost_class + self.time_weight * cost_time
            C = C.cpu().detach().numpy()

            # Replace NaNs with a large number to avoid issues in linear_sum_assignment
            C = np.nan_to_num(C, nan=100.0, posinf=100.0, neginf=-100.0)

            # Hungarian algorithm finds the optimal 1-to-1 matching
            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))

        # 2. LOSS COMPUTATION
        # Prepare targets for classification (default 0: BACKGROUND class)
        class_ids = torch.cat([clip_labels[matched_c] for clip_labels, (matched_r, matched_c) in zip(labels, indices)])
        target_classes = torch.zeros((B, Q), dtype=torch.int64, device=pred_logits.device)
        
        # Insert the matched classes into the target tensor
        batch_idx = torch.cat([torch.full_like(I, i) for i, (I, _) in enumerate(indices)])
        query_idx = torch.cat([I for (I, _) in indices])
        if len(batch_idx) > 0:
            target_classes[batch_idx, query_idx] = class_ids

        # Classification Loss (Cross Entropy applied to ALL queries)
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=self.ce_weight)

        # Regression Time Loss (L1 applied only to MATCHED queries)
        if len(batch_idx) > 0:
            matched_pred_time = pred_time[batch_idx, query_idx]
            matched_tgt_time = torch.cat([clip_time[matched_c] for clip_time, (matched_r, matched_c) in zip(timestamps, indices)])
            loss_time = F.l1_loss(matched_pred_time, matched_tgt_time)
        else:
            loss_time = torch.tensor(0.0, device=pred_logits.device)

        # Total Loss
        total_loss = loss_ce + self.time_weight * loss_time

        return total_loss