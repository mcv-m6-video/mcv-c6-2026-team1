import numpy as np
import torch


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