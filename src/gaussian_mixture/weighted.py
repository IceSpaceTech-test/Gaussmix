"""Sample-weight helpers for Gaussian mixture models."""
from __future__ import annotations
import numpy as np


def normalize_weights(sample_weight):
    if sample_weight is None:
        return None
    sample_weight = np.asarray(sample_weight, dtype=float)
    total = sample_weight.sum()
    if total <= 0.0:
        raise ValueError('sample_weight must have positive sum')
    return sample_weight / total
