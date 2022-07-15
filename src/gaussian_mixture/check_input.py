"""Input validation helpers."""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


def check_input(X, sample_weight=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X must be a 2D array.')
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError('X contains NaN or infinite values.')
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.ndim != 1:
            raise ValueError('sample_weight must be one-dimensional.')
        if sample_weight.shape[0] != X.shape[0]:
            raise ValueError('sample_weight must have length n_samples.')
        if np.any(sample_weight < 0):
            raise ValueError('sample_weight must be non-negative.')
    return X, sample_weight
