"""Utility functions for Gaussian mixture modeling."""
from __future__ import annotations
import numpy as np
from typing import Optional, Union


def log_sum_exp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    a = np.asarray(a)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    tmp = np.exp(a - a_safe)
    s = np.sum(tmp, axis=axis, keepdims=True)
    out = np.log(s)
    if keepdims:
        return out + a_safe
    return np.squeeze(out + a_safe, axis=axis)


def log_mean_exp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    a = np.asarray(a)
    lse = log_sum_exp(a, axis=axis, keepdims=True)
    if axis is None:
        count = a.size
    else:
        count = a.shape[axis]
    out = lse - np.log(count)
    return np.squeeze(out, axis=axis)


def _check_random_state(seed: Optional[Union[int, np.random.RandomState]]) -> np.random.RandomState:
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('Invalid random_state: %r' % seed)
