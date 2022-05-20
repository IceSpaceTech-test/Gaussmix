"""Initialization routines for Gaussian mixtures."""
from __future__ import annotations
import numpy as np
from .utils import _check_random_state


def random_init(X: np.ndarray, n_components: int, random_state=None) -> np.ndarray:
    rng = _check_random_state(random_state)
    n_samples = X.shape[0]
    if n_components > n_samples:
        raise ValueError('n_components must be <= n_samples for random initialization.')
    indices = rng.choice(n_samples, size=n_components, replace=False)
    return X[indices].astype(float)


def kmeans_init(X: np.ndarray, n_components: int, random_state=None) -> np.ndarray:
    rng = _check_random_state(random_state)
    n_samples, n_features = X.shape
    if n_components > n_samples:
        raise ValueError('n_components must be <= n_samples for kmeans initialization.')
    centers = np.empty((n_components, n_features), dtype=float)
    first = rng.randint(0, n_samples)
    centers[0] = X[first]
    closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, n_components):
        probabilities = closest_dist_sq / closest_dist_sq.sum()
        index = rng.choice(n_samples, p=probabilities)
        centers[c] = X[index]
        distance_to_new_center = np.sum((X - centers[c]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, distance_to_new_center)
    return centers


def from_means(means_init, n_components: int) -> np.ndarray:
    means_init = np.asarray(means_init, dtype=float)
    if means_init.shape != (n_components, means_init.shape[1]):
        raise ValueError('means_init must be shaped (n_components, n_features).')
    return means_init
