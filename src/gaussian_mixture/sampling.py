"""Sampling utilities for Gaussian mixture models."""
from __future__ import annotations
import numpy as np
from .utils import _check_random_state


def sample(model, n_samples=1, random_state=None):
    if not hasattr(model, 'weights_'):
        raise ValueError('Model must be fitted before sampling.')
    rng = _check_random_state(random_state)
    n_components = model.weights_.shape[0]
    labels = rng.choice(n_components, size=n_samples, p=model.weights_)
    X = np.zeros((n_samples, model.means_.shape[1]), dtype=float)
    for k in range(n_components):
        nk = np.sum(labels == k)
        if nk == 0:
            continue
        mean = model.means_[k]
        cov = model.covariances_[k] if model.covariance_type != 'tied' else model.covariances_
        if model.covariance_type == 'full':
            X[labels == k] = rng.multivariate_normal(mean, cov, size=nk)
        elif model.covariance_type == 'diag':
            X[labels == k] = rng.normal(mean, np.sqrt(cov), size=(nk, mean.shape[0]))
        elif model.covariance_type == 'spherical':
            X[labels == k] = rng.normal(mean, np.sqrt(cov), size=(nk, mean.shape[0]))
        elif model.covariance_type == 'tied':
            X[labels == k] = rng.multivariate_normal(mean, cov, size=nk)
    return X, labels
