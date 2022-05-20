"""Covariance computations for Gaussian mixture models."""
from __future__ import annotations
import numpy as np
from typing import Tuple

_VALID_COVARIANCE_TYPES = {'full', 'tied', 'diag', 'spherical'}


def validate_covariance_type(covariance_type: str) -> str:
    covariance_type = str(covariance_type).lower()
    if covariance_type not in _VALID_COVARIANCE_TYPES:
        raise ValueError('covariance_type must be one of %s' % sorted(_VALID_COVARIANCE_TYPES))
    return covariance_type


def covariance_parameters(n_components: int, n_features: int, covariance_type: str) -> int:
    covariance_type = validate_covariance_type(covariance_type)
    if covariance_type == 'full':
        return n_components * n_features * (n_features + 1) // 2
    if covariance_type == 'tied':
        return n_features * (n_features + 1) // 2
    if covariance_type == 'diag':
        return n_components * n_features
    return n_components


def compute_covariances(
    totals: np.ndarray,
    X: np.ndarray,
    means: np.ndarray,
    covariance_type: str,
    reg_covar: float,
) -> np.ndarray:
    covariance_type = validate_covariance_type(covariance_type)
    n_components = totals.shape[1]
    n_features = X.shape[1]
    if covariance_type == 'full':
        covariances = np.zeros((n_components, n_features, n_features), dtype=float)
        for k in range(n_components):
            diff = X - means[k]
            weights = totals[:, k]
            nk = np.sum(weights)
            if nk <= 0.0:
                nk = 1.0
            weighted = diff * weights[:, np.newaxis]
            covariances[k] = np.dot(weighted.T, diff) / nk
            covariances[k].flat[::n_features + 1] += reg_covar
        return covariances
    if covariance_type == 'tied':
        cov = np.zeros((n_features, n_features), dtype=float)
        total_weight = np.sum(totals)
        if total_weight <= 0.0:
            total_weight = 1.0
        for k in range(n_components):
            diff = X - means[k]
            weights = totals[:, k]
            weighted = diff * weights[:, np.newaxis]
            cov += np.dot(weighted.T, diff)
        cov /= total_weight
        cov.flat[::n_features + 1] += reg_covar
        return cov
    if covariance_type == 'diag':
        covariances = np.zeros((n_components, n_features), dtype=float)
        for k in range(n_components):
            diff = X - means[k]
            sq = diff ** 2
            weights = totals[:, k]
            nk = np.sum(weights)
            if nk <= 0.0:
                nk = 1.0
            covariances[k] = np.dot(weights, sq) / nk
            covariances[k] += reg_covar
        return covariances
    if covariance_type == 'spherical':
        covariances = np.zeros(n_components, dtype=float)
        for k in range(n_components):
            diff = X - means[k]
            sq = np.sum(diff ** 2, axis=1)
            weights = totals[:, k]
            nk = np.sum(weights)
            if nk <= 0.0:
                nk = 1.0
            covariances[k] = np.dot(weights, sq) / (nk * n_features)
            covariances[k] += reg_covar
        return covariances
    raise ValueError('Unsupported covariance_type: %s' % covariance_type)
