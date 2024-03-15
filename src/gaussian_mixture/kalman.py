"""Kalman-like mixture smoothing for time-series GMM data."""
from __future__ import annotations
import numpy as np
from .base import GaussianMixture
from .covariance import validate_covariance_type
from .utils import log_sum_exp, _check_random_state


def _validate_transition_matrix(matrix: np.ndarray, n_components: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (n_components, n_components):
        raise ValueError('transition_matrix must be square with shape (K, K).')
    row_sums = matrix.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError('Each row of transition_matrix must sum to a positive value.')
    return matrix / row_sums[:, np.newaxis]


def mixture_kalman_filter(
    X,
    n_components: int = 2,
    covariance_type: str = 'full',
    transition_matrix=None,
    max_iter: int = 10,
    random_state=None,
    verbose: int = 0,
):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X must be a 2D time-series array.')
    n_samples, n_features = X.shape
    covariance_type = validate_covariance_type(covariance_type)
    rng = _check_random_state(random_state)
    if transition_matrix is None:
        transition_matrix = np.full((n_components, n_components), 1.0 / n_components, dtype=float)
    else:
        transition_matrix = _validate_transition_matrix(transition_matrix, n_components)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=5,
        n_init=2,
        random_state=rng,
        verbose=verbose,
    )
    gmm.fit(X)

    emission_log_prob = gmm._estimate_log_prob(X, gmm.means_, gmm.covariances_)
    emission_log_weighted = emission_log_prob + np.log(gmm.weights_)[np.newaxis, :]
    log_emission = emission_log_weighted - np.max(emission_log_weighted, axis=1, keepdims=True)
    emission = np.exp(log_emission)
    emission /= np.maximum(emission.sum(axis=1, keepdims=True), 1e-12)

    alpha = np.zeros((n_samples, n_components), dtype=float)
    beta = np.zeros((n_samples, n_components), dtype=float)
    alpha[0] = gmm.weights_ * emission[0]
    alpha[0] /= np.maximum(alpha[0].sum(), 1e-12)
    for t in range(1, n_samples):
        alpha[t] = (alpha[t - 1] @ transition_matrix) * emission[t]
        alpha[t] /= np.maximum(alpha[t].sum(), 1e-12)
    beta[-1] = 1.0
    for t in range(n_samples - 2, -1, -1):
        beta[t] = transition_matrix @ (emission[t + 1] * beta[t + 1])
        beta[t] /= np.maximum(beta[t].sum(), 1e-12)

    smoothed = alpha * beta
    smoothed /= np.maximum(smoothed.sum(axis=1, keepdims=True), 1e-12)
    smoothed_means = smoothed[:, :, np.newaxis] * gmm.means_[np.newaxis, :, :]

    return {
        'gmm': gmm,
        'transition_matrix': transition_matrix,
        'smoothed_weights': smoothed,
        'smoothed_means': smoothed_means,
        'component_means': gmm.means_,
        'component_covariances': gmm.covariances_,
    }
