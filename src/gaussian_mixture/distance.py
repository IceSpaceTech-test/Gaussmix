"""Distribution distance utilities for Gaussian mixtures."""
from __future__ import annotations
import numpy as np

from .sampling import sample
from .utils import _check_random_state


def _symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def _mixture_mean(model) -> np.ndarray:
    return np.average(model.means_, axis=0, weights=model.weights_)


def _mixture_covariance(model) -> np.ndarray:
    n_features = model.means_.shape[1]
    mean = _mixture_mean(model)
    cov = np.zeros((n_features, n_features), dtype=float)
    for k, weight in enumerate(model.weights_):
        if model.covariance_type == 'full':
            comp_cov = model.covariances_[k]
        elif model.covariance_type == 'tied':
            comp_cov = model.covariances_
        elif model.covariance_type == 'diag':
            comp_cov = np.diag(model.covariances_[k])
        elif model.covariance_type == 'spherical':
            comp_cov = np.eye(n_features) * model.covariances_[k]
        else:
            raise ValueError('Unsupported covariance_type: %s' % model.covariance_type)
        diff = model.means_[k] - mean
        cov += weight * (comp_cov + np.outer(diff, diff))
    return cov


def _unit_directions(n_directions: int, dim: int, random_state=None) -> np.ndarray:
    rng = _check_random_state(random_state)
    directions = rng.normal(size=(n_directions, dim))
    norm = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / np.clip(norm, 1e-12, None)


def _sliced_wasserstein_distance(X: np.ndarray, Y: np.ndarray, n_projections: int = 50, random_state=None) -> float:
    if X.shape[1] != Y.shape[1]:
        raise ValueError('X and Y must have the same number of features.')
    projections = _unit_directions(n_projections, X.shape[1], random_state=random_state)
    distances = []
    for direction in projections:
        proj_x = np.sort(X @ direction)
        proj_y = np.sort(Y @ direction)
        distances.append(np.mean(np.abs(proj_x - proj_y)))
    return float(np.mean(distances))


def mixture_wasserstein_distance(X: np.ndarray, model, order: int = 2, n_projections: int = 50, random_state=None) -> float:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X must be a 2D array.')
    if not hasattr(model, 'score_samples') or not hasattr(model, 'weights_'):
        raise ValueError('Model must be a fitted Gaussian mixture.')

    if order == 1:
        n_samples = min(max(X.shape[0], 1000), 5000)
        model_samples, _ = sample(model, n_samples=n_samples, random_state=random_state)
        return _sliced_wasserstein_distance(X, model_samples, n_projections=n_projections, random_state=random_state)
    if order != 2:
        raise ValueError('Only 1- and 2-Wasserstein distances are supported.')

    empirical_mean = np.mean(X, axis=0)
    empirical_cov = np.cov(X.T)
    model_mean = _mixture_mean(model)
    model_cov = _mixture_covariance(model)

    cov12 = _symmetric_sqrt(empirical_cov) @ model_cov @ _symmetric_sqrt(empirical_cov)
    cov12 = _symmetric_sqrt(cov12)
    squared_distance = np.sum((empirical_mean - model_mean) ** 2) + np.trace(empirical_cov + model_cov - 2.0 * cov12)
    return float(np.sqrt(max(squared_distance, 0.0)))
