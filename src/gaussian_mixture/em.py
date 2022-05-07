"""EM helper utilities for Gaussian mixture models."""
from __future__ import annotations
import numpy as np

from .covariance import compute_covariances
from .utils import log_sum_exp


def _estimate_log_prob(X: np.ndarray, means: np.ndarray, covariances: np.ndarray, covariance_type: str) -> np.ndarray:
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_samples, n_components), dtype=float)
    for k in range(n_components):
        mean = means[k]
        if covariance_type == 'full':
            cov = covariances[k]
            prec = np.linalg.inv(cov)
            log_det = np.linalg.slogdet(cov)[1]
            diff = X - mean
            log_prob[:, k] = -0.5 * (np.sum(diff @ prec * diff, axis=1) + n_features * np.log(2 * np.pi) + log_det)
        elif covariance_type == 'tied':
            cov = covariances
            prec = np.linalg.inv(cov)
            log_det = np.linalg.slogdet(cov)[1]
            diff = X - mean
            log_prob[:, k] = -0.5 * (np.sum(diff @ prec * diff, axis=1) + n_features * np.log(2 * np.pi) + log_det)
        elif covariance_type == 'diag':
            cov = covariances[k]
            log_det = np.sum(np.log(cov))
            diff = X - mean
            log_prob[:, k] = -0.5 * (np.sum((diff ** 2) / cov, axis=1) + n_features * np.log(2 * np.pi) + log_det)
        elif covariance_type == 'spherical':
            cov = covariances[k]
            log_det = n_features * np.log(cov)
            diff = X - mean
            log_prob[:, k] = -0.5 * (np.sum(diff ** 2, axis=1) / cov + n_features * np.log(2 * np.pi) + log_det)
        else:
            raise ValueError('Unsupported covariance_type: %s' % covariance_type)
    return log_prob


def e_step(X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray, covariance_type: str):
    log_prob = _estimate_log_prob(X, means, covariances, covariance_type)
    log_weight = np.log(weights)[np.newaxis, :]
    log_resp = log_prob + log_weight
    log_prob_norm = log_sum_exp(log_resp, axis=1)
    log_resp = log_resp - log_prob_norm[:, np.newaxis]
    resp = np.exp(log_resp)
    return resp, log_resp, log_prob_norm


def m_step(X: np.ndarray, resp: np.ndarray, covariance_type: str, reg_covar: float):
    nk = resp.sum(axis=0)
    weights = nk / np.sum(nk)
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = compute_covariances(resp, X, means, covariance_type, reg_covar)
    return weights, means, covariances
