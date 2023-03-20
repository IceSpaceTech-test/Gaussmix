"""Bayesian inverse-Wishart covariance prior utilities."""
from __future__ import annotations
import numpy as np
from typing import Optional


class BayesianCovariancePrior:
    def __init__(
        self,
        prior_mean: Optional[np.ndarray] = None,
        prior_df: float = 1.0,
        prior_scale: float = 1.0,
    ):
        if prior_df <= 0.0:
            raise ValueError('prior_df must be positive.')
        if prior_scale <= 0.0:
            raise ValueError('prior_scale must be positive.')
        self.prior_mean = prior_mean
        self.prior_df = float(prior_df)
        self.prior_scale = float(prior_scale)

    def update(self, covariances: np.ndarray, n_samples: int) -> np.ndarray:
        covariances = np.asarray(covariances, dtype=float)
        if covariances.ndim not in (2, 3):
            raise ValueError('covariances must be a matrix or stack of matrices.')
        if covariances.ndim == 2:
            covariances = covariances[np.newaxis, ...]
        n_components = covariances.shape[0]
        if self.prior_mean is None:
            prior_mean = np.mean(covariances, axis=0)
        else:
            prior_mean = np.asarray(self.prior_mean, dtype=float)
        posterior_covariances = np.zeros_like(covariances)
        for k in range(n_components):
            weight = n_samples + self.prior_df
            posterior_covariances[k] = (n_samples * covariances[k] + self.prior_df * prior_mean) / weight
        return posterior_covariances.squeeze()

    def shrink(self, covariances: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError('shrinkage must be between 0 and 1.')
        covariances = np.asarray(covariances, dtype=float)
        if self.prior_mean is None:
            prior_mean = np.mean(covariances, axis=0)
        else:
            prior_mean = np.asarray(self.prior_mean, dtype=float)
        if covariances.ndim == 2:
            return (1.0 - shrinkage) * covariances + shrinkage * prior_mean
        return (1.0 - shrinkage) * covariances + shrinkage * prior_mean[np.newaxis, ...]
