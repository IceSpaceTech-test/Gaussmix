"""Gaussian mixture embedding functionality."""
from __future__ import annotations
import numpy as np
from .base import GaussianMixture


class GMMEmbedder:
    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = 'full',
        tol: float = 1e-4,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = 'kmeans',
        random_state: int | None = None,
        reg_covar: float = 1e-6,
        verbose: int = 0,
    ):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            reg_covar=reg_covar,
            verbose=verbose,
        )

    def fit(self, X, sample_weight=None):
        self.gmm.fit(X, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        return self.transform(X)

    def transform(self, X):
        return self.gmm.predict_proba(X)

    def reconstruct(self, X):
        X = np.asarray(X, dtype=float)
        responsibilities = self.transform(X)
        return responsibilities @ self.gmm.means_

    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=float)
        reconstructed = self.reconstruct(X)
        return np.mean(np.sum((X - reconstructed) ** 2, axis=1))
