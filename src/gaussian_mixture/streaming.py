"""Streaming Gaussian mixture model implementation."""
from __future__ import annotations
import numpy as np
from .base import GaussianMixture
from .check_input import check_input
from .covariance import compute_covariances


class StreamingGaussianMixture(GaussianMixture):
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
        warm_start: bool = True,
        weights_init=None,
        means_init=None,
        precisions_init=None,
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            reg_covar=reg_covar,
            verbose=verbose,
            warm_start=warm_start,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
        )
        self._n_observed_ = 0

    def partial_fit(self, X, sample_weight=None):
        X, sample_weight = check_input(X, sample_weight)
        n_samples, _ = X.shape
        if not hasattr(self, 'weights_') or self._n_observed_ == 0:
            self.fit(X, sample_weight=sample_weight)
            self._n_observed_ = n_samples
            return self

        resp = np.exp(self._estimate_log_prob_resp(X)[1])
        if sample_weight is not None:
            resp *= sample_weight[:, np.newaxis]
        nk = resp.sum(axis=0)
        total_n = self._n_observed_ + np.sum(sample_weight) if sample_weight is not None else self._n_observed_ + n_samples
        self.weights_ = (self.weights_ * self._n_observed_ + nk) / total_n
        self.means_ = (self.means_ * self._n_observed_ + np.dot(resp.T, X)) / np.maximum(self._n_observed_ + nk[:, np.newaxis], 1.0)
        self.covariances_ = compute_covariances(resp, X, self.means_, self.covariance_type, self.reg_covar)
        self._n_observed_ = total_n
        return self
