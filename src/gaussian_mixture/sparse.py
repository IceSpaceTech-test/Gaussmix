"""Sparse Gaussian mixture model implementation."""
from __future__ import annotations
import numpy as np
from .base import GaussianMixture
from .check_input import check_input
from .covariance import compute_covariances


def _soft_threshold(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - alpha[:, np.newaxis], 0.0)


class SparseGaussianMixture(GaussianMixture):
    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = 'diag',
        tol: float = 1e-4,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = 'kmeans',
        random_state: int | None = None,
        reg_covar: float = 1e-6,
        verbose: int = 0,
        warm_start: bool = False,
        weights_init=None,
        means_init=None,
        precisions_init=None,
        l1_penalty: float = 1.0,
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
        if l1_penalty < 0.0:
            raise ValueError('l1_penalty must be non-negative.')
        self.l1_penalty = float(l1_penalty)
        self.sparse_means_ = None

    def fit(self, X, sample_weight=None):
        X, sample_weight = check_input(X, sample_weight)
        self.n_samples_, self.n_features_in_ = X.shape
        result = super().fit(X, sample_weight=sample_weight)
        self.sparse_means_ = self.means_.copy()
        return result

    def _estimate(self, X, weights, means, covariances, sample_weight):
        lower_bound = -np.inf
        for n_iter in range(1, self.max_iter + 1):
            log_prob = self._estimate_log_prob(X, means, covariances)
            log_resp = log_prob + np.log(weights)[np.newaxis, :]
            log_prob_norm = np.logaddexp.reduce(log_resp, axis=1)
            log_resp -= log_prob_norm[:, np.newaxis]
            resp = np.exp(log_resp)
            if sample_weight is not None:
                resp *= sample_weight[:, np.newaxis]
            nk = resp.sum(axis=0)
            weights = nk / np.sum(nk)
            raw_means = np.dot(resp.T, X) / nk[:, np.newaxis]
            means = _soft_threshold(raw_means, self.l1_penalty / np.maximum(nk, 1.0))
            covariances = compute_covariances(resp, X, means, self.covariance_type, self.reg_covar)
            prev_lower_bound = lower_bound
            lower_bound = np.mean(log_prob_norm)
            if self.verbose > 1:
                print(f'Sparse EM iteration {n_iter}, lower_bound={lower_bound:.6f}')
            if abs(lower_bound - prev_lower_bound) < self.tol:
                break
        self.sparse_means_ = means.copy()
        return lower_bound, n_iter
