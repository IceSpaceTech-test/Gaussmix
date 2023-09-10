"""Robust Gaussian mixture model implementation."""
from __future__ import annotations
import numpy as np
from .base import GaussianMixture
from .check_input import check_input
from .covariance import compute_covariances


class RobustGaussianMixture(GaussianMixture):
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
        warm_start: bool = False,
        weights_init=None,
        means_init=None,
        precisions_init=None,
        huber_delta: float = 1.345,
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
        self.huber_delta = float(huber_delta)
        self.weights_robust_ = None

    def fit(self, X, sample_weight=None):
        X, sample_weight = check_input(X, sample_weight)
        self.n_samples_, self.n_features_in_ = X.shape
        return super().fit(X, sample_weight=sample_weight)

    def _huber_weights(self, X, means, covariances):
        n_samples, n_components = X.shape[0], self.n_components
        weights = np.ones((n_samples, n_components), dtype=float)
        for k in range(n_components):
            diff = X - means[k]
            if self.covariance_type == 'full':
                cov = covariances[k]
                prec = np.linalg.inv(cov)
                dist2 = np.sum(diff @ prec * diff, axis=1)
            elif self.covariance_type == 'tied':
                cov = covariances
                prec = np.linalg.inv(cov)
                dist2 = np.sum(diff @ prec * diff, axis=1)
            elif self.covariance_type == 'diag':
                cov = covariances[k]
                dist2 = np.sum((diff ** 2) / cov, axis=1)
            elif self.covariance_type == 'spherical':
                cov = covariances[k]
                dist2 = np.sum(diff ** 2, axis=1) / cov
            else:
                raise ValueError('Unsupported covariance_type: %s' % self.covariance_type)
            dev = np.sqrt(dist2 + 1e-12)
            weights[:, k] = np.minimum(1.0, self.huber_delta / np.maximum(self.huber_delta, dev))
        return weights

    def _estimate(self, X, weights, means, covariances, sample_weight):
        lower_bound = -np.inf
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            log_prob = self._estimate_log_prob(X, means, covariances)
            log_resp = log_prob + np.log(weights)[np.newaxis, :]
            log_prob_norm = np.logaddexp.reduce(log_resp, axis=1)
            log_resp -= log_prob_norm[:, np.newaxis]
            resp = np.exp(log_resp)
            if sample_weight is not None:
                resp *= sample_weight[:, np.newaxis]
            robust_weights = self._huber_weights(X, means, covariances)
            resp *= robust_weights
            self.weights_robust_ = robust_weights
            nk = resp.sum(axis=0)
            weights = nk / np.sum(nk)
            means = np.dot(resp.T, X) / nk[:, np.newaxis]
            covariances = compute_covariances(resp, X, means, self.covariance_type, self.reg_covar)
            prev_lower_bound = lower_bound
            lower_bound = np.mean(log_prob_norm)
            if self.verbose > 1:
                print(f'Robust E-step {n_iter}, lower_bound={lower_bound:.6f}')
            if abs(lower_bound - prev_lower_bound) < self.tol:
                break
        return lower_bound, n_iter

    def outlier_detection(self, X, threshold: float = 0.99):
        X = np.asarray(X, dtype=float)
        probs = self.predict_proba(X)
        scores = 1.0 - np.max(probs, axis=1)
        cutoff = np.quantile(scores, threshold)
        return scores >= cutoff
