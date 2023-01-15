"""Bayesian Gaussian mixture model implementation."""
from __future__ import annotations
import numpy as np
from .base import GaussianMixture
from .check_input import check_input
from .covariance import compute_covariances


class BayesianGaussianMixture(GaussianMixture):
    def __init__(
        self,
        *args,
        alpha_prior: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if alpha_prior <= 0.0:
            raise ValueError('alpha_prior must be positive.')
        self.alpha_prior = float(alpha_prior)
        self.prior_concentration_ = float(alpha_prior)
        self.posterior_weights_ = None

    def fit(self, X, sample_weight=None):
        X, sample_weight = check_input(X, sample_weight)
        n_samples, _ = X.shape
        if self.n_components < 1:
            raise ValueError('n_components must be at least 1.')
        self.n_samples_ = n_samples
        return super().fit(X, sample_weight=sample_weight)

    def _estimate(self, X, weights, means, covariances, sample_weight):
        lower_bound = -np.inf
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            log_prob = self._estimate_log_prob(X, means, covariances)
            log_resp = log_prob + np.log(weights)[np.newaxis, :]
            log_prob_norm = self._log_sum_exp(log_resp, axis=1)
            log_resp -= log_prob_norm[:, np.newaxis]
            resp = np.exp(log_resp)
            if sample_weight is not None:
                resp *= sample_weight[:, np.newaxis]
            nk = resp.sum(axis=0)
            weights = nk + self.alpha_prior
            weights /= np.sum(weights)
            self.posterior_weights_ = weights.copy()
            means = np.dot(resp.T, X) / nk[:, np.newaxis]
            covariances = compute_covariances(resp, X, means, self.covariance_type, self.reg_covar)
            prev_lower_bound = lower_bound
            lower_bound = np.mean(log_prob_norm)
            if self.verbose > 1:
                print(f'E-step {n_iter}, lower_bound={lower_bound:.6f}')
            if abs(lower_bound - prev_lower_bound) < self.tol:
                break
        return lower_bound, n_iter

    def _log_sum_exp(self, a, axis=None):
        return np.logaddexp.reduce(a, axis=axis)

    def _n_parameters(self):
        return super()._n_parameters()

    def bayesian_log_likelihood(self):
        if self.posterior_weights_ is None:
            raise ValueError('Model must be fitted before computing bayesian_log_likelihood.')
        return self.lower_bound_ + np.sum((self.alpha_prior - 1.0) * np.log(self.weights_ + 1e-15))
