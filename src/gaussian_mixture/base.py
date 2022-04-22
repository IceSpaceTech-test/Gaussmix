"""Core Gaussian mixture model implementation."""
from __future__ import annotations
import numpy as np
from typing import Optional, Sequence

from .check_input import check_input
from .covariance import compute_covariances, covariance_parameters, validate_covariance_type
from .em import e_step, m_step
from .initialize import kmeans_init, random_init
from .sampling import sample
from .utils import log_sum_exp, _check_random_state
from .serialization import save_model as save_model_file, load_model as load_model_file
from .params import get_params as sklearn_get_params, set_params as sklearn_set_params


class GaussianMixture:
    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = 'full',
        tol: float = 1e-4,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = 'kmeans',
        random_state: Optional[int] = None,
        reg_covar: float = 1e-6,
        verbose: int = 0,
        warm_start: bool = False,
        weights_init=None,
        means_init=None,
        precisions_init=None,
    ):
        self.n_components = int(n_components)
        self.covariance_type = validate_covariance_type(covariance_type)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.init_params = str(init_params)
        self.random_state = random_state
        self.reg_covar = float(reg_covar)
        self.verbose = int(verbose)
        self.warm_start = bool(warm_start)
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf
        self.n_samples_ = 0
        self.n_features_in_ = 0
        self.sample_weights_ = None

    def fit(self, X, sample_weight=None):
        X, sample_weight = check_input(X, sample_weight)
        n_samples, n_features = X.shape
        if self.n_components < 1:
            raise ValueError('n_components must be at least 1.')
        self.n_features_in_ = n_features
        self.n_samples_ = n_samples
        rng = _check_random_state(self.random_state)

        best_lower_bound = -np.inf
        best_params = None
        best_history = None

        for trial in range(max(1, self.n_init)):
            weights, means, covariances = self._initialize_params(X, rng, trial)
            result = self._estimate(X, weights, means, covariances, sample_weight)
            if len(result) == 3:
                lower_bound, n_iter, history = result
            else:
                lower_bound, n_iter = result
                history = []
            if self.verbose > 0:
                print(f'Initialization {trial + 1}, lower_bound={lower_bound:.6f}, n_iter={n_iter}')
            if lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                best_params = (weights.copy(), means.copy(), covariances.copy(), n_iter)
                best_history = history

        self.weights_, self.means_, self.covariances_, self.n_iter_ = best_params
        self.lower_bound_ = best_lower_bound
        self.lower_bound_history_ = best_history if best_history is not None else []
        self.converged_ = True
        return self

    def _initialize_params(self, X, rng, trial):
        n_samples, n_features = X.shape
        if self.means_init is not None:
            means = np.asarray(self.means_init, dtype=float)
        elif self.init_params == 'kmeans':
            means = kmeans_init(X, self.n_components, random_state=rng)
        else:
            means = random_init(X, self.n_components, random_state=rng)
        if means.shape != (self.n_components, n_features):
            raise ValueError('means_init must be of shape (n_components, n_features)')

        if self.weights_init is not None:
            weights = np.asarray(self.weights_init, dtype=float)
            if weights.shape != (self.n_components,):
                raise ValueError('weights_init must have shape (n_components,)')
            weights = weights / np.sum(weights)
        else:
            weights = np.full(self.n_components, 1.0 / self.n_components, dtype=float)

        covariance = np.cov(X.T) if X.shape[0] > 1 else np.eye(n_features)
        covariance.flat[::n_features + 1] += self.reg_covar
        if self.covariance_type == 'full':
            covariances = np.tile(covariance[np.newaxis, :, :], (self.n_components, 1, 1))
        elif self.covariance_type == 'tied':
            covariances = covariance
        elif self.covariance_type == 'diag':
            covariances = np.tile(np.diag(covariance), (self.n_components, 1))
        elif self.covariance_type == 'spherical':
            covariances = np.full(self.n_components, np.mean(np.diag(covariance)), dtype=float)
        else:
            raise ValueError('Unsupported covariance_type: %s' % self.covariance_type)
        return weights, means, covariances

    def _estimate(self, X, weights, means, covariances, sample_weight):
        lower_bound = -np.inf
        n_iter = 0
        history = []
        for n_iter in range(1, self.max_iter + 1):
            resp, log_resp, log_prob_norm = e_step(X, weights, means, covariances, self.covariance_type)
            if sample_weight is not None:
                resp = resp * sample_weight[:, np.newaxis]
            weights, means, covariances = m_step(X, resp, self.covariance_type, self.reg_covar)
            prev_lower_bound = lower_bound
            lower_bound = np.mean(log_prob_norm)
            history.append(lower_bound)
            if self.verbose > 1:
                print(f'E-step {n_iter}, lower_bound={lower_bound:.6f}')
            if abs(lower_bound - prev_lower_bound) < self.tol:
                break
        return lower_bound, n_iter, history

    def _estimate_log_prob(self, X, means, covariances):
        n_samples, n_features = X.shape
        n_components = self.n_components
        log_prob = np.empty((n_samples, n_components), dtype=float)
        for k in range(n_components):
            mean = means[k]
            if self.covariance_type == 'full':
                cov = covariances[k]
                prec = np.linalg.inv(cov)
                log_det = np.linalg.slogdet(cov)[1]
                diff = X - mean
                log_prob[:, k] = -0.5 * (np.sum(diff @ prec * diff, axis=1) + n_features * np.log(2 * np.pi) + log_det)
            elif self.covariance_type == 'tied':
                cov = covariances
                prec = np.linalg.inv(cov)
                log_det = np.linalg.slogdet(cov)[1]
                diff = X - mean
                log_prob[:, k] = -0.5 * (np.sum(diff @ prec * diff, axis=1) + n_features * np.log(2 * np.pi) + log_det)
            elif self.covariance_type == 'diag':
                cov = np.clip(covariances[k], 1e-12, None)
                log_det = np.sum(np.log(cov))
                diff = X - mean
                log_prob[:, k] = -0.5 * (np.sum((diff ** 2) / cov, axis=1) + n_features * np.log(2 * np.pi) + log_det)
            elif self.covariance_type == 'spherical':
                cov = np.clip(covariances[k], 1e-12, None)
                log_det = n_features * np.log(cov)
                diff = X - mean
                log_prob[:, k] = -0.5 * (np.sum(diff ** 2, axis=1) / cov + n_features * np.log(2 * np.pi) + log_det)
            else:
                raise ValueError('Unsupported covariance_type: %s' % self.covariance_type)
        return log_prob

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.argmax(log_resp, axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=float)
        _, log_resp = self._estimate_log_prob_resp(X)
        return log_resp

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)
        log_prob = self._estimate_log_prob(X, self.means_, self.covariances_)
        log_weight = np.log(self.weights_)[np.newaxis, :]
        log_prob = log_prob + log_weight
        return log_sum_exp(log_prob, axis=1)

    def score(self, X):
        return np.mean(self.score_samples(X))

    def predict_density(self, X):
        return np.exp(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        X, labels = sample(self, n_samples=n_samples, random_state=random_state)
        counts = np.bincount(labels, minlength=self.n_components).astype(float)
        self.sample_weights_ = counts / float(n_samples)
        return X, labels

    def density_quantile(self, X, q=0.05):
        densities = self.predict_density(X)
        return np.quantile(densities, q)

    @property
    def entropy(self):
        n_samples = min(max(10000, self.n_samples_), 50000)
        X, _ = sample(self, n_samples=n_samples, random_state=self.random_state)
        densities = self.predict_density(X)
        densities = np.clip(densities, 1e-12, None)
        return float(-np.mean(np.log(densities)))

    def _estimate_log_prob_resp(self, X):
        log_prob = self._estimate_log_prob(X, self.means_, self.covariances_)
        log_weight = np.log(self.weights_)[np.newaxis, :]
        log_resp = log_prob + log_weight
        log_prob_norm = log_sum_exp(log_resp, axis=1)
        return log_prob_norm, log_resp - log_prob_norm[:, np.newaxis]

    def _n_parameters(self):
        n_features = self.n_features_in_
        cov_params = covariance_parameters(self.n_components, n_features, self.covariance_type)
        return int(self.n_components - 1 + self.n_components * n_features + cov_params)

    @property
    def bic(self):
        return np.log(self.n_samples_) * self._n_parameters() - 2.0 * self.lower_bound_ * self.n_samples_

    @property
    def aic(self):
        return 2.0 * self._n_parameters() - 2.0 * self.lower_bound_ * self.n_samples_

    @property
    def aicc(self):
        k = self._n_parameters()
        n = self.n_samples_
        if n - k - 1 <= 0:
            raise ValueError('AICc is undefined for n_samples <= n_parameters + 1')
        return self.aic + 2.0 * k * (k + 1.0) / float(n - k - 1)

    def get_params(self, deep=True):
        return sklearn_get_params(self)

    def set_params(self, **params):
        return sklearn_set_params(self, **params)

    def save_model(self, path: str) -> None:
        save_model_file(self, path)

    @classmethod
    def load_model(cls, path: str):
        return load_model_file(path)
