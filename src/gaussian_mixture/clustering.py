"""Hybrid GMM/K-Means clustering implementation."""
from __future__ import annotations
import numpy as np
from .base import GaussianMixture
from .utils import _check_random_state


class GMMKMeans:
    def __init__(
        self,
        n_clusters: int = 2,
        covariance_type: str = 'diag',
        max_iter: int = 100,
        n_init: int = 10,
        threshold: float = 0.5,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.n_clusters = int(n_clusters)
        self.covariance_type = covariance_type
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.threshold = float(threshold)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.cluster_labels_ = None
        self.cluster_centers_ = None
        self.gmm_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = _check_random_state(self.random_state)
        self.gmm_ = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=rng,
            verbose=self.verbose,
        )
        self.gmm_.fit(X)
        probabilities = self.gmm_.predict_proba(X)
        labels = np.argmax(probabilities, axis=1)
        if self.threshold > 0.0:
            max_prob = np.max(probabilities, axis=1)
            labels[max_prob < self.threshold] = -1
        self.cluster_labels_ = labels
        self.cluster_centers_ = self.gmm_.means_.copy()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        probabilities = self.gmm_.predict_proba(X)
        labels = np.argmax(probabilities, axis=1)
        max_prob = np.max(probabilities, axis=1)
        labels[max_prob < self.threshold] = -1
        return labels
