"""Anomaly detection utilities for Gaussian mixture models."""
from __future__ import annotations
import numpy as np


def detect_anomalies(model, X, threshold: float = 0.99):
    X = np.asarray(X, dtype=float)
    if not hasattr(model, 'score_samples'):
        raise ValueError('Model must support score_samples(X).')
    log_likelihood = model.score_samples(X)
    scores = -log_likelihood
    cutoff = np.quantile(scores, threshold)
    return scores >= cutoff


def anomaly_scores(model, X):
    X = np.asarray(X, dtype=float)
    if not hasattr(model, 'score_samples'):
        raise ValueError('Model must support score_samples(X).')
    return -model.score_samples(X)
