"""Parameter helpers for scikit-learn compatibility."""
from __future__ import annotations
from typing import Any, Dict


def get_params(estimator, deep=True):
    params = {}
    if hasattr(estimator, '__dict__'):
        for key, value in estimator.__dict__.items():
            if key.endswith('_'):
                continue
            params[key] = value
    return params


def set_params(estimator, **params):
    for key, value in params.items():
        if not hasattr(estimator, key):
            raise ValueError('Invalid parameter %s for estimator %s.' % (key, estimator.__class__.__name__))
        setattr(estimator, key, value)
    return estimator
