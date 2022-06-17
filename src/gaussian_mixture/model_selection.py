"""Model selection utilities for Gaussian mixture models."""
from __future__ import annotations
import numpy as np
from typing import Literal

Criterion = Literal['bic', 'aic', 'aicc']


def information_criteria(model, X, criterion: Criterion = 'bic') -> float:
    if not hasattr(model, 'fit') or not hasattr(model, 'score'):
        raise ValueError('Model must support fit and score methods.')
    if not hasattr(model, 'n_samples_') or not hasattr(model, 'lower_bound_'):
        raise ValueError('Model must be fitted before information criteria are computed.')
    n_samples = X.shape[0]
    k = model._n_parameters()
    ll = model.lower_bound_ * n_samples
    if criterion == 'bic':
        return np.log(n_samples) * k - 2.0 * ll
    if criterion == 'aic':
        return 2.0 * k - 2.0 * ll
    if criterion == 'aicc':
        aic_value = 2.0 * k - 2.0 * ll
        if n_samples - k - 1 <= 0:
            raise ValueError('AICc is undefined for n_samples <= n_parameters + 1')
        return aic_value + 2.0 * k * (k + 1.0) / (n_samples - k - 1.0)
    raise ValueError('Unsupported criterion: %s' % criterion)
