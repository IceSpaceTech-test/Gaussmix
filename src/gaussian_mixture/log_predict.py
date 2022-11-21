"""Log-probability prediction helpers."""
from __future__ import annotations
import numpy as np


def predict_log_proba(model, X):
    return model.predict_log_proba(X)
