"""Prediction helpers for Gaussian mixture models."""
from __future__ import annotations
import numpy as np


def predict(model, X):
    return model.predict(X)


def predict_proba(model, X):
    return model.predict_proba(X)
