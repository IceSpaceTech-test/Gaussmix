"""Persistence helpers for Gaussian mixture models."""
import pickle
from pathlib import Path
from typing import Any


def save_model(model: Any, path: str) -> None:
    path = Path(path)
    with path.open('wb') as handle:
        pickle.dump(model, handle)


def load_model(path: str) -> Any:
    path = Path(path)
    with path.open('rb') as handle:
        return pickle.load(handle)
