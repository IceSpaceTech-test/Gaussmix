import numpy as np
import pytest

from gaussian_mixture.covariance import compute_covariances, validate_covariance_type


@pytest.mark.parametrize("kind", ["full", "tied", "diag", "spherical"])
def test_validate_covariance_type_accepts_supported_values(kind):
    assert validate_covariance_type(kind) == kind


def test_validate_covariance_type_rejects_unknown_value():
    with pytest.raises(ValueError):
        validate_covariance_type("banana")


def test_compute_diag_covariances_shape():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
    resp = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
    means = np.dot(resp.T, X) / resp.sum(axis=0)[:, None]

    covariances = compute_covariances(resp, X, means, "diag", 1e-6)

    assert covariances.shape == (2, 2)
    assert np.all(covariances > 0)
