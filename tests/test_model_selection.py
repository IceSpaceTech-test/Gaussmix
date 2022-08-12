import numpy as np
import pytest

from gaussian_mixture import GaussianMixture, information_criteria


def test_information_criteria_returns_finite_values():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    model = GaussianMixture(n_components=2, covariance_type="diag", random_state=3, max_iter=10).fit(X)

    assert np.isfinite(information_criteria(model, X, "bic"))
    assert np.isfinite(information_criteria(model, X, "aic"))


def test_aicc_raises_when_parameter_count_is_too_large():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5, 3))
    model = GaussianMixture(n_components=3, covariance_type="full", random_state=4, max_iter=5).fit(X)

    with pytest.raises(ValueError):
        information_criteria(model, X, "aicc")
