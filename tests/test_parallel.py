import numpy as np

from gaussian_mixture import GaussianMixture


def test_repeated_fits_with_same_seed_are_stable():
    X = np.random.default_rng(5).normal(size=(60, 2))
    first = GaussianMixture(n_components=2, covariance_type="diag", random_state=5, max_iter=15).fit(X)
    second = GaussianMixture(n_components=2, covariance_type="diag", random_state=5, max_iter=15).fit(X)

    np.testing.assert_allclose(first.weights_, second.weights_)
