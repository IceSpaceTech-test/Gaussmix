import numpy as np

from gaussian_mixture.initialize import kmeans_init, random_init


def test_random_init_is_reproducible():
    X = np.arange(20, dtype=float).reshape(10, 2)
    first = random_init(X, 3, random_state=0)
    second = random_init(X, 3, random_state=0)
    np.testing.assert_allclose(first, second)


def test_kmeans_init_returns_component_means():
    X = np.arange(20, dtype=float).reshape(10, 2)
    means = kmeans_init(X, 2, random_state=1)
    assert means.shape == (2, 2)
