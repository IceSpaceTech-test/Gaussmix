import numpy as np

from gaussian_mixture.em import e_step, m_step


def test_e_step_responsibilities_sum_to_one():
    X = np.array([[-1.0], [0.0], [1.0]])
    weights = np.array([0.5, 0.5])
    means = np.array([[-1.0], [1.0]])
    covariances = np.array([1.0, 1.0])

    resp, log_resp, log_prob_norm = e_step(X, weights, means, covariances, "spherical")

    assert resp.shape == (3, 2)
    assert log_resp.shape == (3, 2)
    assert log_prob_norm.shape == (3,)
    np.testing.assert_allclose(resp.sum(axis=1), np.ones(3))


def test_m_step_returns_normalized_weights():
    X = np.array([[-1.0], [0.0], [1.0]])
    resp = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])

    weights, means, covariances = m_step(X, resp, "spherical", 1e-6)

    np.testing.assert_allclose(weights.sum(), 1.0)
    assert means.shape == (2, 1)
    assert covariances.shape == (2,)
