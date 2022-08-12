import numpy as np

from gaussian_mixture import GaussianMixture


def test_fit_predict_predict_proba_shapes():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=-2.0, scale=0.2, size=(20, 2)),
        rng.normal(loc=2.0, scale=0.2, size=(20, 2)),
    ])
    model = GaussianMixture(n_components=2, covariance_type="diag", random_state=0, max_iter=20)
    model.fit(X)

    labels = model.predict(X[:5])
    proba = model.predict_proba(X[:5])

    assert labels.shape == (5,)
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)
