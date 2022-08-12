import gaussian_mixture as gm


def test_public_api_exports_core_symbols():
    expected = [
        "GaussianMixture",
        "BayesianGaussianMixture",
        "RobustGaussianMixture",
        "SparseGaussianMixture",
        "StreamingGaussianMixture",
        "information_criteria",
        "log_sum_exp",
    ]
    for name in expected:
        assert hasattr(gm, name)
