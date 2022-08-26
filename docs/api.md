# API Reference

## Core

- `gaussian_mixture.GaussianMixture`: EM estimator with `fit`, `predict`, `predict_proba`, `score`, `score_samples`, `sample`, `get_params`, and `set_params`.
- `gaussian_mixture.validate_covariance_type`: validates `full`, `tied`, `diag`, and `spherical` covariance modes.
- `gaussian_mixture.log_sum_exp` and `gaussian_mixture.log_mean_exp`: numerically stable log-space reducers.

## Model Selection

- `information_criteria(model, X, criterion="bic")`: computes BIC, AIC, or AICc for a fitted model.

## Extensions

- `BayesianGaussianMixture`: variational Bayesian mixture with Dirichlet weight prior.
- `RobustGaussianMixture`: robust fitting with Huber-style reweighting.
- `SparseGaussianMixture`: high-dimensional mixture modeling with L1-regularized means.
- `StreamingGaussianMixture`: online EM updates for streams.
- `GMMKMeans`: hybrid GMM/K-Means clustering.
- `GMMEmbedder`: component-basis embedding and reconstruction helper.
- `GMMVisualizer`: plotting helper for 2D/3D data, decision regions, and likelihood traces.

## Persistence

- `save_model(model, path)` and `load_model(path)` persist trained estimators.
