# Configuration Guide

## Core Estimator Parameters

| Parameter | Default | Notes |
|---|---:|---|
| `n_components` | `1` | Number of mixture components. Use BIC/AIC/AICc or CV to select it. |
| `covariance_type` | `full` | One of `full`, `tied`, `diag`, or `spherical`. |
| `tol` | `1e-4` | Lower values may improve fit quality but increase runtime. |
| `max_iter` | `100` | Raise for difficult, high-dimensional, or highly correlated data. |
| `n_init` | `1` | Increase to reduce sensitivity to initialization. |
| `reg_covar` | `1e-6` | Additive covariance regularization for singular or near-singular matrices. |
| `random_state` | `None` | Set for reproducible experiments. |

## Covariance Choice

- `full`: most expressive, scales as `O(K * D^2)`.
- `tied`: shared covariance across components; useful for correlated data with fewer parameters.
- `diag`: efficient for high-dimensional data when features are weakly correlated.
- `spherical`: fastest and most constrained.

## Recommended Defaults

```toml
n_components = 3
covariance_type = "full"
tol = 1e-4
max_iter = 100
n_init = 5
reg_covar = 1e-6
random_state = 42
```
