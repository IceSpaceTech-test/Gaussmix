# Gaussmix

Gaussmix is a compact Gaussian mixture model library for clustering, density estimation, anomaly detection, model selection, and mixture-model research workflows. The project history in `gaussian_mixture_log.md` describes its evolution from a standard EM implementation into a broader toolkit with Bayesian, robust, sparse, temporal, streaming, visualization, benchmark, and scikit-learn integration modules.

## Highlights

- EM-based Gaussian mixture fitting with `full`, `tied`, `diag`, and `spherical` covariance structures.
- Prediction, posterior probabilities, per-sample scoring, sampling, density estimation, and serialization helpers.
- Model selection with BIC, AIC, AICc, cross-validation helpers, and benchmark scripts.
- Extensions for Bayesian GMMs, robust fitting, sparse high-dimensional mixtures, Kalman/time-series mixtures, streaming updates, feature selection, anomaly detection, and visual analysis.
- A scikit-learn-style API designed for pipelines and estimator-compatible workflows.

## Installation

```bash
git clone git@github.com:IceSpaceTech-test/Gaussmix.git
cd Gaussmix
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,docs,benchmarks]"
```

## Quick Start

```python
import numpy as np
from gaussian_mixture import GaussianMixture

rng = np.random.default_rng(42)
X = np.vstack([
    rng.normal(loc=(-2, 0), scale=0.45, size=(100, 2)),
    rng.normal(loc=(2, 0), scale=0.55, size=(100, 2)),
])

model = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
model.fit(X)

labels = model.predict(X)
probabilities = model.predict_proba(X)
print(model.score(X), labels[:5], probabilities[:2])
```

## Documentation

- `docs/api.md` lists the public modules and main APIs.
- `docs/usage.md` contains practical examples.
- `docs/configs.md` explains important hyperparameters.
- `docs/benchmarks.md` describes benchmark expectations.
- `docs/troubleshooting.md` covers common fitting and environment issues.

## Development

```bash
source gmm/bin/activate
pytest
ruff check .
black --check .
```

## Release Status

The current documented release is `v1.0.0`, with `v0.1.0` marking the first public release. See `CHANGELOG.md` and `gaussian_mixture_log.md` for the full timeline.
