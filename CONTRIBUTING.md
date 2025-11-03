# Contributing

Thanks for improving Gaussmix. The project is organized around small, testable changes with clear numerical behavior.

## Setup

```bash
cd /home/Jayant/gaussmix
source gmm/bin/activate
python -m pip install -e ".[dev,docs,benchmarks]"
```

## Workflow

1. Open an issue or describe the change clearly in the pull request.
2. Add focused tests for numerical behavior, edge cases, or documentation examples.
3. Run `pytest`, `ruff check .`, and `black --check .`.
4. Keep public APIs compatible with the scikit-learn-style estimator interface unless the changelog explains the migration.

## Numerical Changes

For EM, covariance, model-selection, or log-space changes, include:

- the data shape used for validation;
- random seeds;
- tolerance decisions;
- before/after likelihood or metric behavior.

## Documentation

Update `docs/api.md`, `docs/usage.md`, or `docs/troubleshooting.md` whenever user-facing behavior changes.
