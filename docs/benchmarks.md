# Benchmarks

The benchmark suite is intentionally small and reproducible. It compares Gaussmix against scikit-learn-style baselines on synthetic clustered datasets and, when available, local UCI-style datasets.

## Run

```bash
source gmm/bin/activate
python benchmarks/generate_benchmarks.py --samples 1000 --features 10 --components 3
```

## Metrics

- fit time in seconds;
- mean log-likelihood;
- BIC/AIC where available;
- clustering agreement on synthetic labeled data.

## Interpretation

Benchmark results are machine- and BLAS-dependent. Compare relative behavior under the same Python environment rather than treating absolute times as universal.
