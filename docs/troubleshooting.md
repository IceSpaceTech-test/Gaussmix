# Troubleshooting

## Singular Covariance

Increase `reg_covar`, reduce `n_components`, or switch from `full` to `diag` or `tied`.

## EM Stalls

Increase `max_iter`, try more initializations with `n_init`, or use a more constrained covariance type.

## Probabilities Contain NaN or Inf

Check input data for non-finite values and normalize feature scales before fitting.

## Slow High-Dimensional Fits

Prefer `diag`, `spherical`, or sparse variants when `D > 100`; full covariance scales quadratically in feature count.

## Raspberry Pi Environment

```bash
cd /home/Jayant/gaussmix
source gmm/bin/activate
python -m pip config get global.index-url
```

The configured package index should be `https://pypi.tuna.tsinghua.edu.cn/simple`.
