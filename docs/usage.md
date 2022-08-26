# Usage Guide

## Fit a Model

```python
import numpy as np
from gaussian_mixture import GaussianMixture

X = np.random.default_rng(0).normal(size=(200, 4))
model = GaussianMixture(n_components=3, covariance_type="diag", random_state=0)
model.fit(X)
print(model.predict(X[:5]))
print(model.predict_proba(X[:2]))
```

## Model Selection

```python
from gaussian_mixture import GaussianMixture, information_criteria

candidates = []
for k in range(1, 7):
    model = GaussianMixture(n_components=k, random_state=42).fit(X)
    candidates.append((k, information_criteria(model, X, criterion="bic")))
best_k = min(candidates, key=lambda item: item[1])[0]
```

## Serialization

```python
from gaussian_mixture import save_model, load_model

save_model(model, "model.joblib")
restored = load_model("model.joblib")
```

## Notebooks

Open `examples/gaussian_mixture_demo.ipynb` with the `Python 3.12 (gmm)` kernel on the Raspberry Pi environment.
