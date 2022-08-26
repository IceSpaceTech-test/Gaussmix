from __future__ import annotations

import argparse
import json
import time

import numpy as np

from gaussian_mixture import GaussianMixture, information_criteria


def make_dataset(samples: int, features: int, components: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_component = max(1, samples // components)
    chunks = []
    for k in range(components):
        center = rng.normal(loc=0.0, scale=3.0, size=features) + k
        chunks.append(rng.normal(loc=center, scale=0.6 + 0.1 * k, size=(per_component, features)))
    return np.vstack(chunks)[:samples]


def run_benchmark(samples: int, features: int, components: int, seed: int) -> dict[str, float | int]:
    X = make_dataset(samples, features, components, seed)
    model = GaussianMixture(n_components=components, covariance_type="diag", random_state=seed)
    start = time.perf_counter()
    model.fit(X)
    elapsed = time.perf_counter() - start
    return {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "components": int(components),
        "fit_seconds": round(elapsed, 6),
        "score": float(model.score(X)),
        "bic": float(information_criteria(model, X, criterion="bic")),
        "aic": float(information_criteria(model, X, criterion="aic")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Gaussmix synthetic benchmark.")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.samples, args.features, args.components, args.seed), indent=2))


if __name__ == "__main__":
    main()
