# Benchmarks

`generate_benchmarks.py` builds a reproducible synthetic dataset, fits Gaussmix, and prints a JSON summary. It is a lightweight benchmark intended for local regression checks.

```bash
source /home/Jayant/gaussmix/gmm/bin/activate
python benchmarks/generate_benchmarks.py --samples 1000 --features 10 --components 3
```
