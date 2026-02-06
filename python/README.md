# PyTree (Python binding)

This folder provides a minimal Python binding for the PTree C++ core. The API
is intentionally close to the R package to keep behavior consistent.

## Requirements
- CMake >= 3.18
- Armadillo headers + library (e.g., `libarmadillo-dev` on Ubuntu)
- OpenMP toolchain (g++)
- Python deps: `numpy`, `pybind11`, `scikit-build-core`

## Install
From repo root:
```bash
python3 -m pip install ./python
```

Offline-friendly install (no PyPI access):
```bash
python3 -m pip install ./python --no-build-isolation --no-deps --target python/.venv
export PYTHONPATH=python/.venv
```

Note: `python/.venv` here is just a local target directory, not a virtualenv.
If an old user-site install interferes, set `PYTHONNOUSERSITE=1` or remove the
stale `pytree` package from user site-packages.

## Quickstart
```python
import numpy as np
from pytree import PTree

rng = np.random.default_rng(0)

n_obs = 120
n_months = 12
n_stocks = 20

X = rng.normal(size=(n_obs, 6))
Z = rng.normal(size=(n_obs, 3))
H = rng.normal(size=(n_months, 1))

months = np.repeat(np.arange(n_months), n_obs // n_months)
stocks = rng.integers(0, n_stocks, size=n_obs)

R = rng.normal(size=n_obs)
Y = R + rng.normal(scale=0.1, size=n_obs)

model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2)
model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

leaf_index = model.predict(X, months)
print(leaf_index[:5])
```

## Predict with portfolio/ft output
If you provide `R` (and optional `weight`) at prediction time, the method also
returns the per-month leaf portfolios and factor:
```python
out = model.predict(X, months, R=R)
leaf_index = out["leaf_index"]
portfolio = out["portfolio"]
ft = out["ft"]
```

## Thread control (OpenMP)
Control parallelism with the `n_jobs` parameter:
```python
# Single-threaded (deterministic results)
model = PTree(n_jobs=1)

# Use 4 threads
model = PTree(n_jobs=4)

# Use all available cores (default)
model = PTree(n_jobs=-1)
```

## Quiet mode
Console output is quiet by default. Set `PyTREE_QUIET=0` to enable C++ output:
```bash
PyTREE_QUIET=0 python3 python/examples/minimal_fit_predict.py
```

## API Reference
See [API.md](API.md) for detailed API documentation including all parameters and methods.

## Tests
See `python/tests/README.md` for a minimal validation checklist and commands.

## Data note
Public data for cross-checks should live in:
`python/tests/data/P-Tree-Public-Data-main`
