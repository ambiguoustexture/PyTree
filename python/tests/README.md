# PyTree Tests

This folder contains minimal smoke tests for the Python bindings and checks for
the public data snapshot.

## Prereqs
- CMake >= 3.18
- Armadillo headers + library (e.g., `libarmadillo-dev` on Ubuntu)
- OpenMP toolchain (g++)
- Python deps: `numpy`, `pybind11`, `scikit-build-core`

## Build + Run (offline-friendly)
If you cannot access PyPI from this environment, install into a local target
directory and set `PYTHONPATH`:
```bash
python3 -m pip install ./python --no-build-isolation --no-deps --target python/.venv
export PYTHONPATH=python/.venv

python3 python/tests/test_smoke_fit_predict.py
python3 python/tests/test_validation.py
python3 python/tests/test_determinism.py
python3 python/tests/test_public_data.py
python3 python/tests/compare_public_data.py
```

Note: `python/.venv` here is just a local target directory, not a virtualenv.
If an old user-site install interferes, set `PYTHONNOUSERSITE=1` or remove the
stale `pytree` package from user site-packages.

## Run with pytest (optional)
Some environments auto-load pytest plugins that require network access. Disable
autoload if needed:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=python/.venv \
  python3 -m pytest -q python/tests/test_smoke_fit_predict.py
```

## Test Coverage (What Each Test Checks)
- `test_smoke_fit_predict.py`: end-to-end fit/predict on synthetic data plus
  `ft == portfolio @ leaf_weight` consistency and `from_json` round-trip.
- `test_validation.py`: input validation (NaNs, length mismatches, index ranges).
- `test_determinism.py`: deterministic outputs with `OMP_NUM_THREADS=1`.
- `test_public_data.py`: public data integrity (row/column consistency, no NaNs).
- `compare_public_data.{R,py}`: R/Python stats agreement on public data.

## R vs Python public data stats
If you have R installed, generate reference stats and compare:
```bash
Rscript python/tests/compare_public_data.R
PYTHONPATH=python/.venv python3 python/tests/compare_public_data.py
```

## Data Note
The public data snapshot lives in:
`python/tests/data/P-Tree-Public-Data-main`
