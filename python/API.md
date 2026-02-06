# PyTree API Reference

Complete API documentation for the PyTree Python binding.

## Table of Contents

- [PTree Class](#ptree-class)
  - [Constructor](#constructor)
  - [fit](#fit)
  - [predict](#predict)
  - [to_json](#to_json)
  - [from_json](#from_json)
- [Parameters](#parameters)
- [Examples](#examples)

---

## PTree Class

```python
from pytree import PTree
```

The main class for Panel Tree algorithm implementation.

### Constructor

```python
PTree(**params)
```

Initialize a PTree model with the specified parameters.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_leaf_size` | int | 100 | Minimum number of stocks in a leaf node |
| `max_depth` | int | 5 | Maximum depth of the tree |
| `num_iter` | int | 30 | Maximum number of boosting iterations |
| `num_cutpoints` | int | 4 | Number of cutpoint candidates per characteristic |
| `eta` | float | 1.0 | Regularization parameter toward equal weight |
| `equal_weight` | bool | False | Use equal weight portfolios |
| `no_H` | bool | False | Exclude H matrix from split criterion |
| `abs_normalize` | bool | False | Normalize weights by sum of absolute values |
| `weighted_loss` | bool | False | Use weighted loss function |
| `lambda_mean` | float | 0.0 | SDF mean regularization parameter |
| `lambda_cov` | float | 0.0 | SDF covariance regularization parameter |
| `lambda_mean_factor` | float | 0.0 | Factor for lambda_mean scaling |
| `lambda_cov_factor` | float | 0.0 | Factor for lambda_cov scaling |
| `early_stop` | bool | False | Enable early stopping |
| `stop_threshold` | float | 0.95 | Early stopping threshold (< 1) |
| `lambda_ridge` | float | 0.0 | Ridge regularization for split criterion |
| `a1` | float | 0.05 | Tree structure regularization (inactive) |
| `a2` | float | 1.0 | Tree structure regularization (inactive) |
| `list_K` | np.ndarray | None | Regularization matrix for tree structure |
| `random_split` | bool | False | Use random splits instead of Sharpe optimization |
| `n_jobs` | int | -1 | Number of OpenMP threads (-1 for all cores, 1 for deterministic) |

**Returns:**

A new `PTree` instance with the specified parameters.

**Example:**

```python
# Default parameters
model = PTree()

# Custom parameters for small dataset
model = PTree(min_leaf_size=10, max_depth=3, num_iter=10)

# Deterministic single-threaded execution
model = PTree(n_jobs=1)
```

---

### fit

```python
model.fit(R, Y, X, Z, H, stocks, months, portfolio_weight=None,
          loss_weight=None, first_split_var=None, second_split_var=None)
```

Fit the PTree model to the data.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `R` | array-like | Yes | Individual stock returns, shape (n_obs,) |
| `Y` | array-like | Yes | Auxiliary vector for split criterion, shape (n_obs,) |
| `X` | array-like | Yes | Firm characteristics matrix, shape (n_obs, n_features) |
| `Z` | array-like | Yes | Macroeconomic variables matrix, shape (n_obs, n_macro) |
| `H` | array-like | Yes | Factors matrix, shape (n_months, n_factors) |
| `stocks` | array-like | Yes | Stock indices, shape (n_obs,) |
| `months` | array-like | Yes | Month indices, shape (n_obs,) |
| `portfolio_weight` | array-like | No | Portfolio weights (default: equal weight) |
| `loss_weight` | array-like | No | Loss function weights (default: equal weight) |
| `first_split_var` | array-like | No | Column indices for first split (default: all) |
| `second_split_var` | array-like | No | Column indices for second split (default: all) |

**Returns:**

`self` - The fitted model instance.

**Raises:**

- `ValueError`: If inputs have incorrect shapes, contain NaN, or have mismatched lengths
- `ValueError`: If required parameters (months, stocks) are not provided

**Example:**

```python
import numpy as np
from pytree import PTree

# Generate synthetic data
rng = np.random.default_rng(0)
n_obs, n_months, n_stocks = 120, 12, 20

X = rng.normal(size=(n_obs, 6))
Z = rng.normal(size=(n_obs, 3))
H = rng.normal(size=(n_months, 1))
months = np.repeat(np.arange(n_months), n_obs // n_months)
stocks = rng.integers(0, n_stocks, size=n_obs)
R = rng.normal(size=n_obs)
Y = R + rng.normal(scale=0.1, size=n_obs)

# Fit model
model = PTree(min_leaf_size=5, max_depth=2, num_iter=2)
model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

# Access fitted attributes
print(model.leaf_weight)
print(model.ft)
```

**Thread Control:**

The `n_jobs` parameter set in the constructor controls parallelism during fitting:

```python
# Single-threaded (deterministic)
model = PTree(n_jobs=1)
model.fit(...)

# Multi-threaded (faster)
model = PTree(n_jobs=4)
model.fit(...)
```

---

### predict

```python
model.predict(X, months, R=None, weight=None)
```

Predict leaf indices and optionally compute portfolios.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `X` | array-like | Yes | Firm characteristics, shape (n_obs, n_features) |
| `months` | array-like | Yes | Month indices, shape (n_obs,) |
| `R` | array-like | No | Stock returns for portfolio computation |
| `weight` | array-like | No | Portfolio weights (only used if R provided) |

**Returns:**

If `R` is None:
- `leaf_index`: np.ndarray of shape (n_obs,) - Leaf node indices

If `R` is provided:
- `dict` with keys:
  - `'leaf_index'`: np.ndarray of shape (n_obs,)
  - `'portfolio'`: np.ndarray of shape (n_unique_months, n_leaves)
  - `'ft'`: np.ndarray - Stochastic discount factor

**Raises:**

- `ValueError`: If model is not fitted
- `ValueError`: If inputs have incorrect shapes

**Example:**

```python
# Predict leaf indices only
leaf_index = model.predict(X_test, months_test)

# Predict with portfolios and SDF
result = model.predict(X_test, months_test, R=R_test)
leaf_index = result["leaf_index"]
portfolio = result["portfolio"]
ft = result["ft"]
```

---

### to_json

```python
model.to_json()
```

Serialize the fitted model to a JSON string.  The output includes the
tree structure, leaf weights, leaf IDs and model parameters so that
`from_json()` can restore a fully functional model.

**Returns:**

`str` - JSON representation of the model.

**Raises:**

- `ValueError`: If model is not fitted

**Example:**

```python
json_str = model.to_json()
with open("model.json", "w") as f:
    f.write(json_str)
```

---

### from_json

```python
PTree.from_json(json_string)
```

Class method to deserialize a model from JSON.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `json_string` | str | JSON string from `to_json()` |

**Returns:**

`PTree` - A PTree instance with loaded model state, including
`leaf_weight` and `leaf_id` so that `predict(X, months, R=R)` works.
Both the new envelope format (from `to_json()`) and the legacy
bare-tree JSON (from the C++ core) are accepted.

**Raises:**

- `ValueError`: If json_string is empty or invalid

**Example:**

```python
with open("model.json", "r") as f:
    json_str = f.read()

model = PTree.from_json(json_str)
result = model.predict(X_test, months_test, R=R_test)
```

---

## Attributes

After fitting, the following attributes are populated:

| Attribute | Type | Description |
|-----------|------|-------------|
| `tree` | str | String representation of the tree |
| `json` | str | JSON serialization |
| `leaf_weight` | np.ndarray | Portfolio weights for each leaf |
| `leaf_id` | np.ndarray | Leaf identifiers |
| `ft` | np.ndarray | Stochastic discount factor |
| `ft_benchmark` | np.ndarray | Benchmark SDF |
| `portfolio` | np.ndarray | Leaf portfolios |
| `all_criterion` | list | Training criterion values per iteration |
| `num_months` | int | Number of unique months in training data |
| `num_stocks` | int | Number of unique stocks in training data |
| `params` | dict | Current parameter values |

---

## Examples

### Basic Usage

```python
import numpy as np
from pytree import PTree

# Setup data
rng = np.random.default_rng(0)
n_obs, n_months, n_stocks = 120, 12, 20

X = rng.normal(size=(n_obs, 6))
Z = rng.normal(size=(n_obs, 3))
H = rng.normal(size=(n_months, 1))
months = np.repeat(np.arange(n_months), n_obs // n_months)
stocks = rng.integers(0, n_stocks, size=n_obs)
R = rng.normal(size=n_obs)
Y = R + rng.normal(scale=0.1, size=n_obs)

# Fit model
model = PTree(min_leaf_size=5, max_depth=2, num_iter=2)
model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

# Predict
leaf_index = model.predict(X, months)
```

### Deterministic Results

For reproducible results, use `n_jobs=1`:

```python
model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

# Same data + same seed + n_jobs=1 = identical results every time
```

### Save and Load Model

```python
# Save
json_str = model.to_json()
with open("model.json", "w") as f:
    f.write(json_str)

# Load
with open("model.json", "r") as f:
    json_str = f.read()
model = PTree.from_json(json_str)

# Use for prediction
leaf_index = model.predict(X_new, months_new)
```

### Get Portfolio Returns

```python
result = model.predict(X_test, months_test, R=R_test)
portfolio = result["portfolio"]  # Shape: (n_months, n_leaves)
ft = result["ft"]                # Shape: (n_months, n_weights)

# portfolio @ leaf_weight == ft
assert np.allclose(portfolio @ model.leaf_weight, ft)
```

---

## References

- Cong, L. W., Feng, G., He, J., & He, X. (2025). Growing the efficient frontier on panel trees. *Journal of Financial Economics*, 167, 104024.
