import numpy as np
import pytest

from pytree import PTree


def make_data():
    rng = np.random.default_rng(0)
    n_obs = 40
    n_months = 8
    n_stocks = 10
    X = rng.normal(size=(n_obs, 4))
    Z = rng.normal(size=(n_obs, 2))
    H = rng.normal(size=(n_months, 1))
    months = np.repeat(np.arange(n_months), n_obs // n_months)
    stocks = rng.integers(0, n_stocks, size=n_obs)
    R = rng.normal(size=n_obs)
    Y = rng.normal(size=n_obs)
    return R, Y, X, Z, H, stocks, months


def test_nan_in_X():
    R, Y, X, Z, H, stocks, months = make_data()
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2).fit(
            R=R, Y=Y, X=X_bad, Z=Z, H=H, stocks=stocks, months=months
        )


def test_length_mismatch():
    R, Y, X, Z, H, stocks, months = make_data()
    with pytest.raises(ValueError, match="same length"):
        PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2).fit(
            R=R[:-1], Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months
        )


def test_negative_months():
    R, Y, X, Z, H, stocks, months = make_data()
    months_bad = months.copy().astype(float)
    months_bad[0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2).fit(
            R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months_bad
        )


def main():
    test_nan_in_X()
    test_length_mismatch()
    test_negative_months()


if __name__ == "__main__":
    main()
