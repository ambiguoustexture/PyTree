import numpy as np

from pytree import PTree


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    n_obs = 100
    n_months = 10
    n_stocks = 15
    X = rng.normal(size=(n_obs, 5))
    Z = rng.normal(size=(n_obs, 3))
    H = rng.normal(size=(n_months, 1))
    months = np.repeat(np.arange(n_months), n_obs // n_months)
    stocks = rng.integers(0, n_stocks, size=n_obs)
    R = rng.normal(size=n_obs)
    Y = R + rng.normal(scale=0.05, size=n_obs)
    return R, Y, X, Z, H, stocks, months


def test_determinism():
    R, Y, X, Z, H, stocks, months = make_data()

    model1 = PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2, n_jobs=1)
    model1.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    model2 = PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2, n_jobs=1)
    model2.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    np.testing.assert_allclose(model1.leaf_weight, model2.leaf_weight, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model1.ft, model2.ft, rtol=1e-6, atol=1e-6)


def main():
    test_determinism()


if __name__ == "__main__":
    main()
