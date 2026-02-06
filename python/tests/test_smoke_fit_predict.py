import numpy as np

from pytree import PTree


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    n_obs = 120
    n_months = 12
    n_stocks = 20
    n_chars = 6
    n_z = 3

    X = rng.normal(size=(n_obs, n_chars))
    Z = rng.normal(size=(n_obs, n_z))
    H = rng.normal(size=(n_months, 1))

    months = np.repeat(np.arange(n_months), n_obs // n_months)
    stocks = rng.integers(0, n_stocks, size=n_obs)

    R = rng.normal(size=n_obs)
    Y = R + rng.normal(scale=0.1, size=n_obs)
    return R, Y, X, Z, H, stocks, months


def test_fit_predict():
    R, Y, X, Z, H, stocks, months = make_data()

    model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2)
    model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    leaf_index = model.predict(X, months)
    assert leaf_index.shape[0] == X.shape[0]

    ft = np.asarray(model.ft)
    portfolio = np.asarray(model.portfolio)
    leaf_weight = np.asarray(model.leaf_weight)
    ft2 = portfolio @ leaf_weight
    np.testing.assert_allclose(ft, ft2, rtol=1e-6, atol=1e-6)


def test_predict_with_R():
    R, Y, X, Z, H, stocks, months = make_data()

    model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2)
    model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    leaf_index = model.predict(X, months)
    leaf_weight = np.asarray(model.leaf_weight)

    out = model.predict(X, months, R=R)
    assert out["leaf_index"].shape == leaf_index.shape
    np.testing.assert_allclose(
        out["ft"], out["portfolio"] @ leaf_weight, rtol=1e-6, atol=1e-6
    )

    out_weighted = model.predict(X, months, R=R, weight=np.full_like(R, 2.0))
    np.testing.assert_allclose(
        out_weighted["portfolio"], out["portfolio"], rtol=1e-6, atol=1e-6
    )


def test_from_json_roundtrip():
    R, Y, X, Z, H, stocks, months = make_data()

    model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, num_cutpoints=2)
    model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    model2 = PTree.from_json(model.to_json())
    leaf2 = model2.predict(X, months)
    assert leaf2.shape == model.predict(X, months).shape

    # Full roundtrip: predict with R should also work after from_json
    out = model2.predict(X, months, R=R)
    np.testing.assert_allclose(
        out["ft"], out["portfolio"] @ model2.leaf_weight, rtol=1e-6, atol=1e-6
    )


def main():
    test_fit_predict()
    test_predict_with_R()
    test_from_json_roundtrip()


if __name__ == "__main__":
    main()
