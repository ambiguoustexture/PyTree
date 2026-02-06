"""
Test OpenMP thread control via n_jobs parameter.
"""

import numpy as np
from pytree import PTree


def generate_test_data(seed=0, n_obs=120, n_months=12, n_stocks=20):
    """Generate synthetic test data."""
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n_obs, 6))
    Z = rng.normal(size=(n_obs, 3))
    H = rng.normal(size=(n_months, 1))
    months = np.repeat(np.arange(n_months), n_obs // n_months)
    stocks = rng.integers(0, n_stocks, size=n_obs)
    R = rng.normal(size=n_obs)
    Y = R + rng.normal(scale=0.1, size=n_obs)

    return R, Y, X, Z, H, stocks, months


def test_n_jobs_one():
    """Test that n_jobs=1 runs successfully."""
    R, Y, X, Z, H, stocks, months = generate_test_data()

    model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
    model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    assert model.json is not None
    assert model.leaf_weight is not None
    assert model.leaf_id is not None
    print("  test_n_jobs_one PASSED")


def test_n_jobs_multiple():
    """Test that n_jobs>1 runs successfully."""
    R, Y, X, Z, H, stocks, months = generate_test_data()

    model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=2)
    model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    assert model.json is not None
    assert model.leaf_weight is not None
    print("  test_n_jobs_multiple PASSED")


def test_n_jobs_default():
    """Test that n_jobs=-1 (default) runs successfully."""
    R, Y, X, Z, H, stocks, months = generate_test_data()

    model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=-1)
    model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    assert model.json is not None
    print("  test_n_jobs_default PASSED")


def test_n_jobs_invalid():
    """Test that invalid n_jobs raises ValueError."""
    try:
        model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=-2)
        R, Y, X, Z, H, stocks, months = generate_test_data()
        model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "n_jobs must be -1 or a positive integer" in str(e)
        print("  test_n_jobs_invalid PASSED")


def test_n_jobs_determinism():
    """Test that n_jobs=1 produces deterministic results."""
    R, Y, X, Z, H, stocks, months = generate_test_data(seed=42)

    # Fit twice with same data and n_jobs=1
    model1 = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
    model1.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    model2 = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
    model2.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    # Results should be identical
    assert np.allclose(model1.leaf_weight, model2.leaf_weight)
    assert np.allclose(model1.leaf_id, model2.leaf_id)
    assert np.allclose(model1.ft, model2.ft)
    print("  test_n_jobs_determinism PASSED")


def test_n_jobs_in_params():
    """Test that n_jobs is stored in params."""
    model = PTree(n_jobs=4)
    assert model.params["n_jobs"] == 4

    model_default = PTree()
    assert model_default.params["n_jobs"] == -1
    print("  test_n_jobs_in_params PASSED")


if __name__ == "__main__":
    print("Running n_jobs tests...")
    test_n_jobs_one()
    test_n_jobs_multiple()
    test_n_jobs_default()
    test_n_jobs_invalid()
    test_n_jobs_determinism()
    test_n_jobs_in_params()
    print("\nAll n_jobs tests passed!")
