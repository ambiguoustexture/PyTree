"""
Comprehensive coverage tests for pytree.

Targets all uncovered code paths: validation errors, __repr__, edge cases
in fit/predict/to_json/from_json, and helper functions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pytree import PTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(seed=0, n_obs=120, n_months=12, n_stocks=20, n_chars=6, n_macro=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, n_chars))
    Z = rng.normal(size=(n_obs, n_macro))
    H = rng.normal(size=(n_months, 1))
    months = np.repeat(np.arange(n_months), n_obs // n_months)
    stocks = np.tile(np.arange(n_stocks), n_obs // n_stocks)
    R = rng.normal(size=n_obs)
    Y = R + rng.normal(scale=0.1, size=n_obs)
    return dict(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)


def fitted_model(seed=0):
    data = make_data(seed)
    model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
    model.fit(**data)
    return model, data


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_unfitted(self):
        m = PTree(min_leaf_size=10, max_depth=3, num_iter=5, n_jobs=1)
        r = repr(m)
        assert "fitted=False" in r
        assert "min_leaf_size=10" in r

    def test_fitted(self):
        m, _ = fitted_model()
        r = repr(m)
        assert "fitted=True" in r
        assert "n_leaves=" in r
        assert "n_features=" in r


# ---------------------------------------------------------------------------
# Unknown parameters
# ---------------------------------------------------------------------------

class TestInitValidation:
    def test_unknown_param(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            PTree(bogus_param=42)


# ---------------------------------------------------------------------------
# Helper function edge cases
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_ensure_integer_indices_non_integer_float(self):
        """Line 32: non-integer float values should raise."""
        data = make_data()
        data["months"] = np.array([0.5, 1.5, 2.5] * 40, dtype=float)
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="integer-valued"):
            m.fit(**data)

    def test_as_fortran_matrix_1d(self):
        """Line 50: 1D array passed where 2D required."""
        data = make_data()
        data["X"] = np.ones(120)  # 1D instead of 2D
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="2D array"):
            m.fit(**data)


# ---------------------------------------------------------------------------
# fit() validation paths
# ---------------------------------------------------------------------------

class TestFitValidation:
    def test_empty_R(self):
        """Line 380: empty R."""
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="empty"):
            m.fit(
                R=np.array([]),
                Y=np.array([]),
                X=np.empty((0, 3)),
                Z=np.empty((0, 2)),
                H=np.empty((0, 1)),
                stocks=np.array([]),
                months=np.array([]),
            )

    def test_X_row_mismatch(self):
        """Line 384."""
        data = make_data()
        data["X"] = data["X"][:50]
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="same number of rows"):
            m.fit(**data)

    def test_Z_row_mismatch(self):
        """Line 386."""
        data = make_data()
        data["Z"] = data["Z"][:50]
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="same number of rows"):
            m.fit(**data)

    def test_portfolio_weight_provided(self):
        """Lines 391-393: custom portfolio_weight."""
        data = make_data()
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        # Valid weight - should succeed
        m.fit(**data, portfolio_weight=np.ones(120))
        assert m.json is not None

    def test_portfolio_weight_mismatch(self):
        """Lines 391-393: wrong-length portfolio_weight."""
        data = make_data()
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="portfolio_weight must match"):
            m.fit(**data, portfolio_weight=np.ones(50))

    def test_loss_weight_provided(self):
        """Lines 398-400: custom loss_weight."""
        data = make_data()
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        m.fit(**data, loss_weight=np.ones(120))
        assert m.json is not None

    def test_loss_weight_mismatch(self):
        """Lines 398-400: wrong-length loss_weight."""
        data = make_data()
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="loss_weight must match"):
            m.fit(**data, loss_weight=np.ones(50))

    def test_months_length_mismatch(self):
        """Line 407."""
        data = make_data()
        data["months"] = data["months"][:50]
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="months must match"):
            m.fit(**data)

    def test_stocks_length_mismatch(self):
        """Line 415."""
        data = make_data()
        data["stocks"] = data["stocks"][:50]
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="stocks must match"):
            m.fit(**data)

    def test_H_rows_mismatch(self):
        """Line 423."""
        data = make_data()
        data["H"] = np.ones((5, 1))  # wrong number of months
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="num_months rows"):
            m.fit(**data)

    def test_first_split_var(self):
        """Lines 428-430: explicit first_split_var."""
        data = make_data()
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        m.fit(**data, first_split_var=np.array([0, 1, 2]))
        assert m.json is not None

    def test_second_split_var(self):
        """Lines 435-437: explicit second_split_var."""
        data = make_data()
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        m.fit(**data, second_split_var=np.array([0, 1]))
        assert m.json is not None

    def test_split_var_out_of_range(self):
        """Line 440: split var index >= X columns."""
        data = make_data()
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
        with pytest.raises(ValueError, match="within X columns"):
            m.fit(**data, first_split_var=np.array([0, 99]))

    def test_list_K_provided(self):
        """Line 446: explicit list_K matrix."""
        data = make_data()
        m = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1,
                  list_K=np.eye(3))
        m.fit(**data)
        assert m.json is not None


# ---------------------------------------------------------------------------
# predict() validation paths
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_not_fitted(self):
        """Line 557: predict on unfitted model."""
        m = PTree()
        with pytest.raises(ValueError, match="not fitted"):
            m.predict(np.ones((10, 6)), np.arange(10))

    def test_feature_mismatch(self):
        """Line 560: wrong number of features."""
        m, data = fitted_model()
        with pytest.raises(ValueError, match="features"):
            m.predict(np.ones((10, 99)), np.arange(10))

    def test_months_mismatch(self):
        """Line 567: months length != X rows."""
        m, data = fitted_model()
        with pytest.raises(ValueError, match="months must match"):
            m.predict(data["X"], np.arange(5))

    def test_R_mismatch(self):
        """Line 575: R length != X rows."""
        m, data = fitted_model()
        with pytest.raises(ValueError, match="R must match"):
            m.predict(data["X"], data["months"], R=np.ones(5))

    def test_weight_provided(self):
        """Lines 578-581: explicit weight in predict."""
        m, data = fitted_model()
        result = m.predict(data["X"], data["months"], R=data["R"],
                           weight=np.ones(120) * 2.0)
        assert "portfolio" in result

    def test_weight_mismatch(self):
        """Line 581: weight length != X rows."""
        m, data = fitted_model()
        with pytest.raises(ValueError, match="weight must match"):
            m.predict(data["X"], data["months"], R=data["R"],
                      weight=np.ones(5))

    def test_missing_leaf_id(self):
        """Line 584: model without leaf_id."""
        m, data = fitted_model()
        m.leaf_id = None
        with pytest.raises(ValueError, match="leaf_id"):
            m.predict(data["X"], data["months"], R=data["R"])

    def test_leaf_weight_1d_reshape(self):
        """Line 608: leaf_weight as 1D triggers reshape."""
        m, data = fitted_model()
        # Ensure leaf_weight is 1D
        m.leaf_weight = np.asarray(m.leaf_weight).ravel()
        result = m.predict(data["X"], data["months"], R=data["R"])
        assert "ft" in result


# ---------------------------------------------------------------------------
# to_json / from_json
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_json_not_fitted(self):
        """Line 637."""
        m = PTree()
        with pytest.raises(ValueError, match="not fitted"):
            m.to_json()

    def test_from_json_empty(self):
        """Line 680."""
        with pytest.raises(ValueError, match="empty"):
            PTree.from_json("")

    def test_from_json_legacy_format(self):
        """Lines 698-700: bare tree JSON (legacy format)."""
        m, data = fitted_model()
        # Use the raw C++ json directly (legacy format)
        legacy = m.json  # This is the bare tree json string
        restored = PTree.from_json(legacy)
        assert restored.json is not None
        # Legacy format should not have leaf_weight
        assert restored.leaf_weight is None

    def test_roundtrip_envelope(self):
        """Ensure full envelope format roundtrip works."""
        m, data = fitted_model()
        json_str = m.to_json()
        restored = PTree.from_json(json_str)
        assert restored.n_features_ == m.n_features_
        assert restored.num_months == m.num_months
        np.testing.assert_array_almost_equal(
            restored.leaf_weight, m.leaf_weight
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    passed = 0
    failed = 0
    errors = []

    # Collect all test classes and their methods
    test_classes = [
        TestRepr, TestInitValidation, TestHelpers,
        TestFitValidation, TestPredictValidation, TestSerialization,
    ]

    for cls in test_classes:
        instance = cls()
        for name in sorted(dir(instance)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(instance, name)()
                passed += 1
                print(f"  PASS  {cls.__name__}.{name}")
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{name}", e))
                print(f"  FAIL  {cls.__name__}.{name}: {e}")

    print(f"\n{passed} passed, {failed} failed")
    if errors:
        for name, e in errors:
            print(f"  {name}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
