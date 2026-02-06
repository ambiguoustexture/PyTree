"""
PyTree - Python bindings for PTree (Panel Tree) algorithm.

This package provides a Python interface to the PTree algorithm for asset pricing
and portfolio construction, as described in Cong et al. (2025), Journal of
Financial Economics.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np

from . import _core

__all__ = ["PTree"]


def _ensure_finite(name: str, arr: np.ndarray) -> None:
    """Check if array contains NaN or Inf values."""
    if np.issubdtype(arr.dtype, np.number):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf values")


def _ensure_integer_indices(name: str, arr: np.ndarray) -> np.ndarray:
    """Ensure array contains non-negative integers."""
    if not np.issubdtype(arr.dtype, np.integer):
        if np.any(np.floor(arr) != arr):
            raise ValueError(f"{name} must be integer-valued")
    arr = arr.astype(np.int64)
    if arr.size and arr.min() < 0:
        raise ValueError(f"{name} must be non-negative")
    return arr


def _as_vector(name: str, arr: Any) -> np.ndarray:
    """Convert input to 1D NumPy array with float dtype."""
    vec = np.asarray(arr, dtype=float).reshape(-1)
    _ensure_finite(name, vec)
    return vec


def _as_fortran_matrix(name: str, arr: Any) -> np.ndarray:
    """Convert input to 2D Fortran-ordered NumPy array."""
    mat = np.asarray(arr, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    _ensure_finite(name, mat)
    return np.asfortranarray(mat)


def _map_to_zero_index(arr: np.ndarray) -> np.ndarray:
    """Map arbitrary indices to zero-based consecutive integers."""
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64)


class PTree:
    """
    PTree (Panel Tree) for asset pricing and portfolio construction.

    This class implements the PTree algorithm for growing the efficient frontier
    on panel trees, as described in Cong et al. (2025), Journal of Financial Economics.

    The algorithm partitions the universe of individual stock returns to generate
    leaf basis portfolios and stochastic discount factors (SDFs).

    Parameters
    ----------
    **params : dict
        Algorithm parameters. See Notes for details.

    Attributes
    ----------
    tree : str or None
        String representation of the fitted tree.
    json : str or None
        JSON serialization of the fitted model.
    leaf_weight : np.ndarray or None
        Portfolio weights for each leaf node.
    leaf_id : np.ndarray or None
        Leaf identifiers.
    ft : np.ndarray or None
        Stochastic discount factor (SDF).
    ft_benchmark : np.ndarray or None
        Benchmark SDF.
    portfolio : np.ndarray or None
        Leaf portfolios.
    all_criterion : list or None
        Training criterion values for each iteration.
    num_months : int or None
        Number of unique months in training data.
    num_stocks : int or None
        Number of unique stocks in training data.
    params : dict
        Current parameter values.

    Notes
    -----
    Key parameters:

    - min_leaf_size : int, default 100
        Minimum number of stocks in a leaf node.
    - max_depth : int, default 5
        Maximum depth of the tree.
    - num_iter : int, default 30
        Maximum number of boosting iterations.
    - num_cutpoints : int, default 4
        Number of cutpoint candidates for each characteristic.
    - eta : float, default 1.0
        Regularization parameter toward equal weight.
    - equal_weight : bool, default False
        Use equal weight portfolios.
    - no_H : bool, default False
        Exclude H matrix from split criterion.
    - abs_normalize : bool, default False
        Normalize leaf portfolio weight by w <- w / sum(abs(w)).
    - weighted_loss : bool, default False
        Use weighted loss function.
    - lambda_mean : float, default 0.0
        SDF mean regularization parameter.
    - lambda_cov : float, default 0.0
        SDF covariance regularization parameter.
    - lambda_mean_factor : float, default 0.0
        Factor for lambda_mean scaling.
    - lambda_cov_factor : float, default 0.0
        Factor for lambda_cov scaling.
    - early_stop : bool, default False
        Enable early stopping.
    - stop_threshold : float, default 0.95
        Early stopping threshold (should be < 1).
    - lambda_ridge : float, default 0.0
        Ridge regularization for split criterion.
    - a1 : float, default 0.05
        Regularization for tree structure (not activated).
    - a2 : float, default 1.0
        Regularization for tree structure (not activated).
    - list_K : np.ndarray or None
        Regularization matrix for tree structure.
    - random_split : bool, default False
        Use random splits instead of Sharpe ratio optimization.
    - n_jobs : int, default -1
        Number of OpenMP threads. -1 uses all available cores,
        1 ensures deterministic results.

    Examples
    --------
    Basic usage:

    >>> import numpy as np
    >>> from pytree import PTree
    >>>
    >>> rng = np.random.default_rng(0)
    >>> n_obs, n_months, n_stocks = 120, 12, 20
    >>>
    >>> X = rng.normal(size=(n_obs, 6))
    >>> Z = rng.normal(size=(n_obs, 3))
    >>> H = rng.normal(size=(n_months, 1))
    >>> months = np.repeat(np.arange(n_months), n_obs // n_months)
    >>> stocks = rng.integers(0, n_stocks, size=n_obs)
    >>> R = rng.normal(size=n_obs)
    >>> Y = R + rng.normal(scale=0.1, size=n_obs)
    >>>
    >>> model = PTree(min_leaf_size=5, max_depth=2, num_iter=2)
    >>> model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)
    >>> leaf_index = model.predict(X, months)

    With deterministic output:

    >>> model = PTree(min_leaf_size=5, max_depth=2, num_iter=2, n_jobs=1)
    >>> model.fit(R=R, Y=Y, X=X, Z=Z, H=H, stocks=stocks, months=months)

    Get portfolios and SDF:

    >>> result = model.predict(X, months, R=R)
    >>> leaf_index = result["leaf_index"]
    >>> portfolio = result["portfolio"]
    >>> ft = result["ft"]

    Serialize and deserialize:

    >>> json_str = model.to_json()
    >>> new_model = PTree.from_json(json_str)

    References
    ----------
    .. [1] Cong, L. W., Feng, G., He, J., & He, X. (2025).
           Growing the efficient frontier on panel trees.
           Journal of Financial Economics, 167, 104024.
           https://doi.org/10.1016/j.jfineco.2025.104024
    """

    def __init__(self, **params: Any) -> None:
        """
        Initialize PTree model with parameters.

        Parameters
        ----------
        **params : dict
            Model parameters. See class docstring for details.
        """
        _VALID_PARAMS = {
            "min_leaf_size", "max_depth", "num_iter", "num_cutpoints",
            "eta", "equal_weight", "no_H", "abs_normalize", "weighted_loss",
            "lambda_mean", "lambda_cov", "lambda_mean_factor", "lambda_cov_factor",
            "early_stop", "stop_threshold", "lambda_ridge", "a1", "a2",
            "list_K", "random_split", "n_jobs",
        }
        unknown = set(params) - _VALID_PARAMS
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")

        defaults = dict(
            min_leaf_size=100,
            max_depth=5,
            num_iter=30,
            num_cutpoints=4,
            eta=1.0,
            equal_weight=False,
            no_H=False,
            abs_normalize=False,
            weighted_loss=False,
            lambda_mean=0.0,
            lambda_cov=0.0,
            lambda_mean_factor=0.0,
            lambda_cov_factor=0.0,
            early_stop=False,
            stop_threshold=0.95,
            lambda_ridge=0.0,
            a1=0.05,
            a2=1.0,
            list_K=None,
            random_split=False,
            n_jobs=-1,
        )
        defaults.update(params)
        self.params = defaults
        self.tree: Optional[str] = None
        self.json: Optional[str] = None
        self.leaf_weight: Optional[np.ndarray] = None
        self.leaf_id: Optional[np.ndarray] = None
        self.ft: Optional[np.ndarray] = None
        self.ft_benchmark: Optional[np.ndarray] = None
        self.portfolio: Optional[np.ndarray] = None
        self.all_criterion: Optional[List[np.ndarray]] = None
        self.num_months: Optional[int] = None
        self.num_stocks: Optional[int] = None
        self.n_features_: Optional[int] = None

    def __repr__(self) -> str:
        fitted = self.json is not None
        parts = [f"PTree(fitted={fitted}"]
        if fitted:
            n_leaves = len(self.leaf_id) if self.leaf_id is not None else "?"
            parts.append(f"n_leaves={n_leaves}")
            if self.n_features_ is not None:
                parts.append(f"n_features={self.n_features_}")
        for k in ("min_leaf_size", "max_depth", "num_iter", "n_jobs"):
            parts.append(f"{k}={self.params[k]}")
        return ", ".join(parts) + ")"

    def _set_omp_threads(self) -> int:
        """
        Set OpenMP threads based on n_jobs parameter.

        Returns
        -------
        old_value : int
            Previous max thread count (from ``omp_get_max_threads``).
        """
        n_jobs = self.params.get("n_jobs", -1)
        old_value = _core.get_max_threads()

        if n_jobs == -1:
            # Use all available cores - don't modify
            return old_value
        elif n_jobs >= 1:
            _core.set_num_threads(n_jobs)
        else:
            raise ValueError(f"n_jobs must be -1 or a positive integer, got {n_jobs}")

        return old_value

    def _restore_omp_threads(self, old_value: int) -> None:
        """
        Restore OpenMP thread count.

        Parameters
        ----------
        old_value : int
            Previous max thread count to restore.
        """
        _core.set_num_threads(old_value)

    def fit(
        self,
        R: Any,
        Y: Any,
        X: Any,
        Z: Any,
        H: Any,
        stocks: Any,
        months: Any,
        portfolio_weight: Optional[Any] = None,
        loss_weight: Optional[Any] = None,
        first_split_var: Optional[Any] = None,
        second_split_var: Optional[Any] = None,
    ) -> "PTree":
        """
        Fit the PTree model.

        Parameters
        ----------
        R : array-like
            Vector of individual stock returns, pooled cross-section and time series.
            Shape: (n_observations,)
        Y : array-like
            Auxiliary vector used in the split criterion (Y ~ Z * F).
            Shape: (n_observations,)
        X : array-like
            Matrix of firm characteristics. Shape: (n_observations, n_features)
        Z : array-like
            Matrix of macroeconomic variables. Shape: (n_observations, n_macro)
        H : array-like
            Matrix of factors. Shape: (n_months, n_factors)
        stocks : array-like
            Vector of stock indices. Shape: (n_observations,)
        months : array-like
            Vector of month indices. Shape: (n_observations,)
        portfolio_weight : array-like, optional
            Weight for each stock when generating leaf portfolios
            (e.g., value weight or equal weight). Default is equal weight (all 1s).
        loss_weight : array-like, optional
            Weight for each return in the loss function. Default is equal weight.
        first_split_var : array-like, optional
            Column indices in X to consider for first split (0-indexed).
            Default: all columns.
        second_split_var : array-like, optional
            Column indices in X to consider for second split (0-indexed).
            Default: all columns.

        Returns
        -------
        self : PTree
            The fitted model instance.

        Raises
        ------
        ValueError
            If inputs have incorrect shapes, contain NaN, or have mismatched lengths.
            If months or stocks are not provided.

        Notes
        -----
        This method sets the following attributes after fitting:
        tree, json, leaf_weight, leaf_id, ft, ft_benchmark, portfolio,
        all_criterion, num_months, num_stocks.

        Thread control: Set n_jobs in constructor to control parallelism:

        >>> model = PTree(n_jobs=4)   # Use 4 threads
        >>> model = PTree(n_jobs=1)   # Single-threaded (deterministic)
        >>> model = PTree(n_jobs=-1)  # Use all cores (default)
        """
        # Set OpenMP threads
        old_threads = self._set_omp_threads()

        try:
            R_vec = _as_vector("R", R)
            Y_vec = _as_vector("Y", Y)
            X_mat = _as_fortran_matrix("X", X)
            Z_mat = _as_fortran_matrix("Z", Z)
            H_mat = _as_fortran_matrix("H", H)

            n_obs = R_vec.shape[0]
            if n_obs == 0:
                raise ValueError("R is empty; at least one observation is required")
            if Y_vec.shape[0] != n_obs:
                raise ValueError("Y must have the same length as R")
            if X_mat.shape[0] != n_obs:
                raise ValueError("X must have the same number of rows as R")
            if Z_mat.shape[0] != n_obs:
                raise ValueError("Z must have the same number of rows as R")

            if portfolio_weight is None:
                portfolio_weight_vec = np.ones(n_obs, dtype=float)
            else:
                portfolio_weight_vec = _as_vector("portfolio_weight", portfolio_weight)
                if portfolio_weight_vec.shape[0] != n_obs:
                    raise ValueError("portfolio_weight must match R length")

            if loss_weight is None:
                loss_weight_vec = np.ones(n_obs, dtype=float)
            else:
                loss_weight_vec = _as_vector("loss_weight", loss_weight)
                if loss_weight_vec.shape[0] != n_obs:
                    raise ValueError("loss_weight must match R length")

            months_vec = np.asarray(months)
            _ensure_finite("months", months_vec)
            months_vec = _ensure_integer_indices("months", months_vec)
            months_vec = months_vec.reshape(-1)
            if months_vec.shape[0] != n_obs:
                raise ValueError("months must match R length")
            months_vec = _map_to_zero_index(months_vec)

            stocks_vec = np.asarray(stocks)
            _ensure_finite("stocks", stocks_vec)
            stocks_vec = _ensure_integer_indices("stocks", stocks_vec)
            stocks_vec = stocks_vec.reshape(-1)
            if stocks_vec.shape[0] != n_obs:
                raise ValueError("stocks must match R length")
            stocks_vec = _map_to_zero_index(stocks_vec)

            unique_months = np.unique(months_vec)
            num_months = int(unique_months.shape[0])
            num_stocks = int(np.unique(stocks_vec).shape[0])

            if H_mat.shape[0] != num_months:
                raise ValueError("H must have num_months rows")

            if first_split_var is None:
                first_split_vec = np.arange(X_mat.shape[1], dtype=np.int64)
            else:
                first_split_arr = np.asarray(first_split_var)
                _ensure_finite("first_split_var", first_split_arr)
                first_split_vec = _ensure_integer_indices("first_split_var", first_split_arr)

            if second_split_var is None:
                second_split_vec = np.arange(X_mat.shape[1], dtype=np.int64)
            else:
                second_split_arr = np.asarray(second_split_var)
                _ensure_finite("second_split_var", second_split_arr)
                second_split_vec = _ensure_integer_indices("second_split_var", second_split_arr)

            if np.any(first_split_vec >= X_mat.shape[1]) or np.any(second_split_vec >= X_mat.shape[1]):
                raise ValueError("split variable indices must be within X columns")

            list_K = self.params.get("list_K")
            if list_K is None:
                list_K_mat = np.zeros((2, 2), dtype=float, order="F")
            else:
                list_K_mat = _as_fortran_matrix("list_K", list_K)

            result: Dict[str, Any] = _core.fit(
                R_vec,
                Y_vec,
                X_mat,
                Z_mat,
                H_mat,
                portfolio_weight_vec,
                loss_weight_vec,
                stocks_vec.astype(float),
                months_vec.astype(float),
                unique_months.astype(float),
                first_split_vec.astype(float),
                second_split_vec.astype(float),
                num_stocks,
                num_months,
                int(self.params["min_leaf_size"]),
                int(self.params["max_depth"]),
                int(self.params["num_iter"]),
                int(self.params["num_cutpoints"]),
                float(self.params["eta"]),
                bool(self.params["equal_weight"]),
                bool(self.params["no_H"]),
                bool(self.params["abs_normalize"]),
                bool(self.params["weighted_loss"]),
                float(self.params["lambda_mean"]),
                float(self.params["lambda_cov"]),
                float(self.params["lambda_mean_factor"]),
                float(self.params["lambda_cov_factor"]),
                bool(self.params["early_stop"]),
                float(self.params["stop_threshold"]),
                float(self.params["lambda_ridge"]),
                float(self.params["a1"]),
                float(self.params["a2"]),
                list_K_mat,
                bool(self.params["random_split"]),
            )

            self.tree = result["tree"]
            self.json = result["json"]
            self.leaf_weight = result["leaf_weight"]
            self.leaf_id = result["leaf_id"]
            self.ft = result["ft"]
            self.ft_benchmark = result["ft_benchmark"]
            self.portfolio = result["portfolio"]
            self.all_criterion = result.get("all_criterion")
            self.num_months = num_months
            self.num_stocks = num_stocks
            self.n_features_ = X_mat.shape[1]
        finally:
            # Restore OpenMP threads
            self._restore_omp_threads(old_threads)

        return self

    def predict(
        self,
        X: Any,
        months: Any,
        R: Optional[Any] = None,
        weight: Optional[Any] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict leaf indices and optionally compute portfolios.

        Parameters
        ----------
        X : array-like
            Firm characteristics matrix. Shape: (n_observations, n_features)
            Must have the same number of columns as training data.
        months : array-like
            Month indices for each observation. Shape: (n_observations,)
        R : array-like, optional
            Stock returns. If provided, also returns portfolios and SDF.
            Shape: (n_observations,)
        weight : array-like, optional
            Portfolio weights. Only used if R is provided.
            Shape: (n_observations,)

        Returns
        -------
        leaf_index : np.ndarray
            Predicted leaf node indices for each observation.
            Returned when R is None. Shape: (n_observations,)
        dict
            Dictionary with keys:
            - 'leaf_index': Leaf node indices (n_observations,)
            - 'portfolio': Leaf portfolios (n_unique_months, n_leaves)
            - 'ft': Stochastic discount factor (n_unique_months, n_weights)
            Returned when R is provided.

        Raises
        ------
        ValueError
            If model is not fitted, or if inputs have incorrect shapes.

        Examples
        --------
        Predict leaf indices only:

        >>> leaf_index = model.predict(X_test, months_test)

        Predict with portfolios and SDF:

        >>> result = model.predict(X_test, months_test, R=R_test)
        >>> leaf_index = result["leaf_index"]
        >>> portfolio = result["portfolio"]
        >>> ft = result["ft"]
        """
        if not self.json:
            raise ValueError("Model is not fitted and no json is available")
        X_mat = _as_fortran_matrix("X", X)
        if self.n_features_ is not None and X_mat.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X_mat.shape[1]} features, but model was fitted with {self.n_features_}"
            )
        months_arr = np.asarray(months)
        _ensure_finite("months", months_arr)
        months_arr = _ensure_integer_indices("months", months_arr).reshape(-1)
        if months_arr.shape[0] != X_mat.shape[0]:
            raise ValueError("months must match X rows")
        months_for_core = _map_to_zero_index(months_arr)
        leaf_index = _core.predict(X_mat, self.json, months_for_core.astype(float))
        if R is None:
            return leaf_index

        R_vec = _as_vector("R", R)
        if R_vec.shape[0] != X_mat.shape[0]:
            raise ValueError("R must match X rows")
        if weight is None:
            weight_vec = np.ones_like(R_vec, dtype=float)
        else:
            weight_vec = _as_vector("weight", weight)
            if weight_vec.shape[0] != X_mat.shape[0]:
                raise ValueError("weight must match X rows")

        if self.leaf_id is None or self.leaf_weight is None:
            raise ValueError("Model is missing leaf_id/leaf_weight; fit before using R/weight")

        unique_months = np.unique(months_arr)
        month_index = np.searchsorted(unique_months, months_arr)
        if np.any(month_index >= unique_months.size) or np.any(unique_months[month_index] != months_arr):
            raise ValueError("months contain values outside unique_months")

        leaf_id = np.asarray(self.leaf_id).reshape(-1).astype(np.int64)
        leaf_index_int = np.asarray(leaf_index).reshape(-1).astype(np.int64)
        order = np.argsort(leaf_id)
        leaf_id_sorted = leaf_id[order]
        leaf_pos = np.searchsorted(leaf_id_sorted, leaf_index_int)
        if np.any(leaf_pos >= leaf_id_sorted.size) or np.any(leaf_id_sorted[leaf_pos] != leaf_index_int):
            raise ValueError("leaf_index contains unknown leaf ids")
        leaf_cols = order[leaf_pos]

        portfolio = np.zeros((unique_months.shape[0], leaf_id.shape[0]), dtype=float)
        weight_portfolio = np.zeros_like(portfolio)
        np.add.at(portfolio, (month_index, leaf_cols), weight_vec * R_vec)
        np.add.at(weight_portfolio, (month_index, leaf_cols), weight_vec)
        np.divide(portfolio, weight_portfolio, out=portfolio, where=weight_portfolio != 0)

        leaf_weight = np.asarray(self.leaf_weight, dtype=float)
        if leaf_weight.ndim == 1:
            leaf_weight = leaf_weight.reshape(-1, 1)
        ft = portfolio @ leaf_weight
        return {"leaf_index": leaf_index, "portfolio": portfolio, "ft": ft}

    def to_json(self) -> str:
        """
        Serialize the fitted model to JSON string.

        The output includes the tree structure, leaf weights, leaf IDs,
        and model parameters so that the deserialized model can perform
        full prediction (including portfolio and SDF computation).

        Returns
        -------
        json_str : str
            JSON representation of the model.

        Raises
        ------
        ValueError
            If model is not fitted.

        Examples
        --------
        >>> json_str = model.to_json()
        >>> with open("model.json", "w") as f:
        ...     f.write(json_str)
        """
        if not self.json:
            raise ValueError("Model is not fitted and no json is available")
        payload: Dict[str, Any] = {
            "tree_json": json.loads(self.json),
            "leaf_weight": self.leaf_weight.tolist() if self.leaf_weight is not None else None,
            "leaf_id": self.leaf_id.tolist() if self.leaf_id is not None else None,
            "num_months": self.num_months,
            "num_stocks": self.num_stocks,
            "n_features": self.n_features_,
            "params": {k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in self.params.items() if v is not None},
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, json_string: str) -> "PTree":
        """
        Deserialize a model from JSON string.

        Parameters
        ----------
        json_string : str
            JSON string from ``to_json()``.  Both the new envelope format
            (with leaf_weight/leaf_id) and the legacy bare-tree format are
            accepted.

        Returns
        -------
        model : PTree
            A PTree instance with loaded model state.

        Raises
        ------
        ValueError
            If json_string is empty or invalid JSON.

        Examples
        --------
        >>> with open("model.json", "r") as f:
        ...     json_str = f.read()
        >>> model = PTree.from_json(json_str)
        >>> result = model.predict(X_test, months_test, R=R_test)
        """
        if not json_string:
            raise ValueError("json_string is empty")
        data = json.loads(json_string)

        # New envelope format produced by to_json()
        if isinstance(data, dict) and "tree_json" in data:
            params = data.get("params", {})
            obj = cls(**params)
            obj.json = json.dumps(data["tree_json"])
            if data.get("leaf_weight") is not None:
                obj.leaf_weight = np.asarray(data["leaf_weight"], dtype=float)
            if data.get("leaf_id") is not None:
                obj.leaf_id = np.asarray(data["leaf_id"], dtype=float)
            obj.num_months = data.get("num_months")
            obj.num_stocks = data.get("num_stocks")
            obj.n_features_ = data.get("n_features")
            return obj

        # Legacy format: bare tree JSON (from C++ core directly)
        obj = cls()
        obj.json = json_string
        return obj
