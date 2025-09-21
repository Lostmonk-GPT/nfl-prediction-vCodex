"""Baseline classification model for NFL game outcome predictions."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _to_dataframe(X: pd.DataFrame | np.ndarray | Iterable[Sequence]) -> pd.DataFrame:
    """Coerce ``X`` into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    X:
        Feature matrix represented as a DataFrame, numpy array, or iterable of
        feature rows.

    Returns
    -------
    pandas.DataFrame
        A DataFrame representation of ``X``. If ``X`` is already a DataFrame it
        is returned unchanged. When ``X`` lacks column labels the function
        generates integer-based column names.
    """

    if isinstance(X, pd.DataFrame):
        return X.copy()

    if isinstance(X, np.ndarray):
        return pd.DataFrame(X)

    return pd.DataFrame(list(X))


class BaselineClassifier:
    """Logistic-regression baseline with simple preprocessing.

    The classifier wraps a scikit-learn :class:`~sklearn.pipeline.Pipeline`
    consisting of:

    * :class:`~sklearn.preprocessing.StandardScaler` for numeric columns
    * :class:`~sklearn.preprocessing.OneHotEncoder` for categorical columns
    * :class:`~sklearn.linear_model.LogisticRegression` as the estimator

    Notes
    -----
    The implementation expects inputs as pandas DataFrames to leverage column
    dtypes for preprocessing. Arrays or iterables are accepted but immediately
    converted into a DataFrame. Categorical columns are detected via ``object``
    or ``category`` dtype; all other columns are treated as numeric.
    """

    def __init__(
        self,
        *,
        penalty: str = "l2",
        C: float = 1.0,
        max_iter: int = 500,
        solver: str = "lbfgs",
        random_state: int | None = 42,
    ) -> None:
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state

        self._pipeline: Pipeline | None = None
        self._feature_names: List[str] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray | Iterable[Sequence], y: Iterable) -> "BaselineClassifier":
        """Fit the baseline classifier.

        Parameters
        ----------
        X:
            Feature matrix containing one row per team-game observation.
        y:
            Binary outcome labels (1 for win, 0 for loss).
        """

        X_df = _to_dataframe(X)
        if X_df.empty:
            raise ValueError("Input features are empty; at least one column is required.")

        y_array = np.asarray(list(y))
        if y_array.ndim != 1:
            y_array = y_array.ravel()

        unique_labels = np.unique(y_array)
        if unique_labels.size < 2:
            raise ValueError("Target vector must contain at least two classes.")

        numeric_cols = list(X_df.select_dtypes(include=[np.number, "bool"]).columns)
        categorical_cols = list(
            X_df.select_dtypes(include=["object", "category", "string"]).columns
        )

        transformers = []
        if numeric_cols:
            transformers.append(("numeric", StandardScaler(), numeric_cols))

        if categorical_cols:
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:  # pragma: no cover - fallback for older scikit-learn
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            transformers.append(("categorical", encoder, categorical_cols))

        if not transformers:
            raise ValueError("No valid feature columns detected in input data.")

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
        )

        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_df, y_array)

        self._pipeline = pipeline
        self._feature_names = list(X_df.columns)
        return self

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray | Iterable[Sequence]
    ) -> np.ndarray:
        """Return win/loss probabilities for the provided feature matrix."""

        self._ensure_fitted()
        assert self._pipeline is not None  # For type checkers
        assert self._feature_names is not None

        X_df = _to_dataframe(X)
        missing = [col for col in self._feature_names if col not in X_df.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing required feature columns: {missing_str}")

        X_ordered = X_df[self._feature_names]
        return self._pipeline.predict_proba(X_ordered)

    def predict(self, X: pd.DataFrame | np.ndarray | Iterable[Sequence]) -> np.ndarray:
        """Predict the class label (0/1) for the provided feature matrix."""

        self._ensure_fitted()
        assert self._pipeline is not None
        assert self._feature_names is not None

        X_df = _to_dataframe(X)
        missing = [col for col in self._feature_names if col not in X_df.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing required feature columns: {missing_str}")

        X_ordered = X_df[self._feature_names]
        return self._pipeline.predict(X_ordered)

    @property
    def classes_(self) -> np.ndarray:
        """Return the classes learned during fitting."""

        self._ensure_fitted()
        assert self._pipeline is not None
        model = self._pipeline.named_steps["model"]
        return model.classes_

    def _ensure_fitted(self) -> None:
        if self._pipeline is None or self._feature_names is None:
            raise RuntimeError("BaselineClassifier must be fitted before use.")

