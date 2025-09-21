"""Wrappers for level-0 models used by the stacking ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier

try:  # pragma: no cover - lightgbm is optional
    from lightgbm import LGBMClassifier  # type: ignore

    _HAS_LIGHTGBM = True
except ModuleNotFoundError:  # pragma: no cover - fall back to xgboost
    _HAS_LIGHTGBM = False

from xgboost import XGBClassifier


ArrayLike = pd.DataFrame | np.ndarray | Iterable[Sequence[float]]


def _as_2d_array(X: ArrayLike) -> np.ndarray:
    """Convert input features to a ``numpy.ndarray`` with two dimensions."""

    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(copy=True)
    else:
        arr = np.asarray(list(X) if not isinstance(X, np.ndarray) else X)

    if arr.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional.")

    return arr


def _as_1d_array(y: Iterable[int | float]) -> np.ndarray:
    arr = np.asarray(list(y), dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()

    unique = np.unique(arr)
    if unique.size < 2:
        raise ValueError("Target vector must contain at least two classes.")

    return arr


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -709, 709)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class LogisticModel:
    """Logistic regression without additional preprocessing."""

    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 500
    class_weight: str | dict[str, float] | None = None
    random_state: int | None = 42

    def __post_init__(self) -> None:
        self._model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def fit(self, X: ArrayLike, y: Iterable[int | float]) -> "LogisticModel":
        features = _as_2d_array(X)
        target = _as_1d_array(y)
        self._model.fit(features, target)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        features = _as_2d_array(X)
        return self._model.predict_proba(features)


@dataclass
class RidgeModel:
    """Ridge classifier with logistic link to obtain probabilities."""

    alpha: float = 1.0
    class_weight: str | dict[str, float] | None = None

    def __post_init__(self) -> None:
        self._model = RidgeClassifier(alpha=self.alpha, class_weight=self.class_weight)

    def fit(self, X: ArrayLike, y: Iterable[int | float]) -> "RidgeModel":
        features = _as_2d_array(X)
        target = _as_1d_array(y)
        self._model.fit(features, target)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        features = _as_2d_array(X)
        decision = self._model.decision_function(features)
        positive = _sigmoid(decision)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class GradientBoostingModel:
    """Gradient boosted decision trees with XGBoost/LightGBM backend."""

    learning_rate: float = 0.1
    max_depth: int = 3
    n_estimators: int = 200
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    random_state: int | None = 42

    def __post_init__(self) -> None:
        if _HAS_LIGHTGBM:
            self._model = LGBMClassifier(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="binary",
                random_state=self.random_state,
            )
        else:
            self._model = XGBClassifier(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                tree_method="hist",
                enable_categorical=False,
                n_jobs=1,
            )

    def fit(self, X: ArrayLike, y: Iterable[int | float]) -> "GradientBoostingModel":
        features = _as_2d_array(X)
        target = _as_1d_array(y)
        self._model.fit(features, target)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        features = _as_2d_array(X)
        proba = self._model.predict_proba(features)
        if proba.shape[1] == 1:
            proba = np.column_stack([1.0 - proba[:, 0], proba[:, 0]])
        return proba


MODEL_PARAM_GRIDS: Mapping[str, dict[str, list[float | int]]] = {
    "logistic": {
        "C": [0.1, 1.0, 5.0],
        "solver": ["lbfgs", "liblinear"],
    },
    "ridge": {
        "alpha": [0.1, 1.0, 5.0],
    },
    "gbdt": {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4],
        "n_estimators": [150, 250],
        "subsample": [0.8, 1.0],
    },
}

