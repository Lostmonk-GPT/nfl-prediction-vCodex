"""Stacking ensemble utilities for combining level-0 models."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ArrayLike = pd.DataFrame | np.ndarray
ModelFactory = Callable[[], "SupportsPredictProba"]


class SupportsPredictProba:
    """Protocol-like base to satisfy type checkers for model factories."""

    def fit(self, X: ArrayLike, y: Sequence[int | float]) -> "SupportsPredictProba":  # pragma: no cover - interface shim
        raise NotImplementedError

    def predict_proba(self, X: ArrayLike) -> np.ndarray:  # pragma: no cover - interface shim
        raise NotImplementedError


def _ensure_2d(X: ArrayLike) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        values = X.to_numpy(copy=True)
    else:
        values = np.asarray(X)

    if values.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional.")
    return values


def _ensure_1d(y: Sequence[int | float]) -> np.ndarray:
    values = np.asarray(list(y), dtype=float)
    if values.ndim != 1:
        values = values.ravel()

    if np.unique(values).size < 2:
        raise ValueError("Target vector must contain at least two classes.")
    return values


def _slice_rows(X: ArrayLike, indices: np.ndarray) -> ArrayLike:
    if isinstance(X, pd.DataFrame):
        return X.iloc[indices]
    return X[indices]


def _positive_column(proba: np.ndarray) -> np.ndarray:
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("Predicted probabilities must have shape (n_samples, 2).")
    return proba[:, 1]


def generate_out_of_fold_predictions(
    base_model_factories: Mapping[str, ModelFactory],
    X: ArrayLike,
    y: Sequence[int | float],
    cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Return a DataFrame of out-of-fold probabilities for each base model."""

    if not base_model_factories:
        raise ValueError("At least one base model factory is required for stacking.")

    splits = list(cv_splits)
    if not splits:
        raise ValueError("No cross-validation splits were provided.")

    X_values = _ensure_2d(X)
    y_array = _ensure_1d(y)
    if X_values.shape[0] != y_array.shape[0]:
        raise ValueError("Features and target must contain the same number of rows.")

    index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X_values))
    model_names = list(base_model_factories.keys())
    oof = np.full((len(index), len(model_names)), np.nan, dtype=float)

    for train_idx, val_idx in splits:
        if len(val_idx) == 0:
            raise ValueError("Validation fold contains no rows.")

        X_train = _slice_rows(X, train_idx)
        y_train = y_array[train_idx]
        X_val = _slice_rows(X, val_idx)

        for model_position, (name, factory) in enumerate(base_model_factories.items()):
            model = factory()
            fitted = model.fit(X_train, y_train)
            if fitted is not model:
                model = fitted

            proba = model.predict_proba(X_val)
            oof[val_idx, model_position] = _positive_column(proba)

    return pd.DataFrame(oof, index=index, columns=model_names)


@dataclass
class StackingEnsemble:
    """Stack multiple level-0 models with a logistic meta-learner."""

    base_model_factories: Mapping[str, ModelFactory]
    meta_solver: str = "lbfgs"
    meta_max_iter: int = 500
    random_state: int | None = 42

    def __post_init__(self) -> None:
        if not self.base_model_factories:
            raise ValueError("StackingEnsemble requires at least one base model factory.")
        self._base_model_factories: "OrderedDict[str, ModelFactory]" = OrderedDict(
            self.base_model_factories
        )
        self._meta_model: LogisticRegression | None = None
        self._fitted_base_models: MutableMapping[str, SupportsPredictProba] | None = None
        self._oof_predictions: pd.DataFrame | None = None

    @property
    def base_model_names(self) -> list[str]:
        return list(self._base_model_factories.keys())

    @property
    def oof_predictions_(self) -> pd.DataFrame:
        if self._oof_predictions is None:
            raise RuntimeError("StackingEnsemble has not been fitted yet.")
        return self._oof_predictions

    @property
    def meta_model_(self) -> LogisticRegression:
        if self._meta_model is None:
            raise RuntimeError("StackingEnsemble has not been fitted yet.")
        return self._meta_model

    @property
    def base_models_(self) -> Mapping[str, SupportsPredictProba]:
        if self._fitted_base_models is None:
            raise RuntimeError("StackingEnsemble has not been fitted yet.")
        return self._fitted_base_models

    def fit(
        self,
        X: ArrayLike,
        y: Sequence[int | float],
        cv_splits: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> "StackingEnsemble":
        oof = generate_out_of_fold_predictions(self._base_model_factories, X, y, cv_splits)

        y_array = _ensure_1d(y)

        oof_matrix = oof.to_numpy()
        valid_mask = ~np.isnan(oof_matrix).any(axis=1)
        if not np.any(valid_mask):
            raise RuntimeError("No validation rows were scored for stacking.")

        meta_model = LogisticRegression(
            solver=self.meta_solver,
            max_iter=self.meta_max_iter,
            random_state=self.random_state,
        )
        meta_model.fit(oof_matrix[valid_mask], y_array[valid_mask])

        fitted_base_models: OrderedDict[str, SupportsPredictProba] = OrderedDict()
        for name, factory in self._base_model_factories.items():
            model = factory()
            fitted = model.fit(X, y)
            fitted_base_models[name] = fitted if fitted is not None else model

        self._meta_model = meta_model
        self._fitted_base_models = fitted_base_models
        self._oof_predictions = oof
        return self

    def predict_meta_features(self, X: ArrayLike) -> np.ndarray:
        if self._fitted_base_models is None:
            raise RuntimeError("StackingEnsemble must be fitted before prediction.")

        features = []
        for name in self.base_model_names:
            model = self._fitted_base_models[name]
            proba = model.predict_proba(X)
            features.append(_positive_column(proba))

        return np.column_stack(features)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        meta_features = self.predict_meta_features(X)
        meta_model = self.meta_model_
        return meta_model.predict_proba(meta_features)

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
