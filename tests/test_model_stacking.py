"""Tests for the stacking ensemble utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from nfl_pred.model.splits import time_series_splits
from nfl_pred.model.stacking import (
    StackingEnsemble,
    generate_out_of_fold_predictions,
)


class _BiasedSigmoidModel:
    def __init__(self, column: str, bias: float) -> None:
        self.column = column
        self.bias = bias
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_BiasedSigmoidModel":
        series = X[self.column]
        self._mean = float(series.mean())
        self._std = float(series.std() or 1.0)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("Model must be fitted before prediction.")
        normalized = (X[self.column] - self._mean) / self._std
        positive = 1.0 / (1.0 + np.exp(-normalized)) + self.bias
        clipped = np.clip(positive, 1e-4, 1 - 1e-4)
        return np.column_stack([1.0 - clipped, clipped])


@pytest.fixture()
def toy_stack_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    weeks = np.repeat(np.arange(1, 7), 30)
    feature_1 = rng.normal(size=weeks.size)
    feature_2 = rng.normal(size=weeks.size)
    logits = 1.1 * feature_1 + 1.1 * feature_2
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    outcomes = rng.binomial(1, probabilities)

    df = pd.DataFrame(
        {
            "week": weeks,
            "feature_1": feature_1,
            "feature_2": feature_2,
        }
    )
    return df, outcomes


def test_generate_out_of_fold_predictions_shape(toy_stack_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
    df, y = toy_stack_dataset
    features = df[["feature_1", "feature_2"]]
    splits = list(time_series_splits(df, group_col="week", min_train_weeks=2))

    oof = generate_out_of_fold_predictions(
        {
            "biased_1": lambda: _BiasedSigmoidModel("feature_1", 0.15),
            "biased_2": lambda: _BiasedSigmoidModel("feature_2", -0.08),
        },
        features,
        y,
        splits,
    )

    assert oof.shape == (features.shape[0], 2)
    assert list(oof.columns) == ["biased_1", "biased_2"]

    late_weeks = df["week"] >= 3
    late_predictions = oof.loc[late_weeks]
    assert np.all((0.0 <= late_predictions.to_numpy()) & (late_predictions.to_numpy() <= 1.0))
    assert late_predictions.notna().all().all()
    assert oof.loc[~late_weeks].isna().any(axis=1).all()


def test_stacking_meta_outperforms_best_base_on_holdout(
    toy_stack_dataset: tuple[pd.DataFrame, np.ndarray]
) -> None:
    df, y = toy_stack_dataset
    features = df[["feature_1", "feature_2"]]
    train_mask = df["week"] <= 5
    test_mask = ~train_mask

    X_train = features.loc[train_mask]
    y_train = y[train_mask.to_numpy()]
    X_test = features.loc[test_mask]
    y_test = y[test_mask.to_numpy()]

    factories = {
        "biased_1": lambda: _BiasedSigmoidModel("feature_1", 0.15),
        "biased_2": lambda: _BiasedSigmoidModel("feature_2", -0.08),
    }

    splits = list(time_series_splits(df.loc[train_mask], group_col="week", min_train_weeks=2))

    ensemble = StackingEnsemble(factories, meta_max_iter=600)
    ensemble.fit(X_train, y_train, splits)

    meta_proba = ensemble.predict_proba(X_test)[:, 1]
    base_losses = []
    for model in ensemble.base_models_.values():
        base_prob = model.predict_proba(X_test)[:, 1]
        base_losses.append(log_loss(y_test, base_prob, labels=[0, 1]))

    meta_loss = log_loss(y_test, meta_proba, labels=[0, 1])

    assert meta_loss < min(base_losses)
