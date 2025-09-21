"""Unit tests for additional level-0 model wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_pred.model.models import (
    GradientBoostingModel,
    LogisticModel,
    MODEL_PARAM_GRIDS,
    RidgeModel,
)


@pytest.fixture()
def toy_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "feature_1": rng.normal(size=60),
            "feature_2": rng.normal(size=60),
            "feature_3": rng.integers(0, 2, size=60),
        }
    )
    logits = X["feature_1"] * 0.7 + X["feature_2"] * 0.3 + X["feature_3"] * 0.5
    y = (logits > np.median(logits)).astype(int).to_numpy()
    return X, y


def _assert_probabilities_valid(proba: np.ndarray, n_rows: int) -> None:
    assert proba.shape == (n_rows, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((0.0 <= proba) & (proba <= 1.0))


class TestLogisticModel:
    def test_logistic_model_returns_well_formed_probabilities(self, toy_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = toy_dataset
        model = LogisticModel(max_iter=200)
        fitted = model.fit(X, y)
        assert fitted is model

        proba = model.predict_proba(X.head(5))
        _assert_probabilities_valid(proba, 5)


class TestRidgeModel:
    def test_ridge_model_probability_link(self, toy_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = toy_dataset
        model = RidgeModel(alpha=0.5)
        model.fit(X, y)

        proba = model.predict_proba(X.tail(4))
        _assert_probabilities_valid(proba, 4)

        # Ensure the model is not degenerate (probabilities are not all identical)
        assert np.unique(proba[:, 1]).size > 1


class TestGradientBoostingModel:
    def test_gradient_boosting_model_uses_tree_backend(self, toy_dataset: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = toy_dataset
        model = GradientBoostingModel(n_estimators=60, max_depth=3, learning_rate=0.1)
        model.fit(X, y)

        proba = model.predict_proba(X.iloc[:3])
        _assert_probabilities_valid(proba, 3)


class TestModelParamGrids:
    def test_param_grids_are_defined_for_each_model(self) -> None:
        assert set(MODEL_PARAM_GRIDS) == {"logistic", "ridge", "gbdt"}

        assert MODEL_PARAM_GRIDS["logistic"]["C"] == [0.1, 1.0, 5.0]
        assert MODEL_PARAM_GRIDS["ridge"]["alpha"] == [0.1, 1.0, 5.0]
        assert set(MODEL_PARAM_GRIDS["gbdt"]) == {"learning_rate", "max_depth", "n_estimators", "subsample"}
