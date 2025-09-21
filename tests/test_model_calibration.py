"""Tests for Platt calibration utilities."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from nfl_pred.model.calibration import PlattCalibrator


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class MisCalibratedModel:
    """Deterministic classifier with intentionally skewed probabilities."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights
        base = _sigmoid(logits)
        distorted = np.clip(base**2, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - distorted, distorted])


def test_platt_calibrator_reduces_brier_score() -> None:
    rng = np.random.default_rng(123)
    n_samples = 600
    X = rng.normal(size=(n_samples, 3))
    true_weights = np.array([0.5, -1.25, 0.75])
    probabilities = _sigmoid(X @ true_weights)
    y = rng.binomial(1, probabilities)

    base_model = MisCalibratedModel(true_weights)

    X_valid, y_valid = X[:200], y[:200]
    X_test, y_test = X[200:], y[200:]

    calibrator = PlattCalibrator()
    calibrator.fit(base_model, X_valid, y_valid)

    base_probs = base_model.predict_proba(X_test)[:, 1]
    calibrated_probs = calibrator.predict_proba(X_test)[:, 1]

    base_brier = brier_score_loss(y_test, base_probs)
    calibrated_brier = brier_score_loss(y_test, calibrated_probs)

    assert calibrated_brier < base_brier


def test_platt_calibrator_requires_fit() -> None:
    base_model = MisCalibratedModel(np.array([1.0, -0.5]))
    calibrator = PlattCalibrator()

    with pytest.raises(RuntimeError):
        _ = calibrator.predict_proba(np.zeros((2, 2)))

    X_valid = np.zeros((4, 2))
    y_valid = np.array([0, 1, 0, 1])

    calibrator.fit(base_model, X_valid, y_valid)
    probs = calibrator.predict_proba(np.zeros((3, 2)))
    assert probs.shape == (3, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
