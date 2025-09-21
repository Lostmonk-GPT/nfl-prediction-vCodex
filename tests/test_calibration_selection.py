"""Tests for calibration selection utilities (AI-303)."""

from __future__ import annotations

import numpy as np

from nfl_pred.model.calibration import CalibrationSelector, compare_calibrators


class DummyModel:
    """Simple model returning predetermined probabilities based on ``X``."""

    def __init__(self, transform) -> None:
        self._transform = transform
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        x = np.asarray(X).reshape(-1)
        positive = np.clip(self._transform(x), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - positive, positive])


def _generate_nonlinear_dataset(num_points: int = 30, samples_per_point: int = 40):
    x_grid = np.linspace(0.05, 0.95, num_points)
    true_probs = x_grid**2

    X_valid = np.repeat(x_grid, samples_per_point).reshape(-1, 1)
    y_valid = []
    for prob in true_probs:
        positives = int(round(prob * samples_per_point))
        positives = min(max(positives, 0), samples_per_point)
        y_valid.extend([1] * positives)
        y_valid.extend([0] * (samples_per_point - positives))

    return X_valid, np.asarray(y_valid)


def test_compare_calibrators_prefers_isotonic_for_nonlinear_relationship():
    base_model = DummyModel(lambda x: x)
    X_valid, y_valid = _generate_nonlinear_dataset()

    best_name, metrics, _ = compare_calibrators(
        base_model,
        X_valid,
        y_valid,
        metric="brier",
        minimum_isotonic_samples=50,
    )

    assert best_name == "isotonic"
    assert "isotonic" in metrics
    assert "platt" in metrics
    assert metrics["isotonic"] < metrics["platt"]


def test_compare_calibrators_skips_isotonic_when_sample_small():
    base_model = DummyModel(lambda x: 0.25 + 0.5 * x)
    X_valid = np.linspace(0.1, 0.9, 20).reshape(-1, 1)
    y_valid = (X_valid.ravel() > 0.5).astype(int)

    best_name, metrics, _ = compare_calibrators(
        base_model,
        X_valid,
        y_valid,
        metric="log_loss",
        minimum_isotonic_samples=50,
    )

    assert best_name == "platt"
    assert "platt" in metrics
    assert "isotonic" not in metrics


def test_calibration_selector_uses_best_calibrator_predictions():
    base_model = DummyModel(lambda x: x)
    X_valid, y_valid = _generate_nonlinear_dataset()

    selector = CalibrationSelector(metric="brier", minimum_isotonic_samples=50)
    selector.fit(base_model, X_valid, y_valid)

    assert selector.selected_calibrator_name == "isotonic"
    assert "platt" in selector.metrics_
    assert "isotonic" in selector.metrics_

    calibrated_probs = selector.predict_proba(X_valid)
    assert np.allclose(calibrated_probs, selector.calibrator_.predict_proba(X_valid))
