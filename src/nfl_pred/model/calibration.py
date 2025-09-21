"""Platt scaling calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression


def _as_1d_array(values: Iterable[Any]) -> np.ndarray:
    """Convert ``values`` into a contiguous one-dimensional array."""

    arr = np.asarray(list(values))
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


@dataclass
class CalibrationParams:
    """Parameters learned by the Platt calibrator."""

    slope: float
    intercept: float


class PlattCalibrator:
    """Apply Platt scaling to a binary classifier."""

    def __init__(
        self,
        *,
        clip: float = 1e-6,
        max_iter: int = 200,
        solver: str = "lbfgs",
    ) -> None:
        if clip <= 0 or clip >= 0.5:
            raise ValueError("clip must be in the interval (0, 0.5).")

        self.clip = clip
        self.max_iter = max_iter
        self.solver = solver

        self._base_model: Any | None = None
        self._params: CalibrationParams | None = None
        self._classes: np.ndarray | None = None
        self._positive_index: int | None = None

    def fit(
        self,
        base_model: Any,
        X_valid: Any,
        y_valid: Iterable[Any],
    ) -> "PlattCalibrator":
        """Fit the calibrator using validation data."""

        if not hasattr(base_model, "predict_proba"):
            raise TypeError("base_model must implement predict_proba().")

        probs = np.asarray(base_model.predict_proba(X_valid))
        if probs.ndim != 2 or probs.shape[1] != 2:
            raise ValueError("PlattCalibrator requires binary probabilities with two columns.")

        if hasattr(base_model, "classes_"):
            classes = np.asarray(base_model.classes_)
            if classes.shape[0] != 2:
                raise ValueError("PlattCalibrator only supports binary classifiers.")
        else:
            classes = np.array([0, 1])

        if 1 in classes:
            positive_index = int(np.where(classes == 1)[0][0])
        else:
            positive_index = 1

        y_array = _as_1d_array(y_valid)
        if y_array.size != probs.shape[0]:
            raise ValueError("X_valid and y_valid must contain the same number of rows.")

        positive_probs = probs[:, positive_index]
        clipped = np.clip(positive_probs, self.clip, 1 - self.clip)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)

        model = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
        )
        model.fit(logits, y_array)

        self._base_model = base_model
        self._params = CalibrationParams(
            slope=float(model.coef_[0, 0]),
            intercept=float(model.intercept_[0]),
        )
        self._classes = classes
        self._positive_index = positive_index
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return calibrated probabilities for ``X``."""

        self._ensure_fitted()
        assert self._base_model is not None
        assert self._params is not None
        assert self._positive_index is not None

        base_probs = np.asarray(self._base_model.predict_proba(X))
        if base_probs.ndim != 2 or base_probs.shape[1] != 2:
            raise ValueError("PlattCalibrator requires binary probabilities with two columns.")

        positive_probs = base_probs[:, self._positive_index]
        clipped = np.clip(positive_probs, self.clip, 1 - self.clip)
        logits = np.log(clipped / (1 - clipped))

        calibrated_positive = 1 / (1 + np.exp(-(self._params.slope * logits + self._params.intercept)))
        calibrated = base_probs.copy()
        calibrated[:, self._positive_index] = calibrated_positive
        calibrated[:, 1 - self._positive_index] = 1 - calibrated_positive
        return calibrated

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels using calibrated probabilities."""

        probs = self.predict_proba(X)
        return (probs[:, self._positive_index] >= 0.5).astype(int)

    @property
    def calibration_params(self) -> CalibrationParams:
        """Return the learned calibration parameters."""

        self._ensure_fitted()
        assert self._params is not None
        return self._params

    @property
    def classes_(self) -> np.ndarray:
        """Return the class ordering used by the calibrator."""

        self._ensure_fitted()
        assert self._classes is not None
        return self._classes

    def _ensure_fitted(self) -> None:
        if self._params is None or self._base_model is None or self._positive_index is None:
            raise RuntimeError("PlattCalibrator must be fitted before use.")
