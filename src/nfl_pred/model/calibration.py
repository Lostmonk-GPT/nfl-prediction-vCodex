"""Probability calibration utilities for binary classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


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


class IsotonicCalibrator:
    """Calibrate probabilities using isotonic regression."""

    def __init__(
        self,
        *,
        y_min: float = 0.0,
        y_max: float = 1.0,
        out_of_bounds: str = "clip",
    ) -> None:
        self.y_min = y_min
        self.y_max = y_max
        self.out_of_bounds = out_of_bounds

        self._base_model: Any | None = None
        self._regressor: IsotonicRegression | None = None
        self._classes: np.ndarray | None = None
        self._positive_index: int | None = None

    def fit(
        self,
        base_model: Any,
        X_valid: Any,
        y_valid: Iterable[Any],
    ) -> "IsotonicCalibrator":
        """Fit the isotonic calibrator using validation data."""

        if not hasattr(base_model, "predict_proba"):
            raise TypeError("base_model must implement predict_proba().")

        probs = np.asarray(base_model.predict_proba(X_valid))
        if probs.ndim != 2 or probs.shape[1] != 2:
            raise ValueError("IsotonicCalibrator requires binary probabilities with two columns.")

        if hasattr(base_model, "classes_"):
            classes = np.asarray(base_model.classes_)
            if classes.shape[0] != 2:
                raise ValueError("IsotonicCalibrator only supports binary classifiers.")
        else:
            classes = np.array([0, 1])

        if 1 in classes:
            positive_index = int(np.where(classes == 1)[0][0])
        else:
            positive_index = 1

        y_array = _as_1d_array(y_valid)
        if y_array.size != probs.shape[0]:
            raise ValueError("X_valid and y_valid must contain the same number of rows.")

        regressor = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            out_of_bounds=self.out_of_bounds,
        )
        regressor.fit(probs[:, positive_index], y_array)

        self._base_model = base_model
        self._regressor = regressor
        self._classes = classes
        self._positive_index = positive_index
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return calibrated probabilities for ``X``."""

        self._ensure_fitted()
        assert self._base_model is not None
        assert self._regressor is not None
        assert self._positive_index is not None

        base_probs = np.asarray(self._base_model.predict_proba(X))
        if base_probs.ndim != 2 or base_probs.shape[1] != 2:
            raise ValueError("IsotonicCalibrator requires binary probabilities with two columns.")

        positive_probs = base_probs[:, self._positive_index]
        calibrated_positive = self._regressor.transform(positive_probs)
        calibrated = base_probs.copy()
        calibrated[:, self._positive_index] = calibrated_positive
        calibrated[:, 1 - self._positive_index] = 1 - calibrated_positive
        return calibrated

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels using calibrated probabilities."""

        probs = self.predict_proba(X)
        return (probs[:, self._positive_index] >= 0.5).astype(int)

    @property
    def classes_(self) -> np.ndarray:
        """Return the class ordering used by the calibrator."""

        self._ensure_fitted()
        assert self._classes is not None
        return self._classes

    def _ensure_fitted(self) -> None:
        if self._regressor is None or self._base_model is None or self._positive_index is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before use.")


class _CalibratorProtocol(Protocol):
    """Protocol describing the calibrator interface used for selection."""

    def fit(self, base_model: Any, X_valid: Any, y_valid: Iterable[Any]) -> "_CalibratorProtocol":
        ...

    def predict_proba(self, X: Any) -> np.ndarray:
        ...

    @property
    def classes_(self) -> np.ndarray:
        ...


def _default_calibrator_factories() -> Sequence[tuple[str, Callable[[], _CalibratorProtocol]]]:
    return (
        ("platt", PlattCalibrator),
        ("isotonic", IsotonicCalibrator),
    )


def _compute_metric(
    *,
    metric: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
) -> float:
    if metric == "log_loss":
        return float(log_loss(y_true, probs, labels=classes))
    if metric == "brier":
        if 1 in classes:
            positive_index = int(np.where(classes == 1)[0][0])
        else:
            positive_index = 1
        return float(brier_score_loss(y_true, probs[:, positive_index]))
    raise ValueError("metric must be 'log_loss' or 'brier'.")


def compare_calibrators(
    base_model: Any,
    X_valid: Any,
    y_valid: Iterable[Any],
    *,
    metric: str = "log_loss",
    minimum_isotonic_samples: int = 75,
    calibrator_factories: Sequence[tuple[str, Callable[[], _CalibratorProtocol]]] | None = None,
) -> tuple[str, Dict[str, float], Dict[str, _CalibratorProtocol]]:
    """Fit and evaluate multiple calibrators, returning the best performer."""

    y_array = _as_1d_array(y_valid)
    if y_array.size == 0:
        raise ValueError("y_valid must contain at least one sample.")
    unique = np.unique(y_array)
    if unique.shape[0] != 2:
        raise ValueError("Calibration comparison requires both classes in y_valid.")

    factories = calibrator_factories or _default_calibrator_factories()

    metrics: Dict[str, float] = {}
    fitted: Dict[str, _CalibratorProtocol] = {}
    best_name: str | None = None
    best_metric = float("inf")

    for name, factory in factories:
        if name == "isotonic" and y_array.size < minimum_isotonic_samples:
            continue

        calibrator = factory()
        fitted_calibrator = calibrator.fit(base_model, X_valid, y_array)
        probs = np.asarray(fitted_calibrator.predict_proba(X_valid))
        metric_value = _compute_metric(
            metric=metric,
            y_true=y_array,
            probs=probs,
            classes=fitted_calibrator.classes_,
        )
        metrics[name] = metric_value
        fitted[name] = fitted_calibrator
        if metric_value < best_metric:
            best_metric = metric_value
            best_name = name

    if not metrics:
        raise ValueError("No calibrators were evaluated; check sample sizes and configuration.")

    assert best_name is not None
    return best_name, metrics, fitted


class CalibrationSelector:
    """Select the best calibrator based on validation performance."""

    def __init__(
        self,
        *,
        metric: str = "log_loss",
        minimum_isotonic_samples: int = 75,
        calibrator_factories: Sequence[tuple[str, Callable[[], _CalibratorProtocol]]] | None = None,
    ) -> None:
        self.metric = metric
        self.minimum_isotonic_samples = minimum_isotonic_samples
        self.calibrator_factories = calibrator_factories

        self._selected_name: str | None = None
        self._selected_calibrator: _CalibratorProtocol | None = None
        self._metrics: Dict[str, float] | None = None

    def fit(self, base_model: Any, X_valid: Any, y_valid: Iterable[Any]) -> "CalibrationSelector":
        """Fit available calibrators and retain the best performer."""

        best_name, metrics, fitted = compare_calibrators(
            base_model,
            X_valid,
            y_valid,
            metric=self.metric,
            minimum_isotonic_samples=self.minimum_isotonic_samples,
            calibrator_factories=self.calibrator_factories,
        )

        self._selected_name = best_name
        self._selected_calibrator = fitted[best_name]
        self._metrics = metrics
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict calibrated probabilities using the selected calibrator."""

        calibrator = self._ensure_selected()
        return calibrator.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels using the selected calibrator."""

        calibrator = self._ensure_selected()
        return calibrator.predict(X)

    @property
    def selected_calibrator_name(self) -> str:
        """Name of the chosen calibrator."""

        self._ensure_selected()
        assert self._selected_name is not None
        return self._selected_name

    @property
    def metrics_(self) -> Mapping[str, float]:
        """Validation metric values for each evaluated calibrator."""

        self._ensure_selected()
        assert self._metrics is not None
        return self._metrics

    @property
    def calibrator_(self) -> _CalibratorProtocol:
        """Return the fitted calibrator selected during ``fit``."""

        return self._ensure_selected()

    def _ensure_selected(self) -> _CalibratorProtocol:
        if self._selected_calibrator is None:
            raise RuntimeError("CalibrationSelector must be fitted before use.")
        return self._selected_calibrator
