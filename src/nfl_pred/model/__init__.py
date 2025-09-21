"""Modeling utilities for NFL prediction workflows."""

from .baseline import BaselineClassifier  # noqa: F401
from .calibration import PlattCalibrator  # noqa: F401
from .splits import time_series_splits  # noqa: F401

__all__ = ["BaselineClassifier", "PlattCalibrator", "time_series_splits"]
