"""Feature engineering utilities for NFL prediction models."""

from .windows import RollingMetric, compute_group_rolling_windows

__all__ = ["RollingMetric", "compute_group_rolling_windows"]
