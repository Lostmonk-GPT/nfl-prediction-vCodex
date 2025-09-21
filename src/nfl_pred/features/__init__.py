"""Feature engineering utilities for NFL prediction models."""

from .stadium_join import join_stadium_metadata
from .travel import compute_travel_features, haversine_miles
from .weather import compute_weather_features
from .windows import RollingMetric, compute_group_rolling_windows

__all__ = [
    "RollingMetric",
    "compute_group_rolling_windows",
    "compute_travel_features",
    "compute_weather_features",
    "haversine_miles",
    "join_stadium_metadata",
]
