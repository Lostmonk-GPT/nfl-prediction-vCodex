"""Feature engineering utilities for NFL prediction models."""

from .injury_rollups import build_injury_rollups
from .stadium_join import join_stadium_metadata
from .travel import compute_travel_features, haversine_miles
from .weather import compute_weather_features
from .windows import RollingMetric, compute_group_rolling_windows

__all__ = [
    "RollingMetric",
    "build_injury_rollups",
    "compute_group_rolling_windows",
    "compute_travel_features",
    "compute_weather_features",
    "haversine_miles",
    "join_stadium_metadata",
]
