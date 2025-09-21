"""Travel-related schedule features.

This module computes travel context for each team/game pairing using the
league schedule. The MVP implementation focuses on the essentials needed for
feature engineering:

* A vectorised :func:`haversine_miles` helper for great-circle distance in
  statute miles. This allows downstream code to reuse the utility when other
  distance calculations are required.
* :func:`compute_travel_features` which expands the raw schedule into one row
  per team and derives the distance travelled since the previous game,
  calendar days since the previous kickoff, and whether the matchup is played
  at a neutral site.

The implementation intentionally keeps the coordinate sourcing flexible. If the
schedule already contains venue latitude/longitude columns they are used
directly. Otherwise, callers may supply a ``team_locations`` frame with best
available coordinates for each franchise which acts as a fallback for
non-neutral games.
"""

from __future__ import annotations

from typing import Final, Iterable, Tuple

import numpy as np
import pandas as pd

from nfl_pred.snapshot.visibility import VisibilityContext, filter_schedule

_EARTH_RADIUS_MILES: Final = 3958.7613
_REQUIRED_COLUMNS: Final[set[str]] = {
    "season",
    "week",
    "game_id",
    "start_time",
    "home_team",
    "away_team",
}
_COORDINATE_CANDIDATES: Final[Tuple[str, str]] = (
    ("venue_latitude", "venue_longitude"),
    ("site_latitude", "site_longitude"),
    ("stadium_latitude", "stadium_longitude"),
    ("latitude", "longitude"),
)


def haversine_miles(
    lat1: np.ndarray | float | int | Iterable[float],
    lon1: np.ndarray | float | int | Iterable[float],
    lat2: np.ndarray | float | int | Iterable[float],
    lon2: np.ndarray | float | int | Iterable[float],
) -> np.ndarray:
    """Return great-circle distance in miles between coordinate pairs.

    The function accepts scalars, ``numpy`` arrays, or any iterable convertible
    to an array. Broadcasting semantics follow ``numpy`` rules allowing vector
    operations across coordinate arrays.

    Args:
        lat1: Latitude(s) for the origin in decimal degrees.
        lon1: Longitude(s) for the origin in decimal degrees.
        lat2: Latitude(s) for the destination in decimal degrees.
        lon2: Longitude(s) for the destination in decimal degrees.

    Returns:
        ``numpy.ndarray`` containing the distance in statute miles for each
        coordinate pair. Elements are ``NaN`` when any of the paired
        coordinates are missing.
    """

    lat1_arr = np.asarray(lat1, dtype=float)
    lon1_arr = np.asarray(lon1, dtype=float)
    lat2_arr = np.asarray(lat2, dtype=float)
    lon2_arr = np.asarray(lon2, dtype=float)

    lat1_arr, lon1_arr, lat2_arr, lon2_arr = np.broadcast_arrays(
        lat1_arr, lon1_arr, lat2_arr, lon2_arr
    )

    mask = (
        np.isnan(lat1_arr)
        | np.isnan(lon1_arr)
        | np.isnan(lat2_arr)
        | np.isnan(lon2_arr)
    )

    lat1_rad = np.radians(lat1_arr)
    lon1_rad = np.radians(lon1_arr)
    lat2_rad = np.radians(lat2_arr)
    lon2_rad = np.radians(lon2_arr)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    distances = _EARTH_RADIUS_MILES * c
    distances = distances.astype(float, copy=False)
    distances = np.where(mask, np.nan, distances)
    return np.asarray(distances, dtype=float)


def compute_travel_features(
    schedule: pd.DataFrame,
    team_locations: pd.DataFrame | None = None,
    *,
    asof_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Derive per-team travel metrics from the NFL schedule.

    Args:
        schedule: Raw schedule frame produced by
            :mod:`nfl_pred.ingest.schedules` (or equivalent) containing at
            least the columns listed in :data:`_REQUIRED_COLUMNS`.
        team_locations: Optional frame with columns ``{"team", "latitude",
            "longitude"}`` providing best-available coordinates for each team.
            Used as a fallback when the schedule does not already contain
            explicit venue coordinates.
        asof_ts: Optional visibility cut-off. Games starting strictly after the
            timestamp are excluded before computing travel deltas.

    Returns:
        ``pandas.DataFrame`` with one row per ``(season, week, game_id, team)``
        tuple featuring ``travel_miles``, ``days_since_last``, and a
        ``neutral_site`` indicator suitable for feature joins.
    """

    _validate_schedule(schedule)
    context = VisibilityContext(asof_ts=asof_ts)
    working = filter_schedule(schedule, context=context, kickoff_column="start_time")
    if working.empty:
        return working.copy()

    working = working.copy()

    working["season"] = working["season"].astype(int)
    working["week"] = working["week"].astype(int)
    working["start_time"] = pd.to_datetime(
        working["start_time"], utc=True, errors="coerce"
    )

    neutral_flag = _extract_neutral_flag(working)

    venue_lat, venue_lon = _resolve_venue_coordinates(
        working, neutral_flag, team_locations
    )

    working["venue_latitude"] = venue_lat
    working["venue_longitude"] = venue_lon

    base_columns = [
        "season",
        "week",
        "game_id",
        "start_time",
        "venue_latitude",
        "venue_longitude",
    ]

    home = working[base_columns + ["home_team", "away_team"]].copy()
    home["team"] = home["home_team"].astype(str)
    home["opponent"] = home["away_team"].astype(str)
    home["home_away"] = np.where(neutral_flag, "neutral", "home")
    home["neutral_site"] = neutral_flag.astype(bool)

    away = working[base_columns + ["home_team", "away_team"]].copy()
    away["team"] = away["away_team"].astype(str)
    away["opponent"] = away["home_team"].astype(str)
    away["home_away"] = np.where(neutral_flag, "neutral", "away")
    away["neutral_site"] = neutral_flag.astype(bool)

    combined = pd.concat([home, away], ignore_index=True, sort=False)
    combined = combined.drop(columns=["home_team", "away_team"])

    combined = combined.sort_values(
        ["team", "season", "start_time", "game_id"]
    ).reset_index(drop=True)

    group_keys = ["team", "season"]
    prev_start = combined.groupby(group_keys, group_keys=False)["start_time"].shift(1)
    prev_lat = combined.groupby(group_keys, group_keys=False)["venue_latitude"].shift(1)
    prev_lon = combined.groupby(group_keys, group_keys=False)["venue_longitude"].shift(1)

    combined["days_since_last"] = (
        (combined["start_time"] - prev_start)
        .dt.total_seconds()
        .div(86400.0)
    )

    distances = haversine_miles(
        prev_lat.to_numpy(),
        prev_lon.to_numpy(),
        combined["venue_latitude"].to_numpy(),
        combined["venue_longitude"].to_numpy(),
    )
    combined["travel_miles"] = distances

    ordered_columns = [
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "home_away",
        "start_time",
        "neutral_site",
        "travel_miles",
        "days_since_last",
        "venue_latitude",
        "venue_longitude",
    ]

    combined = combined[ordered_columns].sort_values(
        ["season", "week", "game_id", "team"]
    ).reset_index(drop=True)

    return combined


def _validate_schedule(schedule: pd.DataFrame) -> None:
    missing = sorted(column for column in _REQUIRED_COLUMNS if column not in schedule.columns)
    if missing:
        raise KeyError(f"Schedule frame missing required columns: {missing}")


def _extract_neutral_flag(schedule: pd.DataFrame) -> pd.Series:
    """Return a boolean Series indicating neutral-site games."""

    if "neutral_site" in schedule.columns:
        neutral_raw = schedule["neutral_site"]
    elif "location" in schedule.columns:
        neutral_raw = schedule["location"].astype(str)
        return neutral_raw.str.lower().str.strip().eq("neutral")
    else:
        return pd.Series(False, index=schedule.index, dtype=bool)

    if pd.api.types.is_bool_dtype(neutral_raw):
        return neutral_raw.fillna(False)

    if pd.api.types.is_numeric_dtype(neutral_raw):
        return neutral_raw.fillna(0).astype(int).astype(bool)

    neutral_str = neutral_raw.astype(str).str.lower().str.strip()
    truthy = {"true", "t", "yes", "y", "1", "neutral"}
    return neutral_str.isin(truthy)


def _resolve_venue_coordinates(
    schedule: pd.DataFrame,
    neutral_flag: pd.Series,
    team_locations: pd.DataFrame | None,
) -> tuple[pd.Series, pd.Series]:
    """Determine venue latitude/longitude for each scheduled game."""

    lat_series, lon_series = _get_schedule_coordinates(schedule)
    if lat_series is not None and lon_series is not None:
        return lat_series, lon_series

    if team_locations is None:
        nan_series = pd.Series(np.nan, index=schedule.index, dtype="float64")
        return nan_series, nan_series.copy()

    required = {"team", "latitude", "longitude"}
    missing = sorted(required - set(team_locations.columns))
    if missing:
        raise KeyError(
            "team_locations frame missing required columns: {missing}".format(
                missing=missing
            )
        )

    location_lookup = team_locations.copy()
    location_lookup["latitude"] = pd.to_numeric(
        location_lookup["latitude"], errors="coerce"
    )
    location_lookup["longitude"] = pd.to_numeric(
        location_lookup["longitude"], errors="coerce"
    )

    location_lookup = location_lookup.set_index("team")[["latitude", "longitude"]]

    lat = schedule["home_team"].map(location_lookup["latitude"])
    lon = schedule["home_team"].map(location_lookup["longitude"])

    # Neutral-site games cannot be inferred from team home locations; mark them
    # as missing so downstream code does not assume zero travel.
    lat = lat.where(~neutral_flag, np.nan)
    lon = lon.where(~neutral_flag, np.nan)

    return lat, lon


def _get_schedule_coordinates(
    schedule: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None]:
    for lat_col, lon_col in _COORDINATE_CANDIDATES:
        if lat_col in schedule.columns and lon_col in schedule.columns:
            lat = pd.to_numeric(schedule[lat_col], errors="coerce")
            lon = pd.to_numeric(schedule[lon_col], errors="coerce")
            return lat, lon
    return None, None


__all__ = ["compute_travel_features", "haversine_miles"]
