"""Client for interacting with Meteostat for historical weather data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Protocol

import pandas as pd
from meteostat import Daily, Hourly, Stations

from .storage import WeatherArtifactStore


class MeteostatClientError(RuntimeError):
    """Raised when Meteostat does not return the expected data."""


class _FetcherProtocol(Protocol):
    """Protocol describing objects which can fetch data frames."""

    def fetch(self, limit: int | None = None) -> pd.DataFrame:  # pragma: no cover - interface
        ...


@dataclass
class StationRecord:
    """Normalized representation of a Meteostat station."""

    station_id: str
    name: str | None
    latitude: float
    longitude: float
    distance_km: float
    distance_miles: float
    elevation_m: float | None
    timezone: str | None


class MeteostatClient:
    """Wrapper around the Meteostat Python library."""

    def __init__(
        self,
        *,
        stations_factory: Callable[[], Stations] | None = None,
        hourly_cls: type[Hourly] = Hourly,
        daily_cls: type[Daily] = Daily,
        max_station_distance_miles: float = 10.0,
        artifact_store: WeatherArtifactStore | None = None,
        artifact_version: str = "unknown",
        artifact_ttl_seconds: float | None = None,
    ) -> None:
        self._stations_factory = stations_factory or Stations
        self._hourly_cls = hourly_cls
        self._daily_cls = daily_cls
        self._max_distance_miles = max_station_distance_miles
        self._artifact_store = artifact_store
        self._artifact_version = artifact_version
        self._artifact_ttl_seconds = artifact_ttl_seconds

    # ------------------------------------------------------------------
    # Station utilities
    # ------------------------------------------------------------------
    def nearest_station(self, latitude: float, longitude: float) -> StationRecord:
        """Return the nearest station within the configured distance."""

        stations = self._stations_factory()
        fetcher: _FetcherProtocol = stations.nearby(latitude, longitude)
        df = fetcher.fetch(10)
        if df.empty:
            raise MeteostatClientError("No Meteostat stations found for coordinates")

        station_rows = _normalize_stations(df, latitude, longitude)
        if not station_rows:
            raise MeteostatClientError("Unable to normalize Meteostat station payload")

        station = station_rows[0]
        if station.distance_miles > self._max_distance_miles:
            raise MeteostatClientError(
                "Nearest Meteostat station is beyond the allowed distance"
            )

        return station

    # ------------------------------------------------------------------
    # Weather data utilities
    # ------------------------------------------------------------------
    def hourly(
        self,
        station: str | StationRecord,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch hourly data for the provided window and station."""

        station_id = station.station_id if isinstance(station, StationRecord) else station
        fetcher = self._hourly_cls(station_id, start, end)
        df = fetcher.fetch()
        payload = _payload_from_frame(df)
        self._persist_raw(
            endpoint="hourly",
            params={
                "station_id": station_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            payload=payload,
        )
        return _normalize_hourly(df, station_id)

    def daily(
        self,
        station: str | StationRecord,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch daily data for the provided window and station."""

        station_id = station.station_id if isinstance(station, StationRecord) else station
        fetcher = self._daily_cls(station_id, start, end)
        df = fetcher.fetch()
        payload = _payload_from_frame(df)
        self._persist_raw(
            endpoint="daily",
            params={
                "station_id": station_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            payload=payload,
        )
        return _normalize_daily(df, station_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _persist_raw(
        self,
        *,
        endpoint: str,
        params: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        if self._artifact_store is None:
            return
        try:
            self._artifact_store.save(
                source="meteostat",
                endpoint=endpoint,
                params=params,
                payload=payload,
                version=self._artifact_version,
                ttl_seconds=self._artifact_ttl_seconds,
            )
        except Exception:  # pragma: no cover - defensive
            pass


# ----------------------------------------------------------------------
# Normalization helpers
# ----------------------------------------------------------------------


def _normalize_stations(
    df: pd.DataFrame, latitude: float, longitude: float
) -> list[StationRecord]:
    records: list[StationRecord] = []
    for row in df.itertuples(index=False):
        station_id = getattr(row, "id", None)
        station_lat = getattr(row, "latitude", None)
        station_lon = getattr(row, "longitude", None)
        if not station_id or station_lat is None or station_lon is None:
            continue
        distance_km = _haversine_km(latitude, longitude, float(station_lat), float(station_lon))
        distance_miles = _km_to_miles(distance_km)
        records.append(
            StationRecord(
                station_id=str(station_id),
                name=_maybe_str(getattr(row, "name", None)),
                latitude=float(station_lat),
                longitude=float(station_lon),
                distance_km=distance_km,
                distance_miles=distance_miles,
                elevation_m=_maybe_float(getattr(row, "elevation", None)),
                timezone=_maybe_str(getattr(row, "timezone", None)),
            )
        )
    records.sort(key=lambda rec: rec.distance_km)
    return records


def _normalize_hourly(df: pd.DataFrame, station_id: str) -> list[dict[str, Any]]:
    if df.empty:
        return []

    records: list[dict[str, Any]] = []
    frame = df.reset_index()
    for row in frame.itertuples(index=False):
        timestamp = getattr(row, frame.columns[0])
        records.append(
            {
                "station_id": station_id,
                "time": _timestamp_to_iso(timestamp),
                "temperature_c": _maybe_float(getattr(row, "temp", None)),
                "dewpoint_c": _maybe_float(getattr(row, "dwpt", None)),
                "relative_humidity_pct": _maybe_float(getattr(row, "rhum", None)),
                "precipitation_mm": _maybe_float(getattr(row, "prcp", None)),
                "snowfall_mm": _maybe_float(getattr(row, "snow", None)),
                "wind_direction_deg": _maybe_float(getattr(row, "wdir", None)),
                "wind_speed_mps": _kmh_to_mps(getattr(row, "wspd", None)),
                "wind_gust_mps": _kmh_to_mps(getattr(row, "wpgt", None)),
                "pressure_hpa": _maybe_float(getattr(row, "pres", None)),
                "sunshine_minutes": _maybe_float(getattr(row, "tsun", None)),
            }
        )
    return records


def _normalize_daily(df: pd.DataFrame, station_id: str) -> list[dict[str, Any]]:
    if df.empty:
        return []

    frame = df.reset_index()
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        date_value = getattr(row, frame.columns[0])
        records.append(
            {
                "station_id": station_id,
                "date": _timestamp_to_iso(date_value),
                "temperature_avg_c": _maybe_float(getattr(row, "tavg", None)),
                "temperature_min_c": _maybe_float(getattr(row, "tmin", None)),
                "temperature_max_c": _maybe_float(getattr(row, "tmax", None)),
                "precipitation_mm": _maybe_float(getattr(row, "prcp", None)),
                "snowfall_mm": _maybe_float(getattr(row, "snow", None)),
                "wind_direction_deg": _maybe_float(getattr(row, "wdir", None)),
                "wind_speed_mps": _kmh_to_mps(getattr(row, "wspd", None)),
                "wind_gust_mps": _kmh_to_mps(getattr(row, "wpgt", None)),
                "pressure_hpa": _maybe_float(getattr(row, "pres", None)),
                "sunshine_minutes": _maybe_float(getattr(row, "tsun", None)),
            }
        )
    return records


def _payload_from_frame(df: pd.DataFrame) -> Mapping[str, Any]:
    frame = df.reset_index()
    payload_records = frame.to_dict(orient="records")
    return {"records": payload_records}


def _timestamp_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        ts = value
    else:
        ts = pd.to_datetime(value, utc=False, errors="coerce")
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.isoformat()
    return str(ts)


def _maybe_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        if isinstance(value, pd.Timestamp):
            return None
        return None
    if math.isnan(float_value):
        return None
    return float_value


def _kmh_to_mps(value: Any) -> float | None:
    numeric = _maybe_float(value)
    if numeric is None:
        return None
    return numeric / 3.6


def _km_to_miles(km: float) -> float:
    return km * 0.621371


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


__all__ = ["MeteostatClient", "MeteostatClientError", "StationRecord"]
