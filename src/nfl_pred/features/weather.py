"""Weather feature builder integrating forecast and historical sources.

The MVP implementation focuses on deriving a compact set of weather-oriented
features that can be joined onto the per-team feature matrix. The
``compute_weather_features`` entry point expands the raw schedule into one row
per team, resolves the appropriate stadium metadata, and—when possible—queries
the National Weather Service (NWS) gridpoint forecast. If a forecast is not
available, callers may optionally provide a Meteostat client for historical
hourly backfill.

Feature outputs
---------------
``wx_temp``
    Forecast (or historical) temperature in degrees Fahrenheit. ``NaN`` when
    unavailable or when the venue is closed/indoor.
``wx_wind``
    Sustained wind speed in miles per hour. Indoor venues default to ``0`` to
    reflect the absence of wind impact.
``precip``
    Probability of precipitation represented as a fraction in the ``[0, 1]``
    range. When historical Meteostat data are used as a fallback the value is
    approximated as ``1.0`` when measurable precipitation occurred and ``0.0``
    otherwise. Closed venues default to ``0``.

Visibility policy
-----------------
Only games played in venues with ``roof`` values mapping to "outdoor",
"open", or "retractable" receive weather features. All other roof types are
treated as closed/indoor and therefore produce ``NaN``/zero defaults as per the
project requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from nfl_pred.features.stadium_join import join_stadium_metadata
from nfl_pred.weather.meteostat_client import MeteostatClient, MeteostatClientError
from nfl_pred.weather.nws_client import NWSClient, NWSClientError

LOGGER = logging.getLogger(__name__)

_OUTDOOR_ROOF_TYPES = {"outdoor", "open", "retractable"}
_INDOOR_DEFAULTS = {
    "wx_temp": np.nan,
    "wx_wind": 0.0,
    "precip": 0.0,
}


@dataclass(slots=True)
class _WeatherReading:
    """Container for derived weather attributes."""

    temperature_f: float | None
    wind_mph: float | None
    precip_probability: float | None


def compute_weather_features(
    schedule: pd.DataFrame,
    stadiums: pd.DataFrame,
    *,
    nws_client: NWSClient | None = None,
    meteostat_client: MeteostatClient | None = None,
    asof_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Derive weather features for the provided schedule.

    Args:
        schedule: Schedule frame with at least ``season``, ``week``,
            ``game_id``, ``start_time``, ``home_team``, ``away_team``, and a
            ``stadium`` or ``venue`` column for metadata enrichment.
        stadiums: Authoritative stadium reference table.
        nws_client: Optional National Weather Service client used to retrieve
            pre-game forecasts. When omitted, forecast values remain ``NaN``.
        meteostat_client: Optional Meteostat client used as a historical
            fallback when the forecast is unavailable.
        asof_ts: Optional visibility cutoff timestamp. Forecast payloads with
            ``updated_at`` values after this timestamp are ignored.

    Returns:
        ``pandas.DataFrame`` containing ``wx_temp``, ``wx_wind``, and
        ``precip`` columns keyed by ``season``, ``week``, ``game_id``, and
        ``team``.
    """

    if schedule.empty:
        return _empty_weather_frame()

    working = schedule.copy()
    working["start_time"] = pd.to_datetime(working["start_time"], utc=True, errors="coerce")

    enriched = join_stadium_metadata(working, stadiums)
    enriched = enriched.rename(
        columns={
            "lat": "venue_latitude",
            "lon": "venue_longitude",
        }
    )

    merged = working.merge(
        enriched[["season", "week", "game_id", "roof", "venue_latitude", "venue_longitude"]],
        on=["season", "week", "game_id"],
        how="left",
    )

    if asof_ts is not None:
        asof_ts = _ensure_utc(asof_ts)

    point_cache: dict[tuple[float, float], Mapping[str, Any]] = {}
    forecast_cache: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    station_cache: dict[tuple[float, float], Any] = {}

    records: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        roof = _normalize_roof(getattr(row, "roof", None))
        kickoff: pd.Timestamp | None = getattr(row, "start_time", None)
        venue_lat = _maybe_float(getattr(row, "venue_latitude", None))
        venue_lon = _maybe_float(getattr(row, "venue_longitude", None))

        if roof not in _OUTDOOR_ROOF_TYPES:
            reading = _INDOOR_DEFAULTS.copy()
        else:
            reading = _collect_weather_reading(
                kickoff,
                venue_lat,
                venue_lon,
                nws_client=nws_client,
                meteostat_client=meteostat_client,
                asof_ts=asof_ts,
                point_cache=point_cache,
                forecast_cache=forecast_cache,
                station_cache=station_cache,
            )

        base_payload = {
            "season": getattr(row, "season"),
            "week": getattr(row, "week"),
            "game_id": getattr(row, "game_id"),
        }

        home_team = getattr(row, "home_team")
        away_team = getattr(row, "away_team")

        records.append({"team": home_team, **base_payload, **reading})
        records.append({"team": away_team, **base_payload, **reading})

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return _empty_weather_frame()

    frame = frame.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    return frame


def _collect_weather_reading(
    kickoff: pd.Timestamp | None,
    lat: float | None,
    lon: float | None,
    *,
    nws_client: NWSClient | None,
    meteostat_client: MeteostatClient | None,
    asof_ts: pd.Timestamp | None,
    point_cache: dict[tuple[float, float], Mapping[str, Any]],
    forecast_cache: dict[tuple[str, int, int], Mapping[str, Any]],
    station_cache: dict[tuple[float, float], Any],
) -> Mapping[str, Any]:
    if pd.isna(kickoff) or lat is None or lon is None:
        return {"wx_temp": np.nan, "wx_wind": np.nan, "precip": np.nan}

    kickoff = _ensure_utc(kickoff)

    forecast_reading: _WeatherReading | None = None
    if nws_client is not None:
        forecast_reading = _try_forecast(
            kickoff,
            lat,
            lon,
            nws_client=nws_client,
            asof_ts=asof_ts,
            point_cache=point_cache,
            forecast_cache=forecast_cache,
        )

    if forecast_reading is not None:
        return _reading_to_payload(forecast_reading)

    if meteostat_client is not None:
        historical_reading = _try_meteostat(
            kickoff,
            lat,
            lon,
            meteostat_client=meteostat_client,
            station_cache=station_cache,
        )
        if historical_reading is not None:
            return _reading_to_payload(historical_reading)

    return {"wx_temp": np.nan, "wx_wind": np.nan, "precip": np.nan}


def _try_forecast(
    kickoff: pd.Timestamp,
    lat: float,
    lon: float,
    *,
    nws_client: NWSClient,
    asof_ts: pd.Timestamp | None,
    point_cache: dict[tuple[float, float], Mapping[str, Any]],
    forecast_cache: dict[tuple[str, int, int], Mapping[str, Any]],
) -> _WeatherReading | None:
    point_key = (round(lat, 4), round(lon, 4))
    metadata = point_cache.get(point_key)
    if metadata is None:
        try:
            metadata = nws_client.point_metadata(lat, lon)
            point_cache[point_key] = metadata
        except NWSClientError as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("NWS point metadata lookup failed: %s", exc)
            return None

    grid_id = metadata.get("grid_id")
    grid_x = metadata.get("grid_x")
    grid_y = metadata.get("grid_y")
    if grid_id is None or grid_x is None or grid_y is None:
        return None

    try:
        grid_x_int = int(grid_x)
        grid_y_int = int(grid_y)
    except (TypeError, ValueError):
        return None

    forecast_key = (str(grid_id), grid_x_int, grid_y_int)
    forecast = forecast_cache.get(forecast_key)
    if forecast is None:
        try:
            forecast = nws_client.gridpoint_forecast(str(grid_id), grid_x_int, grid_y_int)
            forecast_cache[forecast_key] = forecast
        except NWSClientError as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("NWS forecast lookup failed: %s", exc)
            return None

    if asof_ts is not None:
        updated_raw = forecast.get("updated_at")
        updated_at = _parse_timestamp(updated_raw)
        if updated_at is not None and updated_at > asof_ts:
            return None

    periods: Iterable[Mapping[str, Any]] = forecast.get("periods", []) or []
    period = _select_period(periods, kickoff)
    if period is None:
        return None

    temp_f = _c_to_f(period.get("temperature_c"))
    wind_mph = _mps_to_mph(period.get("wind_speed_mps"))
    precip_pct = _maybe_float(period.get("probability_of_precipitation_pct"))
    precip_prob = None if precip_pct is None else np.clip(precip_pct / 100.0, 0.0, 1.0)

    return _WeatherReading(temperature_f=temp_f, wind_mph=wind_mph, precip_probability=precip_prob)


def _try_meteostat(
    kickoff: pd.Timestamp,
    lat: float,
    lon: float,
    *,
    meteostat_client: MeteostatClient,
    station_cache: dict[tuple[float, float], Any],
) -> _WeatherReading | None:
    point_key = (round(lat, 4), round(lon, 4))
    station = station_cache.get(point_key)
    if station is None:
        try:
            station = meteostat_client.nearest_station(lat, lon)
            station_cache[point_key] = station
        except MeteostatClientError as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Meteostat station lookup failed: %s", exc)
            return None

    start = (kickoff - pd.Timedelta(hours=1)).to_pydatetime()
    end = (kickoff + pd.Timedelta(hours=1)).to_pydatetime()

    try:
        records = meteostat_client.hourly(station, start, end)
    except MeteostatClientError as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Meteostat hourly lookup failed: %s", exc)
        return None

    if not records:
        return None

    selected = _select_hourly_record(records, kickoff)
    if selected is None:
        return None

    temp_f = _c_to_f(selected.get("temperature_c"))
    wind_mph = _mps_to_mph(selected.get("wind_speed_mps"))
    precip_mm = _maybe_float(selected.get("precipitation_mm"))
    precip_prob = None if precip_mm is None else float(precip_mm > 0.0)

    return _WeatherReading(temperature_f=temp_f, wind_mph=wind_mph, precip_probability=precip_prob)


def _reading_to_payload(reading: _WeatherReading) -> Mapping[str, Any]:
    return {
        "wx_temp": reading.temperature_f,
        "wx_wind": reading.wind_mph,
        "precip": reading.precip_probability,
    }


def _select_period(periods: Iterable[Mapping[str, Any]], kickoff: pd.Timestamp) -> Mapping[str, Any] | None:
    kickoff = _ensure_utc(kickoff)
    best_period: Mapping[str, Any] | None = None
    best_distance: pd.Timedelta | None = None

    for period in periods:
        start = _parse_timestamp(period.get("start_time"))
        end = _parse_timestamp(period.get("end_time"))

        if start is None and end is None:
            continue

        if start is not None and end is not None and start <= kickoff < end:
            return period

        midpoint: pd.Timestamp | None = None
        if start is not None and end is not None:
            midpoint = start + (end - start) / 2
        elif start is not None:
            midpoint = start
        elif end is not None:
            midpoint = end

        if midpoint is None:
            continue

        distance = abs(midpoint - kickoff)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_period = period

    return best_period


def _select_hourly_record(records: Iterable[Mapping[str, Any]], kickoff: pd.Timestamp) -> Mapping[str, Any] | None:
    kickoff = _ensure_utc(kickoff)
    best_record: Mapping[str, Any] | None = None
    best_distance: pd.Timedelta | None = None

    for record in records:
        timestamp = _parse_timestamp(record.get("time"))
        if timestamp is None:
            continue

        distance = abs(timestamp - kickoff)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_record = record

    return best_record


def _empty_weather_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["season", "week", "game_id", "team", "wx_temp", "wx_wind", "precip"])


def _normalize_roof(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower() or None


def _ensure_utc(value: pd.Timestamp | Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Series):  # pragma: no cover - defensive
        ts = ts.iloc[0]
    return _ensure_utc(ts)


def _c_to_f(value: Any) -> float | None:
    temp = _maybe_float(value)
    if temp is None:
        return None
    return temp * 9.0 / 5.0 + 32.0


def _mps_to_mph(value: Any) -> float | None:
    speed = _maybe_float(value)
    if speed is None:
        return None
    return speed * 2.23693629


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

