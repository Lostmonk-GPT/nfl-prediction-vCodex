"""Client for interacting with the National Weather Service API."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Protocol

import requests


class ResponseProtocol(Protocol):
    """Protocol describing the subset of ``requests.Response`` that we use."""

    status_code: int
    headers: Mapping[str, str]

    def json(self) -> Any:  # pragma: no cover - interface declaration
        """Return the decoded JSON payload."""


class TransportProtocol(Protocol):
    """Protocol describing the transport used by :class:`NWSClient`."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: float | tuple[float, float] | None = None,
    ) -> ResponseProtocol:  # pragma: no cover - interface declaration
        """Perform an HTTP request and return a response object."""


RawStore = Callable[[str, Mapping[str, Any]], None]


class NWSClientError(RuntimeError):
    """Raised when the NWS API returns an unexpected response."""


@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


class _TTLCache:
    """A minimal TTL cache used to memoize API responses."""

    def __init__(self, ttl_seconds: float | None, now: Callable[[], float]) -> None:
        self._ttl = ttl_seconds
        self._now = now
        self._store: MutableMapping[str, _CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        if self._ttl is None:
            return None

        entry = self._store.get(key)
        if entry is None:
            return None

        if entry.expires_at <= self._now():
            self._store.pop(key, None)
            return None

        return entry.value

    def set(self, key: str, value: Any) -> None:
        if self._ttl is None:
            return

        self._store[key] = _CacheEntry(expires_at=self._now() + self._ttl, value=value)

    def clear(self) -> None:
        self._store.clear()


_DEFAULT_USER_AGENT = "nfl-prediction/0.0 (+https://example.com)"


class NWSClient:
    """HTTP client that wraps the National Weather Service API."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.weather.gov",
        user_agent: str = _DEFAULT_USER_AGENT,
        timeout: float = 10.0,
        transport: TransportProtocol | None = None,
        raw_store: RawStore | None = None,
        metadata_cache_ttl: float | None = 6 * 60 * 60,
        forecast_cache_ttl: float | None = 15 * 60,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._raw_store = raw_store
        self._headers = {
            "User-Agent": user_agent,
            "Accept": "application/geo+json",
        }
        self._max_retries = max(1, max_retries)
        self._backoff_factor = backoff_factor
        self._sleep = sleep or time.sleep
        self._now = monotonic or time.monotonic

        self._point_cache = _TTLCache(metadata_cache_ttl, self._now)
        self._forecast_cache = _TTLCache(forecast_cache_ttl, self._now)
        self._session: requests.Session | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def point_metadata(self, lat: float, lon: float) -> dict[str, Any]:
        """Return grid metadata for the provided latitude and longitude."""

        cache_key = f"{lat:.4f},{lon:.4f}"
        cached = self._point_cache.get(cache_key)
        if cached is not None:
            return cached

        path = f"points/{lat},{lon}"
        payload = self._request_json(path)
        normalized = _normalize_point_metadata(payload, lat, lon)
        self._point_cache.set(cache_key, normalized)
        return normalized

    def gridpoint_forecast(
        self,
        wfo: str,
        x: int,
        y: int,
        *,
        hourly: bool = False,
    ) -> dict[str, Any]:
        """Return a normalized gridpoint forecast."""

        suffix = "hourly" if hourly else "regular"
        cache_key = f"{wfo.upper()}_{x}_{y}_{suffix}"
        cached = self._forecast_cache.get(cache_key)
        if cached is not None:
            return cached

        endpoint = f"gridpoints/{wfo}/{x},{y}/forecast"
        if hourly:
            endpoint += "/hourly"

        payload = self._request_json(endpoint)
        normalized = _normalize_forecast(payload)
        self._forecast_cache.set(cache_key, normalized)
        return normalized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _request_json(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        transport = self._transport or self._get_session()

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            response = transport.request(
                "GET",
                url,
                headers=self._headers,
                params=params,
                timeout=self._timeout,
            )

            status = response.status_code
            if status == 429 or 500 <= status < 600:
                retry_after = _retry_after_seconds(response.headers)
                if attempt >= self._max_retries - 1:
                    break
                self._sleep(retry_after or self._compute_backoff(attempt))
                continue

            if status >= 400:
                last_error = NWSClientError(
                    f"NWS API returned status {status} for endpoint '{endpoint}'."
                )
                break

            payload = response.json()
            if self._raw_store is not None:
                try:
                    self._raw_store(endpoint, payload)
                except Exception:  # pragma: no cover - defensive
                    pass
            return payload

        if last_error is None:
            last_error = NWSClientError(
                f"NWS API failed after {self._max_retries} attempts for '{endpoint}'."
            )
        raise last_error

    def _get_session(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            session.headers.update(self._headers)
            self._session = session
        return self._session

    def _compute_backoff(self, attempt: int) -> float:
        # Exponential backoff with jitter-friendly minimum floor.
        delay = self._backoff_factor * (2 ** attempt)
        return max(0.1, delay)


# ----------------------------------------------------------------------
# Normalization helpers
# ----------------------------------------------------------------------

def _retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    if not headers:
        return None

    retry_after = headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return float(retry_after)
    except ValueError:
        return None


def _normalize_point_metadata(
    payload: Mapping[str, Any], lat: float, lon: float
) -> dict[str, Any]:
    properties = payload.get("properties", {})
    relative = properties.get("relativeLocation", {})
    rel_props = relative.get("properties", {})
    distance = rel_props.get("distance", {})

    distance_m = _convert_length_to_meters(distance.get("value"), distance.get("unitCode"))
    distance_miles = _meters_to_miles(distance_m) if distance_m is not None else None

    normalized: dict[str, Any] = {
        "input_latitude": lat,
        "input_longitude": lon,
        "grid_id": properties.get("gridId"),
        "grid_x": properties.get("gridX"),
        "grid_y": properties.get("gridY"),
        "forecast_url": properties.get("forecast"),
        "forecast_hourly_url": properties.get("forecastHourly"),
        "observation_stations_url": properties.get("observationStations"),
        "timezone": properties.get("timeZone"),
        "city": rel_props.get("city"),
        "state": rel_props.get("state"),
        "relative_location": relative.get("geometry"),
        "distance_m": distance_m,
        "distance_miles": distance_miles,
        "within_10_miles": distance_miles is not None and distance_miles <= 10.0,
    }

    return normalized


def _normalize_forecast(payload: Mapping[str, Any]) -> dict[str, Any]:
    properties = payload.get("properties", {})
    elevation = properties.get("elevation", {})
    elevation_m = _convert_length_to_meters(elevation.get("value"), elevation.get("unitCode"))

    periods = []
    for period in properties.get("periods", []) or []:
        temperature_c = _normalize_temperature(period.get("temperature"), period.get("temperatureUnit"))
        precip_prob = period.get("probabilityOfPrecipitation", {}) or {}
        precip_value = precip_prob.get("value")
        precip_unit = precip_prob.get("unitCode")
        precip_pct = _convert_probability_to_percent(precip_value, precip_unit)

        wind_speed_mps = _parse_wind_speed(period.get("windSpeed"))
        wind_gust_mps = _parse_wind_speed(period.get("windGust"))

        periods.append(
            {
                "number": period.get("number"),
                "name": period.get("name"),
                "start_time": period.get("startTime"),
                "end_time": period.get("endTime"),
                "is_daytime": period.get("isDaytime"),
                "temperature_c": temperature_c,
                "temperature_original": period.get("temperature"),
                "short_forecast": period.get("shortForecast"),
                "detailed_forecast": period.get("detailedForecast"),
                "wind_direction": period.get("windDirection"),
                "wind_speed_mps": wind_speed_mps,
                "wind_speed": period.get("windSpeed"),
                "wind_gust_mps": wind_gust_mps,
                "wind_gust": period.get("windGust"),
                "probability_of_precipitation_pct": precip_pct,
            }
        )

    normalized = {
        "updated_at": properties.get("updated"),
        "generated_at": properties.get("generatedAt"),
        "units": properties.get("units"),
        "elevation_m": elevation_m,
        "periods": periods,
    }

    return normalized


def _normalize_temperature(value: Any, unit: str | None) -> float | None:
    if value is None:
        return None

    try:
        temp = float(value)
    except (TypeError, ValueError):
        return None

    if unit is None:
        return temp

    unit = unit.upper()
    if unit == "F":
        return (temp - 32.0) * (5.0 / 9.0)
    if unit == "C":
        return temp
    if unit in {"K", "KELVIN"}:
        return temp - 273.15
    return temp


def _parse_wind_speed(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        mph = float(value)
    else:
        numbers = [float(match) for match in _FIND_NUMBER_PATTERN.findall(str(value))]
        if not numbers:
            return None
        mph = sum(numbers) / len(numbers)

    return mph * 0.44704


def _convert_probability_to_percent(value: Any, unit: str | None) -> float | None:
    if value is None:
        return None

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None

    if unit is None:
        return probability

    unit = unit.lower()
    if unit in {"wmoUnit:percent", "percent", "%"}:
        return probability
    if unit in {"wmoUnit:proportion", "proportion"}:
        return probability * 100.0
    return probability


def _convert_length_to_meters(value: Any, unit: str | None) -> float | None:
    if value is None:
        return None

    try:
        magnitude = float(value)
    except (TypeError, ValueError):
        return None

    if unit is None:
        return magnitude

    unit = unit.lower()
    if unit in {"wmounit:m", "m", "meter", "meters"}:
        return magnitude
    if unit in {"wmounit:km", "km", "kilometer", "kilometers"}:
        return magnitude * 1000.0
    if unit in {"wmounit:ft_us", "ft", "foot", "feet"}:
        return magnitude * 0.3048
    if unit in {"wmounit:mi_us", "mi", "mile", "miles"}:
        return magnitude * 1609.344
    return magnitude


def _meters_to_miles(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1609.344


_FIND_NUMBER_PATTERN = __import__("re").compile(r"\d+(?:\.\d+)?")


# ----------------------------------------------------------------------
# Module-level convenience wrappers
# ----------------------------------------------------------------------

_default_client: NWSClient | None = None


def _get_default_client() -> NWSClient:
    global _default_client
    if _default_client is None:
        _default_client = NWSClient()
    return _default_client


def point_metadata(lat: float, lon: float, *, client: NWSClient | None = None) -> dict[str, Any]:
    """Return point metadata using the shared :class:`NWSClient` instance."""

    active_client = client or _get_default_client()
    return active_client.point_metadata(lat, lon)


def gridpoint_forecast(
    wfo: str,
    x: int,
    y: int,
    *,
    hourly: bool = False,
    client: NWSClient | None = None,
) -> dict[str, Any]:
    """Return gridpoint forecast using the shared :class:`NWSClient` instance."""

    active_client = client or _get_default_client()
    return active_client.gridpoint_forecast(wfo, x, y, hourly=hourly)


__all__ = ["NWSClient", "point_metadata", "gridpoint_forecast", "NWSClientError"]
