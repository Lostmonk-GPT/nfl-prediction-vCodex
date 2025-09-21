from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from nfl_pred.weather.nws_client import NWSClient, NWSClientError


@dataclass
class _StubResponse:
    payload: Mapping[str, Any]
    status_code: int = 200
    headers: Mapping[str, str] | None = None

    def json(self) -> Mapping[str, Any]:
        return self.payload


class _StubTransport:
    def __init__(self, responses: dict[str, list[_StubResponse]]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: float | tuple[float, float] | None = None,
    ) -> _StubResponse:
        self.calls.append(url)
        try:
            responses = self._responses[url]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected URL requested: {url}") from exc
        if not responses:
            raise AssertionError(f"No stub response remaining for {url}")
        return responses.pop(0)


POINT_PAYLOAD = {
    "properties": {
        "gridId": "LWX",
        "gridX": 96,
        "gridY": 70,
        "forecast": "https://api.weather.gov/gridpoints/LWX/96,70/forecast",
        "forecastHourly": "https://api.weather.gov/gridpoints/LWX/96,70/forecast/hourly",
        "observationStations": "https://api.weather.gov/gridpoints/LWX/96,70/stations",
        "timeZone": "America/New_York",
        "relativeLocation": {
            "geometry": {
                "type": "Point",
                "coordinates": [-77.4705, 39.0438],
            },
            "properties": {
                "distance": {"unitCode": "wmoUnit:m", "value": 8046.72},
                "city": "Ashburn",
                "state": "VA",
            },
        },
    }
}


FORECAST_PAYLOAD = {
    "properties": {
        "updated": "2024-04-01T12:00:00+00:00",
        "generatedAt": "2024-04-01T11:59:00+00:00",
        "units": "us",
        "elevation": {"unitCode": "wmoUnit:m", "value": 120.0},
        "periods": [
            {
                "number": 1,
                "name": "Today",
                "startTime": "2024-04-01T13:00:00+00:00",
                "endTime": "2024-04-01T19:00:00+00:00",
                "isDaytime": True,
                "temperature": 70,
                "temperatureUnit": "F",
                "shortForecast": "Sunny",
                "detailedForecast": "Sunny with light winds",
                "windSpeed": "5 to 10 mph",
                "windGust": "20 mph",
                "windDirection": "SW",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 10},
            },
            {
                "number": 2,
                "name": "Tonight",
                "startTime": "2024-04-01T19:00:00+00:00",
                "endTime": "2024-04-02T05:00:00+00:00",
                "isDaytime": False,
                "temperature": 55,
                "temperatureUnit": "F",
                "shortForecast": "Clear",
                "detailedForecast": "Clear with calm winds",
                "windSpeed": "5 mph",
                "windDirection": "NW",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 0},
            },
        ],
    }
}


@pytest.fixture
def transport_factory() -> tuple[_StubTransport, dict[str, list[_StubResponse]]]:
    responses: dict[str, list[_StubResponse]] = {}
    transport = _StubTransport(responses)
    return transport, responses


def _build_client(
    transport: _StubTransport,
    *,
    now_values: list[float] | None = None,
    sleep_calls: list[float] | None = None,
) -> NWSClient:
    now_iter = iter(now_values or [0.0, 0.0, 0.0])

    def fake_now() -> float:
        try:
            return next(now_iter)
        except StopIteration:  # pragma: no cover - defensive
            return 9999.0

    sleep_log = sleep_calls if sleep_calls is not None else []

    def fake_sleep(seconds: float) -> None:
        sleep_log.append(seconds)

    return NWSClient(
        transport=transport,
        metadata_cache_ttl=60.0,
        forecast_cache_ttl=60.0,
        monotonic=fake_now,
        sleep=fake_sleep,
    )


def test_point_metadata_normalizes_fields_and_caches(transport_factory: tuple[_StubTransport, dict[str, list[_StubResponse]]]) -> None:
    transport, responses = transport_factory
    url = "https://api.weather.gov/points/39.0,-77.0"
    responses[url] = [_StubResponse(POINT_PAYLOAD)]

    client = _build_client(transport, now_values=[0.0, 1.0])

    result = client.point_metadata(39.0, -77.0)
    assert result["grid_id"] == "LWX"
    assert result["grid_x"] == 96
    assert result["grid_y"] == 70
    assert pytest.approx(result["distance_miles"], rel=1e-3) == 5.0
    assert result["within_10_miles"] is True

    cached = client.point_metadata(39.0, -77.0)
    assert cached is result  # cache returns same object
    assert transport.calls == [url]


def test_gridpoint_forecast_normalizes_units_and_caches(transport_factory: tuple[_StubTransport, dict[str, list[_StubResponse]]]) -> None:
    transport, responses = transport_factory
    url = "https://api.weather.gov/gridpoints/LWX/96,70/forecast"
    responses[url] = [_StubResponse(FORECAST_PAYLOAD)]

    client = _build_client(transport, now_values=[0.0, 1.0])
    forecast = client.gridpoint_forecast("LWX", 96, 70)

    assert forecast["elevation_m"] == pytest.approx(120.0)
    assert len(forecast["periods"]) == 2
    first_period = forecast["periods"][0]
    assert first_period["temperature_c"] == pytest.approx(21.111, rel=1e-3)
    assert first_period["wind_speed_mps"] == pytest.approx(3.3528, rel=1e-3)
    assert first_period["wind_gust_mps"] == pytest.approx(8.9408, rel=1e-3)
    assert first_period["probability_of_precipitation_pct"] == 10

    # second call served from cache
    cached = client.gridpoint_forecast("lwx", 96, 70)
    assert cached is forecast
    assert transport.calls == [url]


def test_hourly_forecast_uses_distinct_cache_key(transport_factory: tuple[_StubTransport, dict[str, list[_StubResponse]]]) -> None:
    transport, responses = transport_factory
    base = "https://api.weather.gov/gridpoints/LWX/96,70/forecast"
    responses[base] = [_StubResponse(FORECAST_PAYLOAD)]
    hourly_url = f"{base}/hourly"
    responses[hourly_url] = [_StubResponse(FORECAST_PAYLOAD)]

    client = _build_client(transport, now_values=[0.0, 0.0, 0.0])

    _ = client.gridpoint_forecast("LWX", 96, 70, hourly=False)
    _ = client.gridpoint_forecast("LWX", 96, 70, hourly=True)

    assert transport.calls == [base, hourly_url]


def test_backoff_respects_retry_after_header(transport_factory: tuple[_StubTransport, dict[str, list[_StubResponse]]]) -> None:
    transport, responses = transport_factory
    url = "https://api.weather.gov/points/39.0,-77.0"
    responses[url] = [
        _StubResponse(POINT_PAYLOAD, status_code=429, headers={"Retry-After": "1"}),
        _StubResponse(POINT_PAYLOAD),
    ]

    sleep_calls: list[float] = []
    client = _build_client(transport, now_values=[0.0, 1.0, 2.0], sleep_calls=sleep_calls)

    result = client.point_metadata(39.0, -77.0)
    assert result["grid_id"] == "LWX"
    assert sleep_calls == [1.0]


def test_raises_error_after_retries_exhausted(transport_factory: tuple[_StubTransport, dict[str, list[_StubResponse]]]) -> None:
    transport, responses = transport_factory
    url = "https://api.weather.gov/points/39.0,-77.0"
    responses[url] = [
        _StubResponse(POINT_PAYLOAD, status_code=500),
        _StubResponse(POINT_PAYLOAD, status_code=500),
        _StubResponse(POINT_PAYLOAD, status_code=500),
    ]

    client = _build_client(transport, now_values=[0.0, 1.0, 2.0])

    with pytest.raises(NWSClientError):
        client.point_metadata(39.0, -77.0)


def test_cache_expires_after_ttl(transport_factory: tuple[_StubTransport, dict[str, list[_StubResponse]]]) -> None:
    transport, responses = transport_factory
    url = "https://api.weather.gov/gridpoints/LWX/96,70/forecast"
    responses[url] = [_StubResponse(FORECAST_PAYLOAD), _StubResponse(FORECAST_PAYLOAD)]

    client = _build_client(transport, now_values=[0.0, 10.0, 70.0])

    _ = client.gridpoint_forecast("LWX", 96, 70)
    _ = client.gridpoint_forecast("LWX", 96, 70)
    _ = client.gridpoint_forecast("LWX", 96, 70)

    assert transport.calls == [url, url]
