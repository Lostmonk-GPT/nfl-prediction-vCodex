"""Tests for weather feature integration relying on local fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_pred.features.weather import compute_weather_features
from nfl_pred.weather.meteostat_client import MeteostatClient, MeteostatClientError
from nfl_pred.weather.nws_client import NWSClient


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class _StubResponse:
    """Minimal response object for :class:`NWSClient` transport stubs."""

    def __init__(self, payload: Any) -> None:
        self.status_code = 200
        self._payload = payload
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return self._payload


class _StubTransport:
    """Return canned NWS responses based on the requested endpoint."""

    def __init__(self, metadata_payload: Any, forecast_payload: Any) -> None:
        self._metadata = metadata_payload
        self._forecast = forecast_payload

    def request(self, method: str, url: str, **_: Any) -> _StubResponse:
        if "/points/" in url:
            return _StubResponse(self._metadata)
        if "/gridpoints/" in url:
            return _StubResponse(self._forecast)
        raise AssertionError(f"Unexpected URL: {url}")


class _StubFetcher:
    """Simulate the Meteostat ``fetch`` API by returning a prepared frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def fetch(self, limit: int | None = None) -> pd.DataFrame:  # pragma: no cover - signature only
        _ = limit
        return self._frame


class _StubStations:
    """Simulate the ``Stations`` entry point used by :class:`MeteostatClient`."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def nearby(self, latitude: float, longitude: float) -> _StubFetcher:
        self.last_query = (latitude, longitude)
        return _StubFetcher(self._frame)


@pytest.fixture(scope="module")
def nws_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = FIXTURES_DIR / "nws" / "point_metadata.json"
    forecast_path = FIXTURES_DIR / "nws" / "forecast.json"
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    with forecast_path.open("r", encoding="utf-8") as fh:
        forecast = json.load(fh)
    return metadata, forecast


def _load_station_frame(filename: str) -> pd.DataFrame:
    path = FIXTURES_DIR / "meteostat" / filename
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return pd.DataFrame(payload)


def test_forecast_normalization_uses_fixture_units(nws_payloads: tuple[dict[str, Any], dict[str, Any]]) -> None:
    """Weather features convert forecast units from the NWS fixture."""

    metadata_payload, forecast_payload = nws_payloads
    nws_client = NWSClient(
        transport=_StubTransport(metadata_payload, forecast_payload),
        metadata_cache_ttl=None,
        forecast_cache_ttl=None,
        max_retries=1,
    )

    schedule = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 5,
                "game_id": "2024_05_DAL_NYG",
                "home_team": "DAL",
                "away_team": "NYG",
                "stadium": "AT&T Stadium",
                "start_time": "2024-10-10T17:00:00+00:00",
            }
        ]
    )
    stadiums = pd.DataFrame(
        [
            {
                "venue": "AT&T Stadium",
                "teams": ["DAL"],
                "roof": "retractable",
                "surface": "artificial_turf",
                "tz": "America/Chicago",
                "lat": 32.7473,
                "lon": -97.09451,
                "neutral_site": False,
            }
        ]
    )

    features = compute_weather_features(schedule, stadiums, nws_client=nws_client)
    assert len(features) == 2
    assert features["wx_temp"].notna().all()
    assert features["wx_wind"].notna().all()
    assert pytest.approx(features["wx_temp"].iloc[0], rel=1e-5) == 72.0
    assert pytest.approx(features["wx_wind"].iloc[0], rel=1e-5) == 12.0
    assert pytest.approx(features["precip"].iloc[0], rel=1e-5) == 0.40


def test_indoor_roof_defaults_to_null_weather() -> None:
    """Indoor venues receive null/zero weather values regardless of forecast."""

    schedule = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 6,
                "game_id": "2024_06_MIN_DET",
                "home_team": "MIN",
                "away_team": "DET",
                "stadium": "Ford Field",
                "start_time": "2024-10-17T17:00:00+00:00",
            }
        ]
    )
    stadiums = pd.DataFrame(
        [
            {
                "venue": "Ford Field",
                "teams": ["DET"],
                "roof": "indoors",
                "surface": "artificial_turf",
                "tz": "America/Detroit",
                "lat": 42.3400,
                "lon": -83.0456,
                "neutral_site": False,
            }
        ]
    )

    features = compute_weather_features(schedule, stadiums)
    assert features["wx_temp"].isna().all()
    assert (features["wx_wind"] == 0.0).all()
    assert (features["precip"] == 0.0).all()


def test_meteostat_nearest_station_enforces_distance_threshold() -> None:
    """Meteostat nearest station selection respects the 10 mile constraint."""

    base_lat = 32.7473
    base_lon = -97.09451

    near_frame = _load_station_frame("stations_near.json")
    client = MeteostatClient(stations_factory=lambda: _StubStations(near_frame))
    station = client.nearest_station(base_lat, base_lon)
    assert station.station_id == "722596"
    assert station.distance_miles <= 10.0

    far_frame = _load_station_frame("stations_far.json")
    far_client = MeteostatClient(
        stations_factory=lambda: _StubStations(far_frame),
        max_station_distance_miles=10.0,
    )
    with pytest.raises(MeteostatClientError):
        far_client.nearest_station(base_lat, base_lon)
