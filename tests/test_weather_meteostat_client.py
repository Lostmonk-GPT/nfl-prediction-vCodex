from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_pred.weather.meteostat_client import (
    MeteostatClient,
    MeteostatClientError,
    StationRecord,
)
from nfl_pred.weather.storage import WeatherArtifactStore


class _StationsStub:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.requested: tuple[float, float] | None = None

    def nearby(self, lat: float, lon: float) -> "_StationsStub":
        self.requested = (lat, lon)
        return self

    def fetch(self, limit: int | None = None) -> pd.DataFrame:
        return self._df


class _HourlyStub:
    def __init__(self, station: str, start: datetime, end: datetime) -> None:
        assert station == "TEST"
        self._df = (
            pd.DataFrame(
                {
                    "time": [pd.Timestamp("2024-01-01T12:00:00+00:00")],
                    "temp": [10.0],
                    "dwpt": [2.0],
                    "rhum": [80.0],
                    "prcp": [1.2],
                    "snow": [0.0],
                    "wdir": [270.0],
                    "wspd": [18.0],
                    "wpgt": [36.0],
                    "pres": [1015.0],
                    "tsun": [30.0],
                }
            ).set_index("time")
        )

    def fetch(self) -> pd.DataFrame:
        return self._df


class _DailyStub:
    def __init__(self, station: str, start: datetime, end: datetime) -> None:
        assert station == "TEST"
        self._df = (
            pd.DataFrame(
                {
                    "time": [pd.Timestamp("2024-01-01T00:00:00+00:00")],
                    "tavg": [5.0],
                    "tmin": [1.0],
                    "tmax": [9.0],
                    "prcp": [2.5],
                    "snow": [0.2],
                    "wdir": [180.0],
                    "wspd": [10.0],
                    "wpgt": [20.0],
                    "pres": [1008.0],
                    "tsun": [120.0],
                }
            ).set_index("time")
        )

    def fetch(self) -> pd.DataFrame:
        return self._df


def test_nearest_station_returns_normalized_record() -> None:
    df = pd.DataFrame(
        [
            {
                "id": "NEAR",
                "name": "Nearby Station",
                "latitude": 40.05,
                "longitude": -75.0,
                "elevation": 100.0,
                "timezone": "America/New_York",
            },
            {
                "id": "FAR",
                "name": "Far Station",
                "latitude": 41.0,
                "longitude": -75.0,
                "elevation": 200.0,
                "timezone": "America/New_York",
            },
        ]
    )
    stub = _StationsStub(df)
    client = MeteostatClient(stations_factory=lambda: stub)

    record = client.nearest_station(40.0, -75.0)

    assert record.station_id == "NEAR"
    assert record.name == "Nearby Station"
    assert record.timezone == "America/New_York"
    assert record.distance_miles < 10.0
    assert stub.requested == (40.0, -75.0)


def test_nearest_station_raises_when_outside_radius() -> None:
    df = pd.DataFrame(
        [
            {
                "id": "FAR",
                "name": "Far Station",
                "latitude": 41.0,
                "longitude": -75.0,
            }
        ]
    )
    stub = _StationsStub(df)
    client = MeteostatClient(stations_factory=lambda: stub, max_station_distance_miles=5.0)

    with pytest.raises(MeteostatClientError):
        client.nearest_station(40.0, -75.0)


def test_hourly_normalization_converts_units_and_persists_raw(tmp_path: Path) -> None:
    store = WeatherArtifactStore(base_dir=tmp_path)
    client = MeteostatClient(
        stations_factory=lambda: _StationsStub(pd.DataFrame()),
        hourly_cls=_HourlyStub,
        artifact_store=store,
        artifact_version="test-suite",
        artifact_ttl_seconds=3600.0,
    )

    result = client.hourly("TEST", datetime(2024, 1, 1), datetime(2024, 1, 2))

    assert len(result) == 1
    record = result[0]
    assert record["temperature_c"] == pytest.approx(10.0)
    assert record["wind_speed_mps"] == pytest.approx(5.0)
    assert record["wind_gust_mps"] == pytest.approx(10.0)
    assert record["time"].startswith("2024-01-01T12:00:00")

    artifact = store.load(
        "meteostat",
        "hourly",
        {
            "station_id": "TEST",
            "start": datetime(2024, 1, 1).isoformat(),
            "end": datetime(2024, 1, 2).isoformat(),
        },
    )
    assert artifact is not None
    assert artifact.payload["records"][0]["temp"] == 10.0
    assert artifact.metadata.version == "test-suite"
    assert artifact.metadata.ttl_expires_at is not None


def test_daily_normalization_supports_station_record_input() -> None:
    client = MeteostatClient(
        stations_factory=lambda: _StationsStub(pd.DataFrame()),
        daily_cls=_DailyStub,
    )

    station = StationRecord(
        station_id="TEST",
        name="Station",
        latitude=40.0,
        longitude=-75.0,
        distance_km=1.0,
        distance_miles=0.6,
        elevation_m=10.0,
        timezone="America/New_York",
    )

    result = client.daily(station, datetime(2024, 1, 1), datetime(2024, 1, 2))

    assert len(result) == 1
    record = result[0]
    assert record["temperature_avg_c"] == pytest.approx(5.0)
    assert record["wind_speed_mps"] == pytest.approx(10.0 / 3.6)
    assert record["date"].startswith("2024-01-01T00:00:00")
