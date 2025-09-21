from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from nfl_pred.weather.storage import (
    WeatherArtifactStore,
    build_artifact_key,
)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = WeatherArtifactStore(base_dir=tmp_path)
    called_at = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    params = {"lat": 39.0, "lon": -77.0, "options": ["hourly", "daily"]}

    metadata = store.save(
        source="nws",
        endpoint="gridpoints/LWX/96,70/forecast",
        params=params,
        payload={"example": True},
        version="v1",
        called_at=called_at,
        ttl_seconds=600.0,
    )

    artifact = store.load("nws", "gridpoints/LWX/96,70/forecast", params)
    assert artifact is not None
    assert artifact.payload["example"] is True
    assert artifact.metadata.key == metadata.key
    assert artifact.metadata.called_at == called_at
    assert artifact.metadata.ttl_expires_at == called_at + timedelta(seconds=600)

    manifest_data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert metadata.key in manifest_data["entries"]


def test_build_artifact_key_is_deterministic(tmp_path: Path) -> None:
    params = {"station": "TEST", "start": "2024-01-01", "end": "2024-01-02"}
    key_a = build_artifact_key("meteostat", "hourly", params)
    key_b = build_artifact_key("meteostat", "hourly", dict(reversed(list(params.items()))))
    assert key_a == key_b

    store = WeatherArtifactStore(base_dir=tmp_path)
    store.save(source="meteostat", endpoint="hourly", params=params, payload={"records": []})
    artifact = store.load_by_key(key_a)
    assert artifact is not None
    assert artifact.metadata.source == "meteostat"
