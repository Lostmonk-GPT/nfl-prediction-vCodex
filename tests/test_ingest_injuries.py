"""Tests for the injuries ingestion routine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.api.types import DatetimeTZDtype

from nfl_pred.ingest.injuries import ingest_injuries


class _DummyPolarsFrame:
    """Minimal stub mimicking the ``to_pandas`` interface of a Polars frame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self, use_pyarrow_extension_array: bool = True) -> pd.DataFrame:  # noqa: ARG002
        return self._df


def test_ingest_injuries_persists_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seasons = [2022, 2023]
    event_times = {
        2022: datetime(2022, 9, 1, 12, 0, tzinfo=timezone.utc),
        2023: datetime(2023, 9, 2, 15, 30, tzinfo=timezone.utc),
    }

    def fake_load_injuries(season: int) -> _DummyPolarsFrame:
        df = pd.DataFrame(
            {
                "season": [season],
                "week": [1],
                "team": ["KC"],
                "date_modified": [event_times[season]],
            }
        )
        return _DummyPolarsFrame(df)

    # Configure paths to remain within the temporary directory.
    data_dir = tmp_path / "data"
    duckdb_path = tmp_path / "duck.db"
    monkeypatch.setenv("NFLPRED__PATHS__DATA_DIR", str(data_dir))
    monkeypatch.setenv("NFLPRED__PATHS__DUCKDB_PATH", str(duckdb_path))

    # Ensure predictable source version metadata.
    monkeypatch.setattr(
        "nfl_pred.ingest.injuries.nflreadpy",
        SimpleNamespace(load_injuries=fake_load_injuries, __version__="1.2.3"),
        raising=False,
    )

    registrations: list[tuple[str, str]] = []
    opened_db_paths: list[str] = []

    class DummyDuckDBClient:
        def __init__(self, db_path: str, read_only: bool = False) -> None:  # noqa: ARG002
            opened_db_paths.append(db_path)

        def __enter__(self) -> "DummyDuckDBClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool | None:  # noqa: ANN001, D401
            return None

        def register_parquet(self, path: str, view: str) -> None:
            registrations.append((path, view))

    monkeypatch.setattr("nfl_pred.ingest.injuries.DuckDBClient", DummyDuckDBClient)

    output_path = ingest_injuries(seasons)

    assert output_path == data_dir / "raw" / "injuries.parquet"
    assert output_path.exists()

    result = pd.read_parquet(output_path)
    assert set(result["season"]) == {2022, 2023}
    assert isinstance(result["event_time"].dtype, DatetimeTZDtype)
    assert set(result["event_time"]) == set(event_times.values())
    assert result["source"].eq("nflreadpy").all()
    assert result["source_version"].eq("1.2.3").all()
    assert isinstance(result["pulled_at"].dtype, DatetimeTZDtype)

    assert registrations == [(str(output_path), "injuries_raw")]
    assert opened_db_paths == [str(duckdb_path)]
