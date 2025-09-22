from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nfl_pred.audit import gather_audit_record, write_audit_record
from nfl_pred.storage.duckdb_client import DuckDBClient


def _write_file(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        with path.open("wb") as handle:
            handle.write(content)
    else:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(content)


def test_gather_audit_record_deterministic(tmp_path: Path) -> None:
    model_path = tmp_path / "model.bin"
    spec_path = tmp_path / "feature.md"
    _write_file(model_path, b"model-bytes")
    _write_file(spec_path, "# feature spec")

    rows = pd.DataFrame(
        [
            {
                "game_id": "2023_01_phi_ne",
                "team": "PHI",
                "value": 1.2345,
                "timestamp": datetime(2024, 1, 10, 18, 30, tzinfo=timezone.utc),
            },
            {
                "game_id": "2023_01_dal_nyg",
                "team": "DAL",
                "value": 0.75,
                "timestamp": pd.Timestamp("2024-01-10T18:30:00Z"),
            },
        ]
    )

    dataset_versions = {"schedules": "v1", "features": "2024.01.10"}

    record = gather_audit_record(
        season=2024,
        week=1,
        snapshot_at=datetime(2024, 1, 10, 19, 0, tzinfo=timezone.utc),
        model_id="baseline",
        dataset_versions=dataset_versions,
        code_version="abcd1234",
        feature_spec=spec_path,
        model_artifact=model_path,
        input_rows=rows.iloc[::-1],  # reversed order should not change hashes
        asof_ts=datetime(2024, 1, 10, 18, 0, tzinfo=timezone.utc),
        metadata={"run_id": "run-001", "parameters": {"lr": 0.1}},
    )

    assert record.season == 2024
    assert record.week == 1
    assert record.model_hash != "unknown"
    assert record.input_row_count == 2
    assert len(record.input_row_hashes) == 2

    # Input row hashes should be sorted and deterministic regardless of row order.
    recomputed = gather_audit_record(
        season=2024,
        week=1,
        snapshot_at=datetime(2024, 1, 10, 19, 0, tzinfo=timezone.utc),
        model_id="baseline",
        dataset_versions={"features": "2024.01.10", "schedules": "v1"},
        code_version="abcd1234",
        feature_spec=spec_path,
        model_artifact=model_path,
        input_rows=rows,
        asof_ts=datetime(2024, 1, 10, 18, 0, tzinfo=timezone.utc),
        metadata={"parameters": {"lr": 0.1}, "run_id": "run-001"},
    )

    assert record.input_row_hashes == recomputed.input_row_hashes
    assert record.input_rows_digest == recomputed.input_rows_digest
    assert record.upstream_versions == recomputed.upstream_versions
    assert record.metadata == recomputed.metadata


def test_write_audit_record_upsert(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.duckdb"
    model_path = tmp_path / "model.bin"
    spec_path = tmp_path / "feature.md"
    _write_file(model_path, b"model-bytes")
    _write_file(spec_path, "# feature spec")

    rows = pd.DataFrame(
        [
            {"game_id": "2023_01_phi_ne", "value": 1.0},
            {"game_id": "2023_01_dal_nyg", "value": 0.5},
        ]
    )

    first_record = gather_audit_record(
        season=2024,
        week=5,
        snapshot_at=datetime(2024, 2, 10, 19, 0, tzinfo=timezone.utc),
        model_id="baseline",
        dataset_versions={"features": "2024.02.10"},
        code_version="abcd1234",
        feature_spec=spec_path,
        model_artifact=model_path,
        input_rows=rows,
    )

    write_audit_record(first_record, duckdb_path=db_path)

    with DuckDBClient(str(db_path)) as client:
        frame = client.read_sql("SELECT * FROM audit")

    assert len(frame) == 1
    stored = frame.iloc[0]
    assert stored["code_version"] == "abcd1234"
    assert stored["model_hash"] == first_record.model_hash
    assert json.loads(stored["upstream_versions_json"]) == first_record.upstream_versions
    assert json.loads(stored["input_row_hashes_json"]) == list(first_record.input_row_hashes)
    assert stored["input_rows_hash"] == first_record.input_rows_digest

    updated_record = gather_audit_record(
        season=2024,
        week=5,
        snapshot_at=datetime(2024, 2, 10, 19, 0, tzinfo=timezone.utc),
        model_id="baseline",
        dataset_versions={"features": "2024.02.10"},
        code_version="abcd9999",
        feature_spec=spec_path,
        model_artifact=model_path,
        input_rows=rows,
    )

    write_audit_record(updated_record, duckdb_path=db_path)

    with DuckDBClient(str(db_path)) as client:
        frame = client.read_sql("SELECT * FROM audit")

    assert len(frame) == 1
    stored = frame.iloc[0]
    assert stored["code_version"] == "abcd9999"
    assert stored["input_rows_hash"] == updated_record.input_rows_digest

