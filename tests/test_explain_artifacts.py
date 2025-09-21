"""Tests for explainability artifact persistence utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from nfl_pred.explain import ExplainabilityArtifactManager


def _write_source_file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_persist_creates_deterministic_structure_and_logs_metadata(tmp_path: Path) -> None:
    base_dir = tmp_path / "artifacts"
    tracking_dir = tmp_path / "mlruns"

    manager = ExplainabilityArtifactManager(base_dir=base_dir)

    source_values = _write_source_file(tmp_path / "values.parquet", "dummy")
    source_plot = _write_source_file(tmp_path / "summary.png", "plot-bytes")

    run_id: str | None = None
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{tracking_dir}")

    try:
        with mlflow.start_run() as run:
            record = manager.persist(
                "shap",
                {
                    "values": source_values,
                    "summary": source_plot,
                },
                season=2023,
                week=5,
                model_id="model-001",
                description="Week 5 SHAP artifacts",
                extra={"feature_set": "toy"},
            )
            run_id = run.info.run_id
    finally:
        mlflow.set_tracking_uri(previous_uri)

    expected_dir = base_dir / "season=2023" / "week=05" / "model=model-001" / "shap"
    assert expected_dir.exists(), "Artifact directory should be created deterministically."
    assert record.metadata_path == expected_dir / "metadata.json"

    values_path = expected_dir / "values.parquet"
    summary_path = expected_dir / "summary.png"
    assert values_path.read_text(encoding="utf-8") == "dummy"
    assert summary_path.read_text(encoding="utf-8") == "plot-bytes"

    metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    assert metadata["season"] == 2023
    assert metadata["week"] == 5
    assert metadata["model_id"] == "model-001"
    assert metadata["artifact_type"] == "shap"
    assert metadata["description"] == "Week 5 SHAP artifacts"
    assert metadata["extra"] == {"feature_set": "toy"}

    assert run_id is not None
    run_artifact_dir = tracking_dir / "0" / run_id / "artifacts"
    mlflow_path = (
        run_artifact_dir
        / "explain"
        / "season=2023"
        / "week=05"
        / "model=model-001"
        / "shap"
    )
    assert (mlflow_path / "metadata.json").exists(), "Metadata should be logged to MLflow."
    assert (mlflow_path / "values.parquet").exists()
    assert (mlflow_path / "summary.png").exists()


def test_discover_and_cleanup(tmp_path: Path) -> None:
    base_dir = tmp_path / "artifacts"
    manager = ExplainabilityArtifactManager(base_dir=base_dir)

    src = _write_source_file(tmp_path / "src.txt", "a")

    record_one = manager.persist(
        "shap",
        {"values": src},
        season=2023,
        week=5,
        model_id="model-xyz",
    )
    record_two = manager.persist(
        "shap",
        {"values": src},
        season=2023,
        week=6,
        model_id="model-xyz",
    )
    record_three = manager.persist(
        "shap",
        {"values": src},
        season=2023,
        week=7,
        model_id="model-xyz",
    )

    _stamp_metadata(record_one.metadata_path, "2023-01-01T00:00:00+00:00")
    _stamp_metadata(record_two.metadata_path, "2023-01-08T00:00:00+00:00")
    _stamp_metadata(record_three.metadata_path, "2023-01-15T00:00:00+00:00")

    discovered = manager.discover(model_id="model-xyz")
    assert len(discovered) == 3

    removed_before = manager.cleanup(before=datetime(2023, 1, 6, tzinfo=timezone.utc))
    assert record_one.metadata_path.parent in removed_before

    remaining = manager.discover(model_id="model-xyz")
    assert {rec.metadata.week for rec in remaining} == {6, 7}

    removed_max = manager.cleanup(max_per_model=1)
    assert record_two.metadata_path.parent in removed_max

    final_records = manager.discover(model_id="model-xyz")
    assert len(final_records) == 1
    assert final_records[0].metadata.week == 7


def _stamp_metadata(path: Path, created_at: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["created_at"] = created_at
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

