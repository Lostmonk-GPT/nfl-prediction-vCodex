"""Tests for MLflow hygiene utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.entities import ViewType

from nfl_pred.registry.hygiene import (
    RetentionPolicy,
    apply_standard_tags,
    build_standard_tags,
    enforce_retention_policy,
)


def test_build_standard_tags_normalizes_inputs() -> None:
    payload = build_standard_tags(
        seasons=[2023, "2024"],
        weeks=[5, "06"],
        snapshot_ats=["2024-01-01T00:00:00Z", datetime(2024, 1, 2, tzinfo=timezone.utc)],
        model_id="model_123",
        promoted=False,
        lineage="abc123",
    )

    tags = payload.to_tags()
    assert tags["season"] == "2023|2024"
    assert tags["week"] == "5|6"
    assert "snapshot_at" in tags
    assert tags["model_id"] == "model_123"
    assert tags["promoted"] == "false"
    assert tags["lineage"] == "abc123"


def test_apply_standard_tags_sets_values(tmp_path: Path) -> None:
    tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(str(tracking_dir))
    mlflow.set_experiment("hygiene-tags")

    with mlflow.start_run() as run:
        payload = build_standard_tags(seasons=[2023], weeks=[1], snapshot_ats=[datetime.now(timezone.utc)])
        result = apply_standard_tags(payload, extra_tags={"custom": "value"})

    client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir))
    stored = client.get_run(run.info.run_id)

    assert result["season"] == "2023"
    assert stored.data.tags["season"] == "2023"
    assert stored.data.tags["custom"] == "value"


def test_enforce_retention_policy_prunes_runs(tmp_path: Path) -> None:
    tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(str(tracking_dir))
    mlflow.set_experiment("hygiene-cleanup")

    client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir))

    run_artifacts: dict[str, Path] = {}
    metrics = [0.3, 0.2, 0.4, 0.1]
    for idx, metric in enumerate(metrics):
        with mlflow.start_run() as run:
            mlflow.log_metric("holdout_brier", metric)
            artifact_file = tmp_path / f"artifact_{idx}.txt"
            artifact_file.write_text("payload", encoding="utf-8")
            mlflow.log_artifact(str(artifact_file))
            if idx == 0:
                mlflow.set_tag("promoted", "true")
        artifact_uri = run.info.artifact_uri
        parsed = urlparse(artifact_uri)
        run_artifacts[run.info.run_id] = Path(parsed.path)

    policy = RetentionPolicy(
        max_age_days=None,
        keep_last_runs=1,
        keep_top_runs=1,
        metric="holdout_brier",
        metric_goal="min",
        protect_promoted=True,
    )

    report = enforce_retention_policy(
        tracking_uri=str(tracking_dir),
        experiment="hygiene-cleanup",
        policy=policy,
        dry_run=False,
        delete_artifacts=True,
    )

    assert report.scanned_runs == len(metrics)
    assert report.protected_runs  # promoted run should be protected
    assert report.deleted_runs  # at least one run should be pruned

    remaining = client.search_runs(
        [client.get_experiment_by_name("hygiene-cleanup").experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
    )
    remaining_ids = {run.info.run_id for run in remaining}

    for run_id, path in run_artifacts.items():
        if run_id in report.deleted_runs:
            assert not path.exists()
        else:
            assert run_id in remaining_ids
            assert path.exists()
