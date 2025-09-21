from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_pred.pipeline.train import run_training_pipeline
from nfl_pred.storage.duckdb_client import DuckDBClient


def _build_payload(*, team: str, opponent: str, strength: float, is_home: bool, label: int, week: int) -> str:
    payload = {
        "team": team,
        "opponent": opponent,
        "team_strength": strength + 0.1 * week,
        "opp_strength": strength - 0.1 * week,
        "is_home": is_home,
        "label_team_win": label,
    }
    return json.dumps(payload, sort_keys=True)


def test_run_training_pipeline_creates_artifacts(tmp_path: Path) -> None:
    duckdb_path = tmp_path / "toy.duckdb"
    data_dir = tmp_path / "artifacts"
    tracking_dir = tmp_path / "mlruns"

    records: list[dict[str, object]] = []
    base_timestamp = pd.Timestamp("2023-08-01", tz="UTC")
    teams = [("BUF", "NYJ"), ("KC", "LAC")]

    for week in range(1, 6):
        game_id = f"2023_{week:02d}_001"
        asof_ts = base_timestamp + pd.Timedelta(days=7 * week)
        snapshot_at = asof_ts + pd.Timedelta(hours=2)
        home_team, away_team = teams[week % len(teams)]

        home_label = 1 if week % 2 == 0 else 0
        away_label = 1 - home_label

        records.append(
            {
                "season": 2023,
                "week": week,
                "game_id": game_id,
                "team_side": "home",
                "asof_ts": asof_ts,
                "snapshot_at": snapshot_at,
                "feature_set": "toy",
                "payload_json": _build_payload(
                    team=home_team,
                    opponent=away_team,
                    strength=0.6,
                    is_home=True,
                    label=home_label,
                    week=week,
                ),
            }
        )
        records.append(
            {
                "season": 2023,
                "week": week,
                "game_id": game_id,
                "team_side": "away",
                "asof_ts": asof_ts,
                "snapshot_at": snapshot_at,
                "feature_set": "toy",
                "payload_json": _build_payload(
                    team=away_team,
                    opponent=home_team,
                    strength=0.55,
                    is_home=False,
                    label=away_label,
                    week=week,
                ),
            }
        )

    features_df = pd.DataFrame.from_records(records)

    with DuckDBClient(str(duckdb_path)) as client:
        client.apply_schema()
        client.write_df(features_df, table="features", mode="replace")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"paths:",
                f"  data_dir: {data_dir}",
                f"  duckdb_path: {duckdb_path}",
                f"mlflow:",
                f"  tracking_uri: {tracking_dir}",
                f"features:",
                f"  windows:",
                f"    short: 2",
                f"    mid: 4",
            ]
        ),
        encoding="utf-8",
    )

    result = run_training_pipeline(
        config_path=config_path,
        feature_set="toy",
        min_train_weeks=2,
        calibration_weeks=1,
        n_splits=None,
        random_state=7,
    )

    assert result.model_path.exists()
    assert result.model_path.stat().st_size > 0
    assert result.reliability_plot_path.exists()
    assert result.metrics["holdout_brier"] >= 0
    assert result.metrics["cv_mean_brier"] >= 0
    assert result.fold_metrics, "Expected cross-validation metrics."

    mlflow_run_dir = tracking_dir / "0" / result.run_id
    assert mlflow_run_dir.exists(), "MLflow run directory should exist."
