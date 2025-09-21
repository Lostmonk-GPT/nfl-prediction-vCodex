import json
from pathlib import Path

import pandas as pd
import pytest

from nfl_pred.pipeline.predict import run_inference_pipeline
from nfl_pred.pipeline.train import run_training_pipeline
from nfl_pred.storage.duckdb_client import DuckDBClient


def _build_payload(
    *,
    team: str,
    opponent: str,
    strength: float,
    is_home: bool,
    label: int | None,
    week: int,
) -> str:
    payload = {
        "team": team,
        "opponent": opponent,
        "team_strength": strength + 0.1 * week,
        "opp_strength": strength - 0.1 * week,
        "is_home": is_home,
        "label_team_win": label,
    }
    return json.dumps(payload, sort_keys=True)


def test_run_inference_pipeline_generates_predictions(tmp_path: Path) -> None:
    duckdb_path = tmp_path / "toy.duckdb"
    data_dir = tmp_path / "artifacts"
    tracking_dir = tmp_path / "mlruns"

    records: list[dict[str, object]] = []
    base_timestamp = pd.Timestamp("2023-08-01", tz="UTC")
    teams = [("BUF", "NYJ"), ("KC", "LAC")]

    for week in range(1, 7):
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

    # Upcoming week without labels for inference
    week = 7
    game_id = f"2023_{week:02d}_001"
    asof_ts = base_timestamp + pd.Timedelta(days=7 * week)
    snapshot_at = asof_ts + pd.Timedelta(hours=2)
    home_team, away_team = teams[week % len(teams)]

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
                strength=0.62,
                is_home=True,
                label=None,
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
                strength=0.57,
                is_home=False,
                label=None,
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

    train_result = run_training_pipeline(
        config_path=config_path,
        feature_set="toy",
        min_train_weeks=2,
        calibration_weeks=1,
        n_splits=None,
        random_state=11,
    )

    prediction_result = run_inference_pipeline(
        model_path=train_result.model_path,
        season=2023,
        week=7,
        model_id="baseline_toy",
        feature_set="toy",
        snapshot_at=pd.Timestamp("2023-09-20", tz="UTC"),
        config_path=config_path,
    )

    predictions_df = prediction_result.predictions_df
    assert not predictions_df.empty
    assert set(["p_home_win", "p_away_win"]).issubset(predictions_df.columns)
    assert predictions_df.loc[0, "p_home_win"] + predictions_df.loc[0, "p_away_win"] == pytest.approx(1.0)
    assert predictions_df.loc[0, "model_id"] == "baseline_toy"

    with DuckDBClient(str(duckdb_path)) as client:
        stored = client.read_sql(
            "SELECT p_home_win, p_away_win, model_id FROM predictions WHERE season = 2023 AND week = 7"
        )

    assert len(stored) == 1
    assert stored.loc[0, "model_id"] == "baseline_toy"
    assert stored.loc[0, "p_home_win"] + stored.loc[0, "p_away_win"] == pytest.approx(1.0)
