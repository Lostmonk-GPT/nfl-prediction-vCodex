from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_pred.features.build_features import FeatureBuildResult
from nfl_pred.pipeline.predict import InferenceResult
from nfl_pred.snapshot.runner import SnapshotRunner, SnapshotStage


def _write_config(path: Path, data_dir: Path, duckdb_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "paths:",
                f"  data_dir: {data_dir}",
                f"  duckdb_path: {duckdb_path}",
                "mlflow:",
                "  tracking_uri: ./mlruns",
                "features:",
                "  windows:",
                "    short: 4",
                "    mid: 8",
            ]
        ),
        encoding="utf-8",
    )


def test_snapshot_runner_executes_stages_in_order(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    duckdb_path = tmp_path / "nfl.duckdb"
    data_dir.mkdir()
    _write_config(config_path, data_dir, duckdb_path)

    season = 2024
    week = 1

    stage_defs = [
        SnapshotStage(name="T-24h", refresh_rosters=True, refresh_injuries=True, feature_write_mode="replace"),
        SnapshotStage(name="T-100m", refresh_injuries=True),
        SnapshotStage(name="T-80-75m", refresh_injuries=True),
        SnapshotStage(name="T-60m", refresh_injuries=True, produce_predictions=True),
    ]

    calls: dict[str, list[object]] = {
        "injuries": [],
        "rosters": [],
        "teams": [],
        "schedule": [],
        "pbp": [],
        "features": [],
        "inference": [],
    }

    def fake_injuries(seasons: list[int]) -> Path:
        calls["injuries"].append(list(seasons))
        return tmp_path / "injuries.parquet"

    def fake_rosters(seasons: list[int]) -> Path:
        calls["rosters"].append(list(seasons))
        return tmp_path / "rosters.parquet"

    def fake_teams() -> Path:
        calls["teams"].append("teams")
        return tmp_path / "teams.parquet"

    def fake_schedule_loader(_: Path, seasons: list[int]) -> pd.DataFrame:
        calls["schedule"].append(list(seasons))
        return pd.DataFrame(
            {
                "season": [season],
                "week": [week],
                "game_id": ["2024_01_BUF_MIA"],
                "team": ["home"],
                "start_time": [pd.Timestamp("2024-09-08T16:25:00Z")],
            }
        )

    def fake_pbp_loader(_: Path, seasons: list[int]) -> pd.DataFrame:
        calls["pbp"].append(list(seasons))
        return pd.DataFrame(
            {
                "season": [season],
                "week": [week],
                "game_id": ["2024_01_BUF_MIA"],
                "posteam": ["BUF"],
                "defteam": ["MIA"],
            }
        )

    def fake_features(
        pbp_df: pd.DataFrame,
        schedule_df: pd.DataFrame,
        *,
        asof_ts: pd.Timestamp,
        snapshot_at: pd.Timestamp,
        feature_set: str,
        write_mode: str,
        duckdb_path: str,
    ) -> FeatureBuildResult:
        calls["features"].append(
            {
                "asof_ts": asof_ts,
                "snapshot_at": snapshot_at,
                "feature_set": feature_set,
                "write_mode": write_mode,
                "duckdb_path": duckdb_path,
                "pbp_rows": len(pbp_df),
                "schedule_rows": len(schedule_df),
            }
        )
        return FeatureBuildResult(
            features_df=pd.DataFrame({"season": [season], "week": [week], "game_id": ["2024_01_BUF_MIA"], "team": ["home"]}),
            payload_df=pd.DataFrame(
                {
                    "season": [season],
                    "week": [week],
                    "game_id": ["2024_01_BUF_MIA"],
                    "team_side": ["home"],
                    "snapshot_at": [snapshot_at],
                    "payload_json": ["{}"],
                }
            ),
        )

    def fake_inference(
        *,
        model_path: Path,
        season: int,
        week: int,
        model_id: str | None,
        feature_set: str,
        feature_snapshot_at: pd.Timestamp,
        snapshot_at: pd.Timestamp,
        config_path: Path,
        duckdb_path: Path,
        write_mode: str,
    ) -> InferenceResult:
        calls["inference"].append(
            {
                "model_path": model_path,
                "season": season,
                "week": week,
                "model_id": model_id,
                "feature_set": feature_set,
                "feature_snapshot_at": feature_snapshot_at,
                "snapshot_at": snapshot_at,
                "config_path": config_path,
                "duckdb_path": duckdb_path,
                "write_mode": write_mode,
            }
        )
        return InferenceResult(
            predictions_df=pd.DataFrame({"game_id": ["2024_01_BUF_MIA"], "snapshot_at": [snapshot_at]}),
            model_path=model_path,
            model_id=model_id or "model",
            season=season,
            week=week,
        )

    runner = SnapshotRunner(
        config_path=config_path,
        stages=stage_defs,
        ingest_injuries_fn=fake_injuries,
        ingest_rosters_fn=fake_rosters,
        ingest_teams_fn=fake_teams,
        schedule_loader=fake_schedule_loader,
        pbp_loader=fake_pbp_loader,
        feature_builder=fake_features,
        inference_runner=fake_inference,
    )

    stage_times = {
        "T-24h": "2024-09-07T16:25:00Z",
        "T-100m": "2024-09-08T14:45:00Z",
        "T-80-75m": ["2024-09-08T15:05:00Z", "2024-09-08T15:10:00Z"],
        "T-60m": "2024-09-08T15:25:00Z",
    }

    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"artifact")

    executions = runner.run(
        season=season,
        week=week,
        stage_times=stage_times,
        model_path=model_path,
        model_id="model",
    )

    assert calls["rosters"] == [[season]]
    assert calls["teams"] == ["teams"]
    assert calls["injuries"] == [[season], [season], [season], [season], [season]]
    assert calls["schedule"] == [[season]]
    assert calls["pbp"] == [[season]]
    assert len(calls["features"]) == 5
    assert len(calls["inference"]) == 1

    expected_order = ["T-24h", "T-100m", "T-80-75m", "T-80-75m", "T-60m"]
    assert [execution.stage.name for execution in executions] == expected_order

    replace_modes = [entry["write_mode"] for entry in calls["features"]]
    assert replace_modes[0] == "replace"
    assert all(mode == "append" for mode in replace_modes[1:])

    assert calls["inference"][0]["feature_snapshot_at"].isoformat() == "2024-09-08T15:25:00+00:00"
    assert executions[-1].prediction_result is not None


def test_missing_stage_timestamp_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    duckdb_path = tmp_path / "nfl.duckdb"
    data_dir.mkdir()
    _write_config(config_path, data_dir, duckdb_path)

    runner = SnapshotRunner(
        config_path=config_path,
        stages=[SnapshotStage(name="T-24h")],
        schedule_loader=lambda *_: pd.DataFrame(),
        pbp_loader=lambda *_: pd.DataFrame(),
        feature_builder=lambda *args, **kwargs: FeatureBuildResult(
            features_df=pd.DataFrame(), payload_df=pd.DataFrame()
        ),
    )

    with pytest.raises(ValueError, match="T-24h"):
        runner.run(season=2024, week=1, stage_times={}, model_path=tmp_path / "model.joblib")


def test_prediction_stage_requires_model_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    duckdb_path = tmp_path / "nfl.duckdb"
    data_dir.mkdir()
    _write_config(config_path, data_dir, duckdb_path)

    stage = SnapshotStage(name="T-60m", produce_predictions=True)

    runner = SnapshotRunner(
        config_path=config_path,
        stages=[stage],
        schedule_loader=lambda *_: pd.DataFrame(),
        pbp_loader=lambda *_: pd.DataFrame(),
        feature_builder=lambda *args, **kwargs: FeatureBuildResult(
            features_df=pd.DataFrame(), payload_df=pd.DataFrame()
        ),
    )

    with pytest.raises(ValueError, match="model_path"):
        runner.run(
            season=2024,
            week=1,
            stage_times={"T-60m": "2024-09-08T15:25:00Z"},
        )
