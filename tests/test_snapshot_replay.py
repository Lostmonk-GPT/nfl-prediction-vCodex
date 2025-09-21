from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_pred.features.build_features import FeatureBuildResult
from nfl_pred.pipeline.predict import InferenceResult
from nfl_pred.snapshot.runner import SnapshotRunner, SnapshotStage
from nfl_pred.snapshot.visibility import VisibilityContext, filter_play_by_play, filter_schedule
from nfl_pred.visibility import filter_visible_rows


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


def test_snapshot_replay_excludes_post_cutoff_rows(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    duckdb_path = tmp_path / "nfl.duckdb"
    data_dir.mkdir()
    _write_config(config_path, data_dir, duckdb_path)

    season = 2023
    week = 5
    game_id = "2023_05_BUF_JAX"
    cutoff = pd.Timestamp("2023-10-08T13:30:00Z")

    schedule_df = pd.DataFrame(
        {
            "season": [season, season],
            "week": [week, week + 1],
            "game_id": [game_id, "2023_06_BUF_NE"],
            "team": ["home", "home"],
            "start_time": [
                pd.Timestamp("2023-10-08T12:30:00Z"),
                pd.Timestamp("2023-10-15T17:00:00Z"),
            ],
        }
    )

    pbp_df = pd.DataFrame(
        {
            "season": [season] * 3,
            "week": [week] * 3,
            "game_id": [game_id] * 3,
            "posteam": ["BUF", "BUF", "JAX"],
            "defteam": ["JAX", "JAX", "BUF"],
            "event_time": [
                pd.Timestamp("2023-10-08T12:45:00Z"),
                pd.Timestamp("2023-10-08T13:15:00Z"),
                pd.Timestamp("2023-10-08T13:45:00Z"),
            ],
            "asof_ts": [
                pd.Timestamp("2023-10-08T12:50:00Z"),
                pd.Timestamp("2023-10-08T13:20:00Z"),
                pd.Timestamp("2023-10-08T13:50:00Z"),
            ],
            "epa": [0.1, -0.2, 0.3],
            "success": [1, 0, 1],
            "down": [1, 2, 3],
            "pass": [1, 0, 0],
            "rush": [0, 1, 1],
            "qb_dropback": [1, 0, 0],
            "sack": [0, 0, 0],
            "play_action": [0, 0, 0],
            "shotgun": [1, 0, 0],
            "no_huddle": [0, 0, 0],
            "penalty": [0, 0, 0],
            "special_teams_play": [0, 0, 0],
            "yards_gained": [12, 5, 8],
        }
    )

    injuries_df = pd.DataFrame(
        {
            "season": [season, season],
            "week": [week, week],
            "team": ["BUF", "BUF"],
            "player": ["Player A", "Player B"],
            "event_time": [
                pd.Timestamp("2023-10-08T12:00:00Z"),
                pd.Timestamp("2023-10-08T14:00:00Z"),
            ],
        }
    )

    captured: dict[str, pd.DataFrame] = {}

    def fake_ingest_injuries(seasons: list[int]) -> Path:
        assert seasons == [season]
        return tmp_path / "injuries.parquet"

    def fake_schedule_loader(_: Path, seasons: list[int]) -> pd.DataFrame:
        assert seasons == [season]
        return schedule_df

    def fake_pbp_loader(_: Path, seasons: list[int]) -> pd.DataFrame:
        assert seasons == [season]
        return pbp_df

    def fake_feature_builder(
        pbp: pd.DataFrame,
        schedule: pd.DataFrame,
        *,
        asof_ts: pd.Timestamp,
        snapshot_at: pd.Timestamp,
        feature_set: str,
        write_mode: str,
        duckdb_path: str,
    ) -> FeatureBuildResult:
        assert feature_set == "mvp_v1"
        assert write_mode == "append"
        context = VisibilityContext(season=season, week=week, asof_ts=asof_ts)

        pbp_visible = filter_play_by_play(pbp, context=context)
        schedule_visible = filter_schedule(schedule, context=context)
        injuries_visible = filter_visible_rows(
            injuries_df,
            season=season,
            week=week,
            asof_ts=asof_ts,
            event_time_col="event_time",
        )

        captured["pbp_visible"] = pbp_visible
        captured["schedule_visible"] = schedule_visible
        captured["injuries_visible"] = injuries_visible

        payload = {
            "visible_plays": int(pbp_visible.shape[0]),
            "injury_rows": int(injuries_visible.shape[0]),
        }

        features = pd.DataFrame(
            {
                "season": [season],
                "week": [week],
                "game_id": [game_id],
                "team": ["BUF"],
                "visible_plays": [payload["visible_plays"]],
                "injury_rows": [payload["injury_rows"]],
            }
        )

        payload_df = pd.DataFrame(
            {
                "season": [season],
                "week": [week],
                "game_id": [game_id],
                "team": ["BUF"],
                "snapshot_at": [snapshot_at],
                "asof_ts": [asof_ts],
                "feature_set": [feature_set],
                "payload_json": [json.dumps(payload)],
            }
        )

        return FeatureBuildResult(features_df=features, payload_df=payload_df)

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
        assert write_mode == "append"
        payload = captured["pbp_visible"].shape[0]
        predictions = pd.DataFrame(
            {
                "game_id": [game_id],
                "snapshot_at": [snapshot_at],
                "visible_plays": [payload],
                "injury_rows": [captured["injuries_visible"].shape[0]],
            }
        )
        return InferenceResult(
            predictions_df=predictions,
            model_path=model_path,
            model_id=model_id or "model",
            season=season,
            week=week,
        )

    runner = SnapshotRunner(
        config_path=config_path,
        stages=[
            SnapshotStage(
                name="T-60m",
                refresh_injuries=True,
                produce_predictions=True,
                feature_write_mode="append",
            )
        ],
        ingest_injuries_fn=fake_ingest_injuries,
        schedule_loader=fake_schedule_loader,
        pbp_loader=fake_pbp_loader,
        feature_builder=fake_feature_builder,
        inference_runner=fake_inference,
    )

    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"artifact")

    executions = runner.run(
        season=season,
        week=week,
        stage_times={"T-60m": cutoff},
        model_path=model_path,
        model_id="model",
    )

    assert len(executions) == 1
    execution = executions[0]

    assert execution.stage.name == "T-60m"
    assert execution.feature_result is not None
    assert execution.prediction_result is not None

    # Verify that post-cutoff records were removed from the filtered frames.
    assert (pbp_df["event_time"] > cutoff).any()
    assert not (captured["pbp_visible"]["event_time"] > cutoff).any()

    assert (injuries_df["event_time"] > cutoff).any()
    assert not (captured["injuries_visible"]["event_time"] > cutoff).any()

    # Deterministic payload reflecting only pre-cutoff data.
    features_df = execution.feature_result.features_df
    assert features_df[["visible_plays", "injury_rows"]].to_dict("list") == {
        "visible_plays": [2],
        "injury_rows": [1],
    }

    predictions_df = execution.prediction_result.predictions_df
    assert predictions_df[["visible_plays", "injury_rows"]].to_dict("list") == {
        "visible_plays": [2],
        "injury_rows": [1],
    }
