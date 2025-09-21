"""Snapshot runner orchestrating the T-24h → T-60m workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

from nfl_pred.config import Config, DEFAULT_CONFIG_PATH, load_config
from nfl_pred.features.build_features import FeatureBuildResult, build_and_store_features
from nfl_pred.ingest import ingest_injuries, ingest_rosters, ingest_teams
from nfl_pred.pipeline.predict import InferenceResult, run_inference_pipeline

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotStage:
    """Configuration describing a single snapshot stage."""

    name: str
    refresh_rosters: bool = False
    refresh_injuries: bool = False
    produce_predictions: bool = False
    feature_write_mode: str = "append"

    def __post_init__(self) -> None:
        if self.feature_write_mode not in {"create", "replace", "append"}:
            raise ValueError(
                "feature_write_mode must be one of {'create', 'replace', 'append'}."
            )


@dataclass(slots=True)
class StageExecution:
    """Record detailing the outcome of a snapshot stage run."""

    stage: SnapshotStage
    timestamp: pd.Timestamp
    feature_result: FeatureBuildResult | None
    prediction_result: InferenceResult | None


DEFAULT_SNAPSHOT_STAGES: list[SnapshotStage] = [
    SnapshotStage(
        name="T-24h",
        refresh_rosters=True,
        refresh_injuries=True,
        feature_write_mode="replace",
    ),
    SnapshotStage(name="T-100m", refresh_injuries=True),
    SnapshotStage(name="T-80-75m", refresh_injuries=True),
    SnapshotStage(
        name="T-60m",
        refresh_injuries=True,
        produce_predictions=True,
    ),
]

StageTimeInput = (
    pd.Timestamp
    | datetime
    | str
    | Sequence[pd.Timestamp | datetime | str]
    | Mapping[object, pd.Timestamp | datetime | str]
)


class SnapshotRunner:
    """Coordinate snapshot stages while respecting visibility cut-offs."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        stages: Sequence[SnapshotStage] | None = None,
        ingest_injuries_fn: Callable[[Sequence[int]], Path] | None = None,
        ingest_rosters_fn: Callable[[Sequence[int]], Path] | None = None,
        ingest_teams_fn: Callable[[], Path] | None = None,
        schedule_loader: Callable[[Path, Sequence[int]], pd.DataFrame] | None = None,
        pbp_loader: Callable[[Path, Sequence[int]], pd.DataFrame] | None = None,
        feature_builder: Callable[..., FeatureBuildResult] | None = None,
        inference_runner: Callable[..., InferenceResult] | None = None,
    ) -> None:
        self._config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        self._config: Config = load_config(self._config_path)
        self._stages: list[SnapshotStage] = list(stages) if stages is not None else list(
            DEFAULT_SNAPSHOT_STAGES
        )

        self._ingest_injuries = ingest_injuries_fn or ingest_injuries
        self._ingest_rosters = ingest_rosters_fn or ingest_rosters
        self._ingest_teams = ingest_teams_fn or ingest_teams

        self._schedule_loader = schedule_loader or _load_schedule_from_disk
        self._pbp_loader = pbp_loader or _load_pbp_from_disk

        self._feature_builder = feature_builder
        self._inference_runner = inference_runner

    @property
    def config(self) -> Config:
        """Expose the resolved configuration for downstream consumers."""

        return self._config

    @property
    def stages(self) -> list[SnapshotStage]:
        """Return the configured stage definitions."""

        return list(self._stages)

    def run(
        self,
        *,
        season: int,
        week: int,
        stage_times: Mapping[str, StageTimeInput],
        feature_set: str = "mvp_v1",
        model_path: str | Path | None = None,
        model_id: str | None = None,
    ) -> list[StageExecution]:
        """Execute the configured snapshot stages in order."""

        prepared_times = self._prepare_stage_times(stage_times)
        season_list = [int(season)]

        data_dir = Path(self._config.paths.data_dir)
        schedule_df = self._schedule_loader(data_dir, season_list)
        pbp_df = self._pbp_loader(data_dir, season_list)

        model_path_obj = Path(model_path) if model_path is not None else None
        duckdb_path = Path(self._config.paths.duckdb_path)

        executions: list[StageExecution] = []

        for stage, timestamps in prepared_times:
            if stage.produce_predictions and model_path_obj is None:
                raise ValueError(
                    "model_path must be provided for stages that generate predictions."
                )

            for timestamp in timestamps:
                LOGGER.info("Running snapshot stage %s at %s", stage.name, timestamp.isoformat())

                if stage.refresh_rosters:
                    self._ingest_rosters(season_list)
                    self._ingest_teams()
                if stage.refresh_injuries:
                    self._ingest_injuries(season_list)

                feature_result = self._run_feature_builder(
                    pbp_df,
                    schedule_df,
                    asof_ts=timestamp,
                    snapshot_at=timestamp,
                    feature_set=feature_set,
                    write_mode=stage.feature_write_mode,
                    duckdb_path=str(duckdb_path),
                )

                prediction_result: InferenceResult | None = None
                if stage.produce_predictions:
                    prediction_result = self._run_inference(
                        model_path=model_path_obj,
                        season=season,
                        week=week,
                        model_id=model_id,
                        feature_set=feature_set,
                        feature_snapshot_at=timestamp,
                        snapshot_at=timestamp,
                        config_path=self._config_path,
                        duckdb_path=duckdb_path,
                        write_mode="append",
                    )

                executions.append(
                    StageExecution(
                        stage=stage,
                        timestamp=timestamp,
                        feature_result=feature_result,
                        prediction_result=prediction_result,
                    )
                )

        return executions

    def _prepare_stage_times(
        self, stage_times: Mapping[str, StageTimeInput]
    ) -> list[tuple[SnapshotStage, list[pd.Timestamp]]]:
        schedule: list[tuple[SnapshotStage, list[pd.Timestamp]]] = []
        for stage in self._stages:
            if stage.name not in stage_times:
                raise ValueError(f"Missing timestamp configuration for snapshot stage '{stage.name}'.")
            raw = stage_times[stage.name]
            timestamps = _coerce_timestamp_list(raw)
            if not timestamps:
                raise ValueError(f"Snapshot stage '{stage.name}' requires at least one timestamp.")
            schedule.append((stage, timestamps))
        return schedule

    def _run_feature_builder(
        self,
        pbp_df: pd.DataFrame,
        schedule_df: pd.DataFrame,
        *,
        asof_ts: pd.Timestamp,
        snapshot_at: pd.Timestamp,
        feature_set: str,
        write_mode: str,
        duckdb_path: str,
    ) -> FeatureBuildResult:
        builder = self._feature_builder or build_and_store_features
        return builder(
            pbp_df,
            schedule_df,
            asof_ts=asof_ts,
            snapshot_at=snapshot_at,
            feature_set=feature_set,
            write_mode=write_mode,
            duckdb_path=duckdb_path,
        )

    def _run_inference(
        self,
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
        runner = self._inference_runner or run_inference_pipeline
        return runner(
            model_path=model_path,
            season=season,
            week=week,
            model_id=model_id,
            feature_set=feature_set,
            feature_snapshot_at=feature_snapshot_at,
            snapshot_at=snapshot_at,
            config_path=config_path,
            duckdb_path=duckdb_path,
            write_mode=write_mode,
        )


def run_snapshot_workflow(
    *,
    season: int,
    week: int,
    stage_times: Mapping[str, StageTimeInput],
    feature_set: str = "mvp_v1",
    model_path: str | Path | None = None,
    model_id: str | None = None,
    config_path: str | Path | None = None,
    stages: Sequence[SnapshotStage] | None = None,
) -> list[StageExecution]:
    """Convenience wrapper suitable for CLI invocation."""

    runner = SnapshotRunner(config_path=config_path, stages=stages)
    return runner.run(
        season=season,
        week=week,
        stage_times=stage_times,
        feature_set=feature_set,
        model_path=model_path,
        model_id=model_id,
    )


def _coerce_timestamp_list(value: StageTimeInput) -> list[pd.Timestamp]:
    if isinstance(value, (pd.Timestamp, datetime, str)):
        return [_ensure_utc_timestamp(value)]

    iterable: Iterable[object]
    if isinstance(value, Mapping):
        iterable = value.values()
    elif isinstance(value, Sequence):
        iterable = value
    else:
        raise TypeError(f"Unsupported timestamp specification: {value!r}")

    timestamps: list[pd.Timestamp] = []
    for item in iterable:
        if isinstance(item, (pd.Timestamp, datetime, str)):
            timestamps.append(_ensure_utc_timestamp(item))
        else:
            raise TypeError(f"Unsupported timestamp specification entry: {item!r}")
    return timestamps


def _ensure_utc_timestamp(value: pd.Timestamp | datetime | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _load_schedule_from_disk(data_dir: Path, seasons: Sequence[int]) -> pd.DataFrame:
    schedule_path = data_dir / "raw" / "schedules.parquet"
    if not schedule_path.exists():
        raise FileNotFoundError(
            f"Schedule Parquet not found at {schedule_path}. Run ingestion before snapshots."
        )
    schedule_df = pd.read_parquet(schedule_path)
    if seasons:
        schedule_df = schedule_df.loc[schedule_df["season"].isin(list(seasons))].copy()
    return schedule_df


def _load_pbp_from_disk(data_dir: Path, seasons: Sequence[int]) -> pd.DataFrame:
    raw_dir = data_dir / "raw"
    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = raw_dir / f"pbp_{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Play-by-play Parquet for season {season} not found at {path}."
            )
        frames.append(pd.read_parquet(path))

    if not frames:
        raise RuntimeError("No play-by-play frames loaded for the requested seasons.")

    return pd.concat(frames, ignore_index=True)


__all__ = [
    "DEFAULT_SNAPSHOT_STAGES",
    "SnapshotRunner",
    "SnapshotStage",
    "StageExecution",
    "run_snapshot_workflow",
]
