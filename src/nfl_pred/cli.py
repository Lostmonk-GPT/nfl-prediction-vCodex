"""Command-line interface for NFL prediction workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import typer

from nfl_pred.config import DEFAULT_CONFIG_PATH, load_config
from nfl_pred.features.build_features import build_and_store_features
from nfl_pred.ingest import ingest_pbp, ingest_rosters, ingest_schedules, ingest_teams
from nfl_pred.logging_setup import setup_logging
from nfl_pred.pipeline import run_inference_pipeline, run_training_pipeline
from nfl_pred.reporting.metrics import (
    compute_classification_metrics,
    compute_reliability_table,
    save_metrics_report,
    save_reliability_report,
)
from nfl_pred.storage.duckdb_client import DuckDBClient
from nfl_pred.snapshot import DEFAULT_SNAPSHOT_STAGES, SnapshotStage, run_snapshot_workflow


app = typer.Typer(help="Operations CLI for ingestion, feature building, training, and reporting.")


def _validate_seasons(seasons: Iterable[int]) -> list[int]:
    values = [int(season) for season in seasons]
    if not values:
        raise typer.BadParameter("At least one season must be provided.")
    return values


def _parse_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise typer.BadParameter(f"Could not parse timestamp '{value}'.") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp


def _load_raw_schedule(config_path: Path, seasons: Iterable[int]) -> pd.DataFrame:
    config = load_config(config_path)
    schedule_path = Path(config.paths.data_dir) / "raw" / "schedules.parquet"
    if not schedule_path.exists():
        raise FileNotFoundError(
            "Schedule Parquet not found. Run 'ingest' before building features or reports."
        )
    schedule_df = pd.read_parquet(schedule_path)
    if seasons:
        schedule_df = schedule_df.loc[schedule_df["season"].isin(list(seasons))].copy()
    return schedule_df


def _load_raw_pbp(config_path: Path, seasons: Iterable[int]) -> pd.DataFrame:
    config = load_config(config_path)
    raw_dir = Path(config.paths.data_dir) / "raw"
    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = raw_dir / f"pbp_{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Play-by-play Parquet for season {season} not found. Run 'ingest' first."
            )
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _resolve_config_path(config: Optional[Path]) -> Path:
    return Path(config) if config is not None else DEFAULT_CONFIG_PATH


def _resolve_latest_model_artifact(models_dir: Path) -> Path:
    candidates = sorted(models_dir.glob("*.joblib"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            "No model artifacts found. Run training or provide --model-path explicitly."
        )
    return candidates[0]


def _build_stage_schedule(
    kickoff_ts: pd.Timestamp,
    *,
    final_only: bool = False,
) -> tuple[dict[str, list[pd.Timestamp]], list[SnapshotStage]]:
    """Construct the default snapshot schedule relative to kickoff."""

    kickoff_utc = kickoff_ts.tz_convert("UTC")

    schedule: dict[str, list[pd.Timestamp]] = {
        "T-24h": [kickoff_utc - pd.Timedelta(hours=24)],
        "T-100m": [kickoff_utc - pd.Timedelta(minutes=100)],
        "T-80-75m": [
            kickoff_utc - pd.Timedelta(minutes=80),
            kickoff_utc - pd.Timedelta(minutes=75),
        ],
        "T-60m": [kickoff_utc - pd.Timedelta(minutes=60)],
    }

    if final_only:
        stages = [stage for stage in DEFAULT_SNAPSHOT_STAGES if stage.name == "T-60m"]
        if not stages:
            raise RuntimeError("T-60m stage definition missing from DEFAULT_SNAPSHOT_STAGES.")
        return {"T-60m": schedule["T-60m"]}, stages

    return schedule, list(DEFAULT_SNAPSHOT_STAGES)


@app.command()
def ingest(
    seasons: List[int] = typer.Option(
        ..., "--seasons", help="Seasons to ingest, e.g. --seasons 2022 2023.", metavar="[SEASON]..."
    )
) -> None:
    """Ingest schedule, play-by-play, and roster data for the requested seasons."""

    setup_logging()
    season_list = _validate_seasons(seasons)

    schedules_path = ingest_schedules(season_list)
    pbp_paths = ingest_pbp(season_list)
    rosters_path = ingest_rosters(season_list)
    teams_path = ingest_teams()

    typer.echo(f"Schedules written to {schedules_path}")
    typer.echo(f"Play-by-play files written: {', '.join(str(path) for path in pbp_paths)}")
    typer.echo(f"Rosters written to {rosters_path}")
    typer.echo(f"Teams written to {teams_path}")


@app.command("build-features")
def build_features(
    seasons: List[int] = typer.Option(
        ..., "--seasons", help="Seasons to include when assembling features.", metavar="[SEASON]..."
    ),
    feature_set: str = typer.Option("mvp_v1", help="Feature set identifier stored with the payload."),
    asof_ts: Optional[str] = typer.Option(
        None, help="Optional visibility cutoff timestamp (UTC ISO format)."
    ),
    snapshot_at: Optional[str] = typer.Option(
        None, help="Override snapshot timestamp (UTC ISO format)."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to configuration YAML.", exists=True, dir_okay=False
    ),
    write_mode: str = typer.Option(
        "replace", help="DuckDB write mode for the features table (create, replace, append)."
    ),
) -> None:
    """Build the modeling feature matrix and persist it to DuckDB."""

    setup_logging()
    season_list = _validate_seasons(seasons)
    config_path = _resolve_config_path(config)

    schedule_df = _load_raw_schedule(config_path, season_list)
    pbp_df = _load_raw_pbp(config_path, season_list)

    asof_timestamp = _parse_timestamp(asof_ts)
    snapshot_timestamp = _parse_timestamp(snapshot_at)

    result = build_and_store_features(
        pbp_df,
        schedule_df,
        asof_ts=asof_timestamp,
        snapshot_at=snapshot_timestamp,
        feature_set=feature_set,
        write_mode=write_mode,
    )

    typer.echo(
        f"Persisted {len(result.payload_df)} feature rows for seasons: {', '.join(map(str, season_list))}."
    )


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to configuration YAML.", exists=True, dir_okay=False
    ),
    feature_set: str = typer.Option("mvp_v1", help="Feature set identifier to train against."),
    label_column: str = typer.Option("label_team_win", help="Label column within the feature payload."),
    min_train_weeks: int = typer.Option(4, help="Minimum training weeks before evaluating folds."),
    calibration_weeks: int = typer.Option(1, help="Weeks reserved for calibration."),
    n_splits: Optional[int] = typer.Option(None, help="Override number of CV splits."),
    random_state: int = typer.Option(42, help="Random seed for model reproducibility."),
) -> None:
    """Run the end-to-end training pipeline."""

    setup_logging()
    config_path = _resolve_config_path(config)

    result = run_training_pipeline(
        config_path=config_path,
        feature_set=feature_set,
        label_column=label_column,
        min_train_weeks=min_train_weeks,
        calibration_weeks=calibration_weeks,
        n_splits=n_splits,
        random_state=random_state,
    )

    typer.echo(f"Training run completed with model artifact at {result.model_path}")
    typer.echo(f"MLflow run ID: {result.run_id}")


@app.command()
def predict(
    season: int = typer.Option(..., help="Season to score."),
    week: int = typer.Option(..., help="Week number to score."),
    model_path: Optional[Path] = typer.Option(
        None, help="Path to trained model artifact (.joblib).", file_okay=True, dir_okay=False
    ),
    model_id: Optional[str] = typer.Option(None, help="Override model identifier stored with predictions."),
    feature_set: Optional[str] = typer.Option(None, help="Feature set to query for inference."),
    feature_snapshot_at: Optional[str] = typer.Option(
        None, help="Optional snapshot cutoff when selecting features."
    ),
    snapshot_at: Optional[str] = typer.Option(
        None, help="Override prediction snapshot timestamp (UTC ISO format)."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to configuration YAML.", exists=True, dir_okay=False
    ),
    duckdb_path: Optional[Path] = typer.Option(
        None, help="Override DuckDB database path used for reading/writing predictions."
    ),
    write_mode: str = typer.Option("append", help="Write mode for the predictions table."),
) -> None:
    """Generate predictions for a given season/week."""

    setup_logging()
    config_path = _resolve_config_path(config)
    config_obj = load_config(config_path)

    models_dir = Path(config_obj.paths.data_dir) / "models"
    if model_path is None:
        model_artifact = _resolve_latest_model_artifact(models_dir)
    else:
        model_artifact = model_path

    inference = run_inference_pipeline(
        model_path=model_artifact,
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

    typer.echo(
        f"Generated {len(inference.predictions_df)} predictions for season {season} week {week}."
    )


@app.command()
def snapshot(
    season: int = typer.Option(..., help="Season to snapshot."),
    week: int = typer.Option(..., help="Week to snapshot."),
    kickoff_at: str = typer.Option(..., "--at", help="Scheduled kickoff timestamp (ISO 8601)."),
    feature_set: str = typer.Option("mvp_v1", help="Feature set identifier used for features and predictions."),
    model_path: Optional[Path] = typer.Option(
        None, help="Path to trained model artifact (.joblib) for prediction stage.", file_okay=True, dir_okay=False
    ),
    model_id: Optional[str] = typer.Option(None, help="Override model identifier stored with predictions."),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to configuration YAML.", exists=True, dir_okay=False
    ),
    final_only: bool = typer.Option(
        False,
        "--final-only/--full-timeline",
        help="Run only the T-60m freeze instead of the full T-24h→T-60m sequence.",
    ),
) -> None:
    """Run the snapshot workflow for a given season and week."""

    setup_logging()
    config_path = _resolve_config_path(config)
    kickoff_ts = _parse_timestamp(kickoff_at)
    if kickoff_ts is None:
        raise typer.BadParameter("A kickoff timestamp must be provided via --at <iso8601>.")

    config_obj = load_config(config_path)
    models_dir = Path(config_obj.paths.data_dir) / "models"

    stage_schedule, stages = _build_stage_schedule(kickoff_ts, final_only=final_only)
    requires_model = any(stage.produce_predictions for stage in stages)

    resolved_model_path: Optional[Path] = None
    if requires_model:
        if model_path is None:
            try:
                resolved_model_path = _resolve_latest_model_artifact(models_dir)
            except FileNotFoundError as exc:
                raise typer.BadParameter(
                    "No model artifact found. Provide --model-path or train a model before running snapshots."
                ) from exc
        else:
            resolved_model_path = model_path

        if not resolved_model_path.exists():
            raise typer.BadParameter(f"Model artifact '{resolved_model_path}' does not exist.")

    executions = run_snapshot_workflow(
        season=season,
        week=week,
        stage_times=stage_schedule,
        feature_set=feature_set,
        model_path=resolved_model_path,
        model_id=model_id,
        config_path=config_path,
        stages=stages,
    )

    typer.echo(
        f"Executed {len(executions)} snapshot stage runs for season {season} week {week} with kickoff {kickoff_ts.isoformat()}"
    )

    prediction_execs = [execution for execution in executions if execution.prediction_result is not None]
    if prediction_execs:
        final_exec = prediction_execs[-1]
        prediction_count = len(final_exec.prediction_result.predictions_df)
        typer.echo(
            f"Generated {prediction_count} predictions at snapshot {final_exec.timestamp.isoformat()} using model {final_exec.prediction_result.model_id}."
        )


@app.command()
def report(
    season: int = typer.Option(..., help="Season to evaluate."),
    week: int = typer.Option(..., help="Week to evaluate."),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to configuration YAML.", exists=True, dir_okay=False
    ),
) -> None:
    """Compute evaluation metrics for predictions and persist reporting artifacts."""

    setup_logging()
    config_path = _resolve_config_path(config)
    config_obj = load_config(config_path)

    db_path = Path(config_obj.paths.duckdb_path)
    reports_dir = Path(config_obj.paths.data_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with DuckDBClient(str(db_path)) as client:
        client.apply_schema()
        predictions = client.read_sql(
            """
            SELECT game_id, season, week, asof_ts, p_home_win, p_away_win, model_id, snapshot_at
            FROM predictions
            WHERE season = ? AND week = ?
            """,
            [season, week],
        )

        if predictions.empty:
            typer.echo("No predictions found for the requested season/week.")
            raise typer.Exit(code=1)

        schedule = _load_raw_schedule(config_path, [season])
        schedule_week = schedule.loc[schedule["week"] == week].copy()
        if schedule_week.empty:
            typer.echo("No schedule rows available for the requested season/week.")
            raise typer.Exit(code=1)

        merged = predictions.merge(schedule_week, on=["season", "week", "game_id"], how="inner")
        if merged.empty:
            typer.echo("Unable to join predictions with schedule results for evaluation.")
            raise typer.Exit(code=1)

        merged["home_score"] = pd.to_numeric(merged.get("home_score"), errors="coerce")
        merged["away_score"] = pd.to_numeric(merged.get("away_score"), errors="coerce")

        score_mask = merged[["home_score", "away_score"]].notna().all(axis=1)
        merged = merged.loc[score_mask].copy()
        if merged.empty:
            typer.echo("Schedule rows are missing final scores; cannot compute metrics.")
            raise typer.Exit(code=1)

        merged["label_home_win"] = (merged["home_score"] > merged["away_score"]).astype(int)

        metrics_input = merged[["p_home_win", "label_home_win"]].copy()
        metrics = compute_classification_metrics(
            metrics_input,
            probability_column="p_home_win",
            label_column="label_home_win",
        )

        reliability = compute_reliability_table(
            metrics_input,
            probability_column="p_home_win",
            label_column="label_home_win",
        )

        timestamp = datetime.now(timezone.utc)
        asof_ts = pd.to_datetime(predictions["asof_ts"], utc=True, errors="coerce").max()

        metrics_name = f"metrics_s{season}_w{week}.csv"
        reliability_name = f"reliability_s{season}_w{week}.csv"
        save_metrics_report(metrics, reports_dir=reports_dir, name=metrics_name)
        save_reliability_report(reliability, reports_dir=reports_dir, name=reliability_name)

        report_records = []
        for column in ("brier_score", "log_loss"):
            if column in metrics.columns:
                report_records.append(
                    {
                        "season": season,
                        "week": week,
                        "asof_ts": asof_ts,
                        "metric": column,
                        "value": float(metrics.iloc[0][column]),
                        "snapshot_at": timestamp,
                    }
                )

        if report_records:
            report_df = pd.DataFrame.from_records(report_records)
            client.write_df(report_df, table="reports", mode="append")

    typer.echo(
        f"Saved metrics and reliability reports for season {season} week {week} to {reports_dir}."
    )


def main() -> None:
    """Entry point for console scripts."""

    app()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


__all__ = ["app", "main"]

