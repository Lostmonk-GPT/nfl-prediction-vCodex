"""Command-line interface for NFL prediction workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import json

import pandas as pd
import typer

from nfl_pred.config import DEFAULT_CONFIG_PATH, load_config
from nfl_pred.features.build_features import build_and_store_features
from nfl_pred.ingest import ingest_pbp, ingest_rosters, ingest_schedules, ingest_teams
from nfl_pred.logging_setup import setup_logging
from nfl_pred.pipeline import run_inference_pipeline, run_training_pipeline
from nfl_pred.registry.hygiene import RetentionPolicy, enforce_retention_policy
from nfl_pred.reporting.expanded import (
    ExpandedMetricConfig,
    build_expanded_metrics,
    plot_expanded_metric,
    prepare_report_records,
    save_expanded_metrics,
)
from nfl_pred.reporting.metrics import (
    compute_classification_metrics,
    compute_reliability_table,
    plot_reliability_curve,
    save_metrics_report,
    save_reliability_report,
)
from nfl_pred.reporting.monitoring_report import (
    MonitoringComputation,
    build_monitoring_summary,
    compute_monitoring_psi_from_features,
    load_feature_payloads,
    plot_psi_barchart,
)
from nfl_pred.storage.duckdb_client import DuckDBClient
from nfl_pred.snapshot import DEFAULT_SNAPSHOT_STAGES, SnapshotStage, run_snapshot_workflow
from nfl_pred.monitoring.triggers import RetrainTriggerConfig


app = typer.Typer(help="Operations CLI for ingestion, feature building, training, and reporting.")


def _validate_seasons(seasons: Iterable[int]) -> list[int]:
    values = [int(season) for season in seasons]
    if not values:
        raise typer.BadParameter("At least one season must be provided.")
    return values


def _merge_season_inputs(
    option_values: Optional[Iterable[int]], extra_values: Iterable[int]
) -> list[int]:
    """Combine season inputs from option flags and free arguments."""

    combined: list[int] = []
    if option_values:
        combined.extend(int(season) for season in option_values)
    combined.extend(int(season) for season in extra_values)
    return _validate_seasons(combined)


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
    seasons: Optional[List[int]] = typer.Option(
        None,
        "--seasons",
        help="Seasons to ingest, e.g. --seasons 2022 2023.",
        metavar="[SEASON]...",
    ),
    extra_seasons: Sequence[int] = typer.Argument(
        (), metavar="[SEASON]...", help="Additional seasons provided without --seasons."
    ),
) -> None:
    """Ingest schedule, play-by-play, and roster data for the requested seasons."""

    setup_logging()
    season_list = _merge_season_inputs(seasons, extra_seasons)

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
    seasons: Optional[List[int]] = typer.Option(
        None,
        "--seasons",
        help="Seasons to include when assembling features.",
        metavar="[SEASON]...",
    ),
    extra_seasons: Sequence[int] = typer.Argument(
        (),
        metavar="[SEASON]...",
        help="Additional seasons provided without --seasons.",
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
    season_list = _merge_season_inputs(seasons, extra_seasons)
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
            WHERE season = ? AND week <= ?
            """,
            [season, week],
        )

        if predictions.empty:
            typer.echo("No predictions found for the requested season/week.")
            raise typer.Exit(code=1)

        schedule = _load_raw_schedule(config_path, [season])
        schedule_slice = schedule.loc[schedule["week"] <= week].copy()
        if schedule_slice.empty:
            typer.echo("No schedule rows available for the requested season/week.")
            raise typer.Exit(code=1)

        merged = predictions.merge(schedule_slice, on=["season", "week", "game_id"], how="inner")
        if merged.empty:
            typer.echo("Unable to join predictions with schedule results for evaluation.")
            raise typer.Exit(code=1)

        merged["home_score"] = pd.to_numeric(merged.get("home_score"), errors="coerce")
        merged["away_score"] = pd.to_numeric(merged.get("away_score"), errors="coerce")

        score_mask = merged[["home_score", "away_score"]].notna().all(axis=1)
        evaluation_df = merged.loc[score_mask].copy()
        if evaluation_df.empty:
            typer.echo("Schedule rows are missing final scores; cannot compute metrics.")
            raise typer.Exit(code=1)

        evaluation_df["label_home_win"] = (
            evaluation_df["home_score"] > evaluation_df["away_score"]
        ).astype(int)

        current_week_df = evaluation_df.loc[evaluation_df["week"] == week].copy()
        if current_week_df.empty:
            typer.echo("No completed games found for the requested week.")
            raise typer.Exit(code=1)

        metrics_input = current_week_df[["p_home_win", "label_home_win"]].copy()
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
        asof_series = pd.to_datetime(
            predictions.loc[predictions["week"] == week, "asof_ts"],
            utc=True,
            errors="coerce",
        )
        asof_ts = asof_series.max()
        if pd.isna(asof_ts):
            asof_ts = timestamp

        metrics_name = f"metrics_s{season}_w{week}.csv"
        reliability_name = f"reliability_s{season}_w{week}.csv"
        save_metrics_report(metrics, reports_dir=reports_dir, name=metrics_name)
        save_reliability_report(reliability, reports_dir=reports_dir, name=reliability_name)

        expanded_config = ExpandedMetricConfig()
        expanded_metrics = build_expanded_metrics(
            evaluation_df,
            probability_column="p_home_win",
            label_column="label_home_win",
            config=expanded_config,
        )
        expanded_name = f"expanded_metrics_s{season}_w{week}.csv"
        save_expanded_metrics(expanded_metrics, reports_dir=reports_dir, name=expanded_name)

        plot_targets = [
            ("weekly", "weekly_brier"),
            ("season_to_date", "season_to_date_brier"),
            (f"rolling_{expanded_config.rolling_window}", f"rolling{expanded_config.rolling_window}_brier"),
        ]
        for window_label, stem in plot_targets:
            try:
                plot_expanded_metric(
                    expanded_metrics,
                    metric="brier_score",
                    window=window_label,
                    season=season,
                    reports_dir=reports_dir,
                    name=f"{stem}_s{season}_w{week}.png",
                    config=expanded_config,
                )
            except ValueError:
                continue

        report_frames: list[pd.DataFrame] = []
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
            report_frames.append(pd.DataFrame.from_records(report_records))

        expanded_week = expanded_metrics.loc[
            expanded_metrics[expanded_config.week_column] == week
        ]
        expanded_records = prepare_report_records(
            expanded_week,
            asof_ts=asof_ts,
            snapshot_at=timestamp,
            config=expanded_config,
        )
        if not expanded_records.empty:
            report_frames.append(expanded_records)

        if report_frames:
            report_df = pd.concat(report_frames, ignore_index=True)
            client.write_df(report_df, table="reports", mode="append")

    typer.echo(
        f"Saved metrics and reliability reports for season {season} week {week} to {reports_dir}."
    )


@app.command()
def monitor(
    season: int = typer.Option(..., help="Season to evaluate for monitoring."),
    week: int = typer.Option(..., help="Target week to evaluate."),
    feature_set: str = typer.Option(
        "mvp_v1", help="Feature set identifier for PSI comparisons."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to configuration YAML.", exists=True, dir_okay=False
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Override directory for monitoring artifacts (defaults to data/reports/monitoring).",
        file_okay=False,
    ),
) -> None:
    """Generate monitoring summary, PSI diagnostics, and retrain trigger evaluation."""

    setup_logging()
    config_path = _resolve_config_path(config)
    config_obj = load_config(config_path)

    db_path = Path(config_obj.paths.duckdb_path)
    data_dir = Path(config_obj.paths.data_dir)

    base_output = output_dir or data_dir / "reports" / "monitoring"
    monitor_dir = base_output / f"season_{season}" / f"week_{week:02d}"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    with DuckDBClient(str(db_path)) as client:
        client.apply_schema()
        predictions = client.read_sql(
            """
            SELECT game_id, season, week, asof_ts, p_home_win, p_away_win, model_id, snapshot_at
            FROM predictions
            WHERE season = ? AND week <= ?
            """,
            [season, week],
        )

    if predictions.empty:
        typer.echo("No predictions found for the requested season/week.")
        raise typer.Exit(code=1)

    schedule_df = _load_raw_schedule(config_path, [season])
    schedule_slice = schedule_df.loc[schedule_df["week"] <= week].copy()
    if schedule_slice.empty:
        typer.echo("No schedule rows available for the requested season/week.")
        raise typer.Exit(code=1)

    merged = predictions.merge(schedule_slice, on=["season", "week", "game_id"], how="inner")
    if merged.empty:
        typer.echo("Unable to join predictions with schedule results for monitoring.")
        raise typer.Exit(code=1)

    merged["home_score"] = pd.to_numeric(merged.get("home_score"), errors="coerce")
    merged["away_score"] = pd.to_numeric(merged.get("away_score"), errors="coerce")
    score_mask = merged[["home_score", "away_score"]].notna().all(axis=1)
    evaluation_df = merged.loc[score_mask].copy()
    if evaluation_df.empty:
        typer.echo("Schedule rows are missing final scores; cannot compute monitoring outputs.")
        raise typer.Exit(code=1)

    evaluation_df["label_home_win"] = (
        evaluation_df["home_score"] > evaluation_df["away_score"]
    ).astype(int)

    current_week_df = evaluation_df.loc[evaluation_df["week"] == week].copy()
    if current_week_df.empty:
        typer.echo("No completed games found for the requested week.")
        raise typer.Exit(code=1)

    metrics_input = current_week_df[["p_home_win", "label_home_win"]].copy()
    weekly_metrics = compute_classification_metrics(
        metrics_input,
        probability_column="p_home_win",
        label_column="label_home_win",
    )

    reliability = compute_reliability_table(
        metrics_input,
        probability_column="p_home_win",
        label_column="label_home_win",
    )
    reliability_plot_path = plot_reliability_curve(
        reliability, path=monitor_dir / f"reliability_s{season}_w{week}.png"
    )

    expanded_config = ExpandedMetricConfig()
    expanded_metrics = build_expanded_metrics(
        evaluation_df,
        probability_column="p_home_win",
        label_column="label_home_win",
        config=expanded_config,
    )

    features_df = load_feature_payloads(db_path, feature_set=feature_set)
    trigger_config = RetrainTriggerConfig()
    psi_summary = compute_monitoring_psi_from_features(
        features_df,
        season=season,
        week=week,
        psi_threshold=trigger_config.psi_threshold,
    )

    asof_series = pd.to_datetime(
        predictions.loc[predictions["week"] == week, "asof_ts"], utc=True, errors="coerce"
    )
    asof_ts = asof_series.max()
    if pd.isna(asof_ts):
        asof_ts = None

    computation: MonitoringComputation = build_monitoring_summary(
        season=season,
        week=week,
        generated_at=datetime.now(timezone.utc),
        asof_ts=asof_ts,
        weekly_metrics=weekly_metrics.iloc[0],
        expanded_metrics=expanded_metrics,
        psi_summary=psi_summary,
        trigger_config=trigger_config,
        expanded_config=expanded_config,
    )

    summary_path = monitor_dir / f"monitoring_s{season}_w{week}.json"
    summary_path.write_text(json.dumps(computation.summary, indent=2), encoding="utf-8")

    metrics_path = monitor_dir / f"weekly_metrics_s{season}_w{week}.csv"
    weekly_metrics.to_csv(metrics_path, index=False)

    reliability_path = monitor_dir / f"reliability_s{season}_w{week}.csv"
    reliability.to_csv(reliability_path, index=False)

    expanded_path = monitor_dir / f"expanded_metrics_s{season}_w{week}.csv"
    expanded_metrics.to_csv(expanded_path, index=False)

    psi_summary_path = monitor_dir / f"psi_summary_s{season}_w{week}.csv"
    psi_summary.feature_psi.to_csv(psi_summary_path, index=False)

    breakdown = psi_summary.feature_psi.attrs.get("breakdown")
    if isinstance(breakdown, pd.DataFrame) and not breakdown.empty:
        breakdown_path = monitor_dir / f"psi_breakdown_s{season}_w{week}.csv"
        breakdown.to_csv(breakdown_path, index=False)

    psi_plot_path = plot_psi_barchart(
        psi_summary, path=monitor_dir / f"psi_top_features_s{season}_w{week}.png"
    )

    plot_targets = [
        ("weekly", "weekly_brier"),
        ("season_to_date", "season_to_date_brier"),
        (f"rolling_{expanded_config.rolling_window}", f"rolling{expanded_config.rolling_window}_brier"),
    ]
    for window_label, stem in plot_targets:
        try:
            plot_expanded_metric(
                expanded_metrics,
                metric="brier_score",
                window=window_label,
                season=season,
                reports_dir=monitor_dir,
                name=f"{stem}_s{season}_w{week}.png",
                config=expanded_config,
            )
        except ValueError:
            continue

    typer.echo(
        "Generated monitoring summary at {} with PSI plot {} and reliability plot {}.".format(
            summary_path,
            psi_plot_path,
            reliability_plot_path,
        )
    )


@app.command("mlflow-hygiene")
def mlflow_hygiene(
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to configuration YAML.", exists=True, dir_okay=False
    ),
    dry_run: Optional[bool] = typer.Option(
        None,
        "--dry-run/--no-dry-run",
        help="Override dry-run behaviour. When true, only log actions without deletion.",
    ),
    delete_artifacts: Optional[bool] = typer.Option(
        None,
        "--delete-artifacts/--keep-artifacts",
        help="Delete artifact directories for pruned runs when not in dry-run mode.",
    ),
) -> None:
    """Apply MLflow retention policies to prune old experiment runs."""

    setup_logging()
    config_path = _resolve_config_path(config)
    config_obj = load_config(config_path)

    hygiene_cfg = config_obj.mlflow.hygiene
    policy_cfg = hygiene_cfg.retention
    policy = RetentionPolicy(
        max_age_days=policy_cfg.max_age_days,
        keep_last_runs=policy_cfg.keep_last_runs,
        keep_top_runs=policy_cfg.keep_top_runs,
        metric=policy_cfg.metric,
        metric_goal=policy_cfg.metric_goal,
        protect_promoted=policy_cfg.protect_promoted,
        min_metric_value=policy_cfg.min_metric_value,
    )

    effective_dry_run = hygiene_cfg.dry_run if dry_run is None else dry_run
    effective_delete_artifacts = (
        hygiene_cfg.delete_artifacts if delete_artifacts is None else delete_artifacts
    )

    report = enforce_retention_policy(
        tracking_uri=config_obj.mlflow.tracking_uri,
        experiment=config_obj.mlflow.experiment,
        policy=policy,
        dry_run=effective_dry_run,
        delete_artifacts=effective_delete_artifacts,
    )

    action = "Would delete" if report.dry_run else "Deleted"
    if report.deleted_runs:
        typer.echo(
            f"{action} {len(report.deleted_runs)} runs: {', '.join(report.deleted_runs)}"
        )
    else:
        typer.echo("No runs eligible for deletion under current policy.")

    if report.protected_runs:
        typer.echo(
            "Protected runs retained: " + ", ".join(report.protected_runs)
        )

    typer.echo(
        "Retention scan complete — inspected "
        f"{report.scanned_runs} runs in experiment '{report.experiment}'."
    )

def main() -> None:
    """Entry point for console scripts."""

    app()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


__all__ = ["app", "main"]

