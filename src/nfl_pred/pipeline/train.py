"""Model training pipeline orchestrating cross-validation and calibration."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Sequence

import joblib
import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss

from nfl_pred.config import dump_config, load_config
from nfl_pred.logging_setup import setup_logging
from nfl_pred.model.baseline import BaselineClassifier
from nfl_pred.model.calibration import PlattCalibrator
from nfl_pred.model.splits import time_series_splits
from nfl_pred.storage.duckdb_client import DuckDBClient

LOGGER = logging.getLogger(__name__)
_PLAYOFF_MODES: Final[frozenset[str]] = frozenset({"include", "regular_only", "postseason_only"})


@dataclass(slots=True)
class FoldMetrics:
    """Evaluation metrics for a single cross-validation fold."""

    fold: int
    validation_season: int
    validation_week: int
    brier_score: float
    log_loss: float


@dataclass(slots=True)
class TrainingResult:
    """Summary of training outputs and persisted artifacts."""

    run_id: str
    model_path: Path
    reliability_plot_path: Path
    metrics: dict[str, float]
    fold_metrics: list[FoldMetrics]


def run_training_pipeline(
    *,
    config_path: str | Path | None = None,
    feature_set: str = "mvp_v1",
    label_column: str = "label_team_win",
    min_train_weeks: int = 4,
    calibration_weeks: int = 1,
    n_splits: int | None = None,
    random_state: int = 42,
) -> TrainingResult:
    """Execute the end-to-end model training workflow."""

    setup_logging()
    config = load_config(config_path)

    mlflow.set_tracking_uri(str(Path(config.mlflow.tracking_uri).expanduser().resolve()))

    features_df = _load_feature_rows(config.paths.duckdb_path, feature_set=feature_set)
    working = _prepare_training_table(features_df, label_column=label_column)

    playoff_mode = config.training.playoffs.mode
    before_rows = len(working)
    working = _apply_playoff_mode(working, mode=playoff_mode)
    after_rows = len(working)
    if after_rows != before_rows:
        LOGGER.info(
            "Applied playoff mode '%s': filtered %s of %s rows.",
            playoff_mode,
            before_rows - after_rows,
            before_rows,
        )

    if working.empty:
        raise ValueError(
            "No eligible training rows available after filtering. "
            f"Check playoff mode '{playoff_mode}' and label availability."
        )

    LOGGER.info("Loaded %s rows for training from feature set '%s'.", len(working), feature_set)

    split_column = "_week_index"
    working[split_column] = _compute_week_index(working["season"], working["week"])

    metadata_columns = {
        "season",
        "week",
        "game_id",
        "team_side",
        "asof_ts",
        "snapshot_at",
        split_column,
    }

    feature_columns = [
        column
        for column in working.columns
        if column not in metadata_columns and column != label_column
    ]
    if not feature_columns:
        raise ValueError("No feature columns available for model training.")

    LOGGER.debug("Training with feature columns: %s", feature_columns)

    train_frame, calibration_frame = _split_calibration_window(
        working,
        split_column=split_column,
        calibration_weeks=calibration_weeks,
    )

    if train_frame.empty:
        raise ValueError("Training data empty after reserving calibration window.")
    if calibration_frame.empty:
        raise ValueError("Calibration window did not contain any rows.")

    X_train = train_frame[feature_columns].reset_index(drop=True)
    y_train = train_frame[label_column].astype(int).reset_index(drop=True)

    X_calib = calibration_frame[feature_columns].reset_index(drop=True)
    y_calib = calibration_frame[label_column].astype(int).reset_index(drop=True)

    if np.unique(y_train).size < 2:
        raise ValueError("Training labels must contain both outcome classes.")
    if np.unique(y_calib).size < 2:
        raise ValueError("Calibration window must contain both outcome classes.")

    fold_metrics = _evaluate_cross_validation(
        train_frame.reset_index(drop=True),
        feature_columns=feature_columns,
        label_column=label_column,
        split_column=split_column,
        min_train_weeks=min_train_weeks,
        n_splits=n_splits,
        random_state=random_state,
    )

    baseline = BaselineClassifier(random_state=random_state)
    baseline.fit(X_train, y_train)

    calibrator = PlattCalibrator()
    calibrator.fit(baseline, X_calib, y_calib)

    holdout_probs = _clip_probabilities(calibrator.predict_proba(X_calib)[:, 1])
    holdout_metrics = {
        "holdout_brier": brier_score_loss(y_calib, holdout_probs),
        "holdout_log_loss": log_loss(y_calib, holdout_probs, labels=[0, 1]),
    }

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    models_dir = Path(config.paths.data_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": baseline,
        "calibrator": calibrator,
        "feature_columns": feature_columns,
        "label_column": label_column,
        "metadata": {
            "feature_set": feature_set,
            "created_at": timestamp,
            "min_train_weeks": min_train_weeks,
            "calibration_weeks": calibration_weeks,
        },
    }
    model_path = models_dir / f"baseline_platt_{timestamp}.joblib"
    joblib.dump(model_artifact, model_path)

    reliability_plot_path = models_dir / f"reliability_{timestamp}.png"
    _create_reliability_plot(y_calib, holdout_probs, reliability_plot_path)

    config_snapshot_path = models_dir / f"config_snapshot_{timestamp}.yaml"
    config_snapshot_path.write_text(dump_config(config), encoding="utf-8")

    if fold_metrics:
        cv_mean_brier = float(np.mean([metric.brier_score for metric in fold_metrics]))
        cv_mean_log_loss = float(np.mean([metric.log_loss for metric in fold_metrics]))
    else:
        cv_mean_brier = float("nan")
        cv_mean_log_loss = float("nan")

    aggregate_metrics = {
        "cv_mean_brier": cv_mean_brier,
        "cv_mean_log_loss": cv_mean_log_loss,
        **holdout_metrics,
    }

    with mlflow.start_run(run_name=f"baseline_logreg_{timestamp}") as active_run:
        mlflow.log_param("model_type", "baseline_logistic_regression")
        mlflow.log_param("calibrator", "platt")
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_calibration_rows", len(X_calib))
        mlflow.log_param("min_train_weeks", min_train_weeks)
        mlflow.log_param("calibration_weeks", calibration_weeks)
        mlflow.log_param("random_state", random_state)

        params = calibrator.calibration_params
        mlflow.log_param("calibration_slope", params.slope)
        mlflow.log_param("calibration_intercept", params.intercept)

        for metric in fold_metrics:
            tag = f"season{metric.validation_season}_week{metric.validation_week}"
            mlflow.log_metric(f"cv_brier_{tag}", metric.brier_score)
            mlflow.log_metric(f"cv_log_loss_{tag}", metric.log_loss)

        for name, value in aggregate_metrics.items():
            mlflow.log_metric(name, value)

        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(reliability_plot_path), artifact_path="plots")
        mlflow.log_artifact(str(config_snapshot_path), artifact_path="config")

        run_id = active_run.info.run_id

    LOGGER.info("Training completed; model artifact stored at %s", model_path)

    return TrainingResult(
        run_id=run_id,
        model_path=model_path,
        reliability_plot_path=reliability_plot_path,
        metrics=aggregate_metrics,
        fold_metrics=fold_metrics,
    )


def _load_feature_rows(duckdb_path: str | Path, *, feature_set: str) -> pd.DataFrame:
    query = """
        SELECT
            season,
            week,
            game_id,
            team_side,
            asof_ts,
            snapshot_at,
            payload_json
        FROM features
        WHERE feature_set = ?
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY season, week, game_id, team_side
            ORDER BY snapshot_at DESC
        ) = 1
        ORDER BY season, week, team_side
    """

    with DuckDBClient(str(duckdb_path)) as client:
        client.apply_schema()
        frame = client.read_sql(query, (feature_set,))

    if frame.empty:
        raise ValueError(f"Feature table returned no rows for feature_set='{feature_set}'.")

    payload_df = pd.DataFrame.from_records(frame["payload_json"].map(json.loads))
    payload_df = payload_df.fillna(value=np.nan)

    combined = pd.concat([frame.drop(columns=["payload_json"]).reset_index(drop=True), payload_df], axis=1)

    combined["team_side"] = combined["team_side"].astype(str)
    combined["season"] = combined["season"].astype(int)
    combined["week"] = combined["week"].astype(int)
    if "home_away" not in combined.columns:
        combined["home_away"] = combined["team_side"].astype(str)

    return combined


def _prepare_training_table(df: pd.DataFrame, *, label_column: str) -> pd.DataFrame:
    working = df.copy()
    if label_column not in working.columns:
        raise KeyError(f"Label column '{label_column}' not found in feature payloads.")

    working = working.loc[working[label_column].isin([0, 1])].copy()
    working[label_column] = working[label_column].astype(int)

    return working


def _split_calibration_window(
    df: pd.DataFrame,
    *,
    split_column: str,
    calibration_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if calibration_weeks < 1:
        raise ValueError("calibration_weeks must be at least 1.")

    unique_weeks = np.sort(df[split_column].unique())
    if unique_weeks.size <= calibration_weeks:
        raise ValueError("Not enough distinct weeks to reserve calibration window.")

    threshold = unique_weeks[-calibration_weeks:]
    calibration_mask = df[split_column].isin(threshold)

    calibration_frame = df.loc[calibration_mask].copy()
    train_frame = df.loc[~calibration_mask].copy()

    return train_frame, calibration_frame


def _apply_playoff_mode(
    df: pd.DataFrame,
    *,
    mode: str,
    column: str = "is_postseason",
) -> pd.DataFrame:
    normalized_mode = mode.lower()
    if normalized_mode not in _PLAYOFF_MODES:
        raise ValueError(
            f"Unsupported playoff mode '{mode}'. Expected one of: {sorted(_PLAYOFF_MODES)}."
        )

    if column not in df.columns:
        if normalized_mode != "include":
            LOGGER.warning(
                "Playoff mode '%s' requested but column '%s' missing; skipping filter.",
                normalized_mode,
                column,
            )
        return df.copy()

    mask = df[column].fillna(False).astype(bool)

    if normalized_mode == "include":
        return df.copy()
    if normalized_mode == "regular_only":
        return df.loc[~mask].copy()

    return df.loc[mask].copy()


def _evaluate_cross_validation(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    label_column: str,
    split_column: str,
    min_train_weeks: int,
    n_splits: int | None,
    random_state: int,
) -> list[FoldMetrics]:
    if min_train_weeks < 1:
        raise ValueError("min_train_weeks must be positive.")

    evaluation_frame = df.reset_index(drop=True)
    evaluation_frame[split_column] = evaluation_frame[split_column].astype(int)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        time_series_splits(
            evaluation_frame,
            group_col=split_column,
            min_train_weeks=min_train_weeks,
            n_splits=n_splits,
        ),
        start=1,
    ):
        X_train = evaluation_frame.loc[train_idx, feature_columns]
        y_train = evaluation_frame.loc[train_idx, label_column]
        X_val = evaluation_frame.loc[val_idx, feature_columns]
        y_val = evaluation_frame.loc[val_idx, label_column]

        model = BaselineClassifier(random_state=random_state + fold_idx)
        model.fit(X_train, y_train)

        probs = _clip_probabilities(model.predict_proba(X_val)[:, 1])
        fold_brier = brier_score_loss(y_val, probs)
        fold_log_loss = log_loss(y_val, probs, labels=[0, 1])

        val_slice = evaluation_frame.loc[val_idx, ["season", "week"]].iloc[0]
        folds.append(
            FoldMetrics(
                fold=fold_idx,
                validation_season=int(val_slice["season"]),
                validation_week=int(val_slice["week"]),
                brier_score=float(fold_brier),
                log_loss=float(fold_log_loss),
            )
        )

    return folds


def _compute_week_index(seasons: Iterable[int], weeks: Iterable[int]) -> np.ndarray:
    seasons_arr = np.asarray(list(seasons), dtype=int)
    weeks_arr = np.asarray(list(weeks), dtype=int)
    return seasons_arr * 100 + weeks_arr


def _clip_probabilities(values: Sequence[float], *, eps: float = 1e-6) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.clip(array, eps, 1 - eps)


def _create_reliability_plot(y_true: Sequence[int], probs: Sequence[float], path: Path) -> None:
    y_array = np.asarray(y_true)
    prob_array = np.asarray(probs)

    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(prob_array, bins, right=True)

    bin_centers: list[float] = []
    observed: list[float] = []
    for bin_id in range(1, len(bins)):
        mask = bin_indices == bin_id
        if not np.any(mask):
            continue
        center = float((bins[bin_id - 1] + bins[bin_id]) / 2)
        empirical = float(y_array[mask].mean())
        bin_centers.append(center)
        observed.append(empirical)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    if bin_centers:
        ax.plot(bin_centers, observed, marker="o", label="Calibrated")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Empirical win rate")
    ax.set_title("Reliability Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main(args: Sequence[str] | None = None) -> TrainingResult:
    parser = argparse.ArgumentParser(description="Train the baseline NFL prediction model.")
    parser.add_argument("--config", dest="config_path", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--feature-set", dest="feature_set", type=str, default="mvp_v1", help="Feature set identifier")
    parser.add_argument("--min-train-weeks", dest="min_train_weeks", type=int, default=4, help="Minimum weeks for training folds")
    parser.add_argument(
        "--calibration-weeks",
        dest="calibration_weeks",
        type=int,
        default=1,
        help="Number of most recent weeks reserved for calibration",
    )
    parser.add_argument("--n-splits", dest="n_splits", type=int, default=None, help="Number of CV splits (defaults to auto)")
    parser.add_argument("--random-state", dest="random_state", type=int, default=42, help="Random seed")

    parsed = parser.parse_args(args=args)

    return run_training_pipeline(
        config_path=parsed.config_path,
        feature_set=parsed.feature_set,
        min_train_weeks=parsed.min_train_weeks,
        calibration_weeks=parsed.calibration_weeks,
        n_splits=parsed.n_splits,
        random_state=parsed.random_state,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
