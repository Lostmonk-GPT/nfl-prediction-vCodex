"""Expanded evaluation reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlflow
import pandas as pd
from matplotlib import pyplot as plt

from .metrics import compute_classification_metrics


@dataclass(frozen=True)
class ExpandedMetricConfig:
    """Configuration for expanded report aggregation."""

    rolling_window: int = 4
    favorite_threshold: float = 0.55
    underdog_threshold: float = 0.45
    season_column: str = "season"
    week_column: str = "week"
    slice_column: str = "slice"
    window_column: str = "window"


_DEFAULT_METRIC_COLUMNS: tuple[str, ...] = ("brier_score", "log_loss")


def build_expanded_metrics(
    df: pd.DataFrame,
    *,
    probability_column: str,
    label_column: str,
    config: ExpandedMetricConfig | None = None,
) -> pd.DataFrame:
    """Compute expanded evaluation metrics across windows and slices.

    Parameters
    ----------
    df:
        Input DataFrame containing at minimum season, week, prediction probability,
        and binary label columns.
    probability_column:
        Name of the column containing predicted probabilities.
    label_column:
        Name of the column containing binary outcome labels (0/1).
    config:
        Optional configuration to override defaults such as rolling window size
        and slice thresholds.

    Returns
    -------
    pandas.DataFrame
        DataFrame including columns ``season``, ``week``, ``window``, ``slice`` and
        metric outputs (Brier score, log-loss, n_observations) for each slice.
    """

    cfg = config or ExpandedMetricConfig()

    required_columns = {cfg.season_column, cfg.week_column, probability_column, label_column}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for expanded metrics: {sorted(missing)}")

    if not 0.0 < cfg.underdog_threshold < cfg.favorite_threshold < 1.0:
        raise ValueError(
            "Favorite/underdog thresholds must satisfy 0 < underdog < favorite < 1."
        )
    if cfg.rolling_window < 1:
        raise ValueError("Rolling window must be at least 1 week.")

    working = df[[cfg.season_column, cfg.week_column, probability_column, label_column]].copy()
    working[label_column] = working[label_column].astype(int)
    working.sort_values([cfg.season_column, cfg.week_column], inplace=True)

    records: list[pd.DataFrame] = []

    weekly = compute_classification_metrics(
        working,
        probability_column=probability_column,
        label_column=label_column,
        group_columns=[cfg.season_column, cfg.week_column],
    )
    weekly[cfg.window_column] = "weekly"
    weekly[cfg.slice_column] = "overall"
    records.append(weekly)

    season_to_date_records: list[dict[str, float | int]] = []
    for season_value, season_df in working.groupby(cfg.season_column, sort=False):
        for week_value in season_df[cfg.week_column].drop_duplicates().sort_values():
            window_df = season_df.loc[season_df[cfg.week_column] <= week_value]
            metrics = compute_classification_metrics(
                window_df,
                probability_column=probability_column,
                label_column=label_column,
            )
            entry = metrics.iloc[0].to_dict()
            entry[cfg.season_column] = season_value
            entry[cfg.week_column] = int(week_value)
            entry[cfg.window_column] = "season_to_date"
            entry[cfg.slice_column] = "overall"
            season_to_date_records.append(entry)
    if season_to_date_records:
        records.append(pd.DataFrame.from_records(season_to_date_records))

    rolling_records: list[dict[str, float | int]] = []
    half_open = cfg.rolling_window - 1
    for season_value, season_df in working.groupby(cfg.season_column, sort=False):
        weeks = season_df[cfg.week_column].drop_duplicates().sort_values()
        for week_value in weeks:
            lower_bound = week_value - half_open
            window_df = season_df.loc[
                (season_df[cfg.week_column] >= lower_bound)
                & (season_df[cfg.week_column] <= week_value)
            ]
            metrics = compute_classification_metrics(
                window_df,
                probability_column=probability_column,
                label_column=label_column,
            )
            entry = metrics.iloc[0].to_dict()
            entry[cfg.season_column] = season_value
            entry[cfg.week_column] = int(week_value)
            entry[cfg.window_column] = f"rolling_{cfg.rolling_window}"
            entry[cfg.slice_column] = "overall"
            rolling_records.append(entry)
    if rolling_records:
        records.append(pd.DataFrame.from_records(rolling_records))

    slice_values = _assign_favorite_slice(
        working[probability_column],
        favorite_threshold=cfg.favorite_threshold,
        underdog_threshold=cfg.underdog_threshold,
    )
    working[cfg.slice_column] = slice_values
    slice_metrics = compute_classification_metrics(
        working,
        probability_column=probability_column,
        label_column=label_column,
        group_columns=[cfg.season_column, cfg.week_column, cfg.slice_column],
    )
    slice_metrics[cfg.window_column] = "weekly"
    records.append(slice_metrics)

    expanded = (
        pd.concat(records, ignore_index=True, sort=False)
        if records
        else pd.DataFrame(columns=[cfg.season_column, cfg.week_column])
    )

    # Ensure consistent column order and presence
    for column in (cfg.slice_column, cfg.window_column):
        if column not in expanded.columns:
            expanded[column] = "overall" if column == cfg.slice_column else "weekly"
    expanded = expanded[
        [
            cfg.season_column,
            cfg.week_column,
            cfg.window_column,
            cfg.slice_column,
            "brier_score",
            "log_loss",
            "n_observations",
        ]
    ].copy()

    expanded[cfg.season_column] = expanded[cfg.season_column].astype(int)
    expanded[cfg.week_column] = expanded[cfg.week_column].astype(int)
    expanded[cfg.slice_column] = expanded[cfg.slice_column].astype(str)
    expanded[cfg.window_column] = expanded[cfg.window_column].astype(str)

    return expanded.sort_values([cfg.season_column, cfg.week_column, cfg.window_column, cfg.slice_column]).reset_index(drop=True)


def prepare_report_records(
    expanded_metrics: pd.DataFrame,
    *,
    asof_ts: pd.Timestamp,
    snapshot_at: pd.Timestamp,
    metric_columns: Sequence[str] = _DEFAULT_METRIC_COLUMNS,
    config: ExpandedMetricConfig | None = None,
) -> pd.DataFrame:
    """Convert expanded metrics into report rows for DuckDB persistence."""

    cfg = config or ExpandedMetricConfig()
    required_columns = {
        cfg.season_column,
        cfg.week_column,
        cfg.window_column,
        cfg.slice_column,
    }
    missing = required_columns.difference(expanded_metrics.columns)
    if missing:
        raise KeyError(f"Expanded metrics missing required columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    for _, row in expanded_metrics.iterrows():
        season = int(row[cfg.season_column])
        week = int(row[cfg.week_column])
        window = str(row[cfg.window_column])
        slice_name = _normalize_token(str(row[cfg.slice_column]))
        for metric_name in metric_columns:
            if metric_name not in expanded_metrics.columns:
                continue
            value = row.get(metric_name)
            if pd.isna(value):
                continue
            metric_key = ".".join(filter(None, (window, slice_name, metric_name)))
            records.append(
                {
                    "season": season,
                    "week": week,
                    "asof_ts": asof_ts,
                    "metric": metric_key,
                    "value": float(value),
                    "snapshot_at": snapshot_at,
                }
            )

    return pd.DataFrame.from_records(records, columns=["season", "week", "asof_ts", "metric", "value", "snapshot_at"])


def save_expanded_metrics(
    expanded_metrics: pd.DataFrame,
    *,
    reports_dir: Path,
    name: str,
) -> Path:
    """Persist expanded metrics to disk and log to MLflow when active."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / name
    expanded_metrics.to_csv(output_path, index=False)

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.log_artifact(str(output_path), artifact_path="reports")

    return output_path


def plot_expanded_metric(
    expanded_metrics: pd.DataFrame,
    *,
    metric: str,
    window: str,
    season: int,
    reports_dir: Path,
    name: str,
    config: ExpandedMetricConfig | None = None,
) -> Path:
    """Plot metric trends across slices for the requested window."""

    cfg = config or ExpandedMetricConfig()
    if metric not in expanded_metrics.columns:
        raise KeyError(f"Metric '{metric}' not present in expanded metrics.")

    subset = expanded_metrics[
        (expanded_metrics[cfg.season_column] == season)
        & (expanded_metrics[cfg.window_column] == window)
    ]
    if subset.empty:
        raise ValueError(
            f"No rows available for season {season} and window '{window}' to plot."
        )

    pivot = (
        subset.pivot_table(
            index=cfg.week_column,
            columns=cfg.slice_column,
            values=metric,
            aggfunc="mean",
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    for slice_name in pivot.columns:
        ax.plot(pivot.index, pivot[slice_name], marker="o", label=str(slice_name))

    ax.set_title(f"{metric.replace('_', ' ').title()} — {window.replace('_', ' ').title()}")
    ax.set_xlabel("Week")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    ax.legend(title="Slice")

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / name
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.log_artifact(str(output_path), artifact_path="reports")

    return output_path


def _assign_favorite_slice(
    probabilities: pd.Series,
    *,
    favorite_threshold: float,
    underdog_threshold: float,
) -> pd.Series:
    labels = []
    for value in probabilities.astype(float):
        if value >= favorite_threshold:
            labels.append("favorite")
        elif value <= underdog_threshold:
            labels.append("underdog")
        else:
            labels.append("toss_up")
    return pd.Series(labels, index=probabilities.index, dtype="string")


def _normalize_token(token: str) -> str:
    return token.strip().replace(" ", "_").lower()


__all__ = [
    "ExpandedMetricConfig",
    "build_expanded_metrics",
    "prepare_report_records",
    "save_expanded_metrics",
    "plot_expanded_metric",
]
