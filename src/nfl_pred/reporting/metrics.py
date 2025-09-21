"""Evaluation metrics and reliability reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class MetricsResult:
    """Container for aggregated classification metrics."""

    brier_score: float
    log_loss: float
    n_observations: int


@dataclass(frozen=True)
class ReliabilityBin:
    """Summary statistics for a single reliability bin."""

    lower: float
    upper: float
    midpoint: float
    count: int
    predicted_mean: float
    observed_rate: float


_DEFAULT_BINS = np.linspace(0.0, 1.0, 11)
_EPSILON = 1e-6


def compute_classification_metrics(
    df: pd.DataFrame,
    *,
    probability_column: str,
    label_column: str,
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Brier score and log-loss overall or grouped by columns."""

    if probability_column not in df.columns:
        raise KeyError(f"Probability column '{probability_column}' missing from DataFrame.")
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' missing from DataFrame.")

    working = df[[probability_column, label_column]].copy()
    working[label_column] = working[label_column].astype(int)

    if group_columns:
        missing = [column for column in group_columns if column not in df.columns]
        if missing:
            raise KeyError(f"Group columns missing from DataFrame: {missing}")
        grouped = df[list(group_columns) + [probability_column, label_column]].copy()
        metrics = (
            grouped.groupby(list(group_columns), dropna=False)[[probability_column, label_column]]
            .apply(_compute_metrics_for_slice, probability_column, label_column)
            .reset_index()
        )
    else:
        metrics = _compute_metrics_for_slice(working, probability_column, label_column)
        metrics = metrics.to_frame().T

    return metrics


def _compute_metrics_for_slice(
    frame: pd.DataFrame,
    probability_column: str,
    label_column: str,
) -> pd.Series:
    if frame.empty:
        raise ValueError("Cannot compute metrics on empty slice.")

    probs = _clip_probabilities(frame[probability_column].to_numpy(dtype=float))
    labels = frame[label_column].to_numpy(dtype=int)

    brier = float(brier_score_loss(labels, probs))
    logl = float(log_loss(labels, probs, labels=[0, 1]))

    return pd.Series(
        {
            "brier_score": brier,
            "log_loss": logl,
            "n_observations": int(len(frame)),
        }
    )


def compute_reliability_table(
    df: pd.DataFrame,
    *,
    probability_column: str,
    label_column: str,
    bins: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return reliability bin statistics using deterministic bin edges."""

    if probability_column not in df.columns:
        raise KeyError(f"Probability column '{probability_column}' missing from DataFrame.")
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' missing from DataFrame.")

    working = df[[probability_column, label_column]].copy()
    working[label_column] = working[label_column].astype(int)

    bin_edges = np.asarray(list(bins) if bins is not None else _DEFAULT_BINS, dtype=float)
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("Bins must define at least two monotonically increasing edges.")
    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError("Bin edges must be strictly increasing.")

    clipped_probs = _clip_probabilities(working[probability_column].to_numpy(dtype=float))
    indices = np.digitize(clipped_probs, bin_edges, right=True)

    records: list[dict[str, float | int]] = []
    for bin_idx in range(1, bin_edges.size):
        mask = indices == bin_idx
        if not np.any(mask):
            continue

        lower = float(bin_edges[bin_idx - 1])
        upper = float(bin_edges[bin_idx])
        midpoint = float((lower + upper) / 2)
        slice_probs = clipped_probs[mask]
        slice_labels = working[label_column].to_numpy(dtype=int)[mask]

        records.append(
            {
                "lower": lower,
                "upper": upper,
                "midpoint": midpoint,
                "count": int(mask.sum()),
                "predicted_mean": float(slice_probs.mean()),
                "observed_rate": float(slice_labels.mean()),
            }
        )

    reliability_df = pd.DataFrame.from_records(records, columns=[
        "lower",
        "upper",
        "midpoint",
        "count",
        "predicted_mean",
        "observed_rate",
    ])

    return reliability_df


def plot_reliability_curve(reliability: pd.DataFrame, *, path: Path) -> Path:
    """Create a reliability curve plot and persist it to ``path``."""

    required_columns = {"midpoint", "observed_rate"}
    if not required_columns.issubset(reliability.columns):
        raise KeyError(
            "Reliability DataFrame must contain columns: 'midpoint' and 'observed_rate'."
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")

    if not reliability.empty:
        ax.plot(
            reliability["midpoint"],
            reliability["observed_rate"],
            marker="o",
            label="Model",
        )

    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Empirical win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return path


def save_metrics_report(metrics: pd.DataFrame, *, reports_dir: Path, name: str = "metrics.csv") -> Path:
    """Persist metrics DataFrame to the reports directory and log to MLflow if active."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / name
    metrics.to_csv(output_path, index=False)

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.log_artifact(str(output_path), artifact_path="reports")

    return output_path


def save_reliability_report(
    reliability: pd.DataFrame,
    *,
    reports_dir: Path,
    name: str = "reliability.csv",
) -> Path:
    """Persist reliability statistics and log to MLflow if an active run exists."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / name
    reliability.to_csv(output_path, index=False)

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.log_artifact(str(output_path), artifact_path="reports")

    return output_path


def _clip_probabilities(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.clip(array, _EPSILON, 1 - _EPSILON)


__all__ = [
    "MetricsResult",
    "ReliabilityBin",
    "compute_classification_metrics",
    "compute_reliability_table",
    "plot_reliability_curve",
    "save_metrics_report",
    "save_reliability_report",
]
