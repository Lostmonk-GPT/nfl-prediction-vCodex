"""Tests for reporting metrics and reliability helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_pred.reporting.metrics import (
    compute_classification_metrics,
    compute_reliability_table,
    plot_reliability_curve,
)


@pytest.fixture()
def sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 2, 2],
            "prob": [0.8, 0.2, 0.65, 0.3],
            "label": [1, 0, 1, 0],
        }
    )


def test_compute_classification_metrics_overall(sample_predictions: pd.DataFrame) -> None:
    metrics = compute_classification_metrics(
        sample_predictions, probability_column="prob", label_column="label"
    )

    assert pytest.approx(metrics.loc[0, "brier_score"], rel=1e-6) == 0.073125
    assert pytest.approx(metrics.loc[0, "log_loss"], rel=1e-6) == 0.308436
    assert metrics.loc[0, "n_observations"] == 4


def test_compute_classification_metrics_by_week(sample_predictions: pd.DataFrame) -> None:
    metrics = compute_classification_metrics(
        sample_predictions,
        probability_column="prob",
        label_column="label",
        group_columns=["season", "week"],
    )

    assert set(metrics.columns) == {"season", "week", "brier_score", "log_loss", "n_observations"}
    assert metrics.shape[0] == 2
    week1 = metrics.loc[metrics["week"] == 1].iloc[0]
    assert pytest.approx(week1["brier_score"], rel=1e-6) == 0.04
    assert week1["n_observations"] == 2


def test_compute_reliability_table_returns_expected_bins(sample_predictions: pd.DataFrame) -> None:
    reliability = compute_reliability_table(
        sample_predictions,
        probability_column="prob",
        label_column="label",
        bins=np.linspace(0.0, 1.0, 6),
    )

    assert set(reliability.columns) == {
        "lower",
        "upper",
        "midpoint",
        "count",
        "predicted_mean",
        "observed_rate",
    }
    assert reliability["count"].sum() == len(sample_predictions)
    # Highest bin should contain the high confidence win.
    assert reliability.loc[reliability["upper"] == 0.8, "observed_rate"].iloc[0] == 1.0


def test_plot_reliability_curve_creates_file(tmp_path: Path) -> None:
    reliability = pd.DataFrame(
        {
            "midpoint": [0.1, 0.3, 0.5, 0.7, 0.9],
            "observed_rate": [0.05, 0.25, 0.55, 0.7, 0.95],
        }
    )

    output = plot_reliability_curve(reliability, path=tmp_path / "reliability.png")

    assert output.exists()
    assert output.stat().st_size > 0


def test_compute_metrics_requires_columns(sample_predictions: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        compute_classification_metrics(
            sample_predictions.drop(columns=["prob"]),
            probability_column="prob",
            label_column="label",
        )


def test_compute_reliability_requires_valid_bins(sample_predictions: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        compute_reliability_table(
            sample_predictions,
            probability_column="prob",
            label_column="label",
            bins=[0.0, 0.0, 1.0],
        )

