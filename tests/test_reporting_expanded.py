"""Tests for expanded evaluation reporting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_pred.reporting.expanded import (
    ExpandedMetricConfig,
    build_expanded_metrics,
    plot_expanded_metric,
    prepare_report_records,
    save_expanded_metrics,
)
from nfl_pred.reporting.metrics import compute_classification_metrics


@pytest.fixture()
def sample_evaluation_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023, 2023, 2023],
            "week": [1, 1, 2, 2, 3, 3],
            "p_home_win": [0.7, 0.35, 0.6, 0.4, 0.8, 0.2],
            "label_home_win": [1, 0, 1, 0, 0, 1],
        }
    )


def test_build_expanded_metrics_windows(sample_evaluation_data: pd.DataFrame) -> None:
    cfg = ExpandedMetricConfig(rolling_window=2)
    expanded = build_expanded_metrics(
        sample_evaluation_data,
        probability_column="p_home_win",
        label_column="label_home_win",
        config=cfg,
    )

    assert set(expanded.columns) == {
        cfg.season_column,
        cfg.week_column,
        cfg.window_column,
        cfg.slice_column,
        "brier_score",
        "log_loss",
        "n_observations",
    }

    weekly_week1 = expanded[
        (expanded[cfg.window_column] == "weekly")
        & (expanded[cfg.slice_column] == "overall")
        & (expanded[cfg.week_column] == 1)
    ].iloc[0]
    expected_week1 = compute_classification_metrics(
        sample_evaluation_data.loc[sample_evaluation_data["week"] == 1],
        probability_column="p_home_win",
        label_column="label_home_win",
    ).iloc[0]
    assert pytest.approx(weekly_week1["brier_score"], rel=1e-6) == expected_week1["brier_score"]

    season_to_date_week2 = expanded[
        (expanded[cfg.window_column] == "season_to_date")
        & (expanded[cfg.week_column] == 2)
        & (expanded[cfg.slice_column] == "overall")
    ].iloc[0]
    expected_s2d = compute_classification_metrics(
        sample_evaluation_data.loc[sample_evaluation_data["week"] <= 2],
        probability_column="p_home_win",
        label_column="label_home_win",
    ).iloc[0]
    assert pytest.approx(season_to_date_week2["log_loss"], rel=1e-6) == expected_s2d["log_loss"]

    rolling_week3 = expanded[
        (expanded[cfg.window_column] == "rolling_2")
        & (expanded[cfg.slice_column] == "overall")
        & (expanded[cfg.week_column] == 3)
    ].iloc[0]
    expected_rolling = compute_classification_metrics(
        sample_evaluation_data.loc[sample_evaluation_data["week"] >= 2],
        probability_column="p_home_win",
        label_column="label_home_win",
    ).iloc[0]
    assert pytest.approx(rolling_week3["brier_score"], rel=1e-6) == expected_rolling["brier_score"]


def test_build_expanded_metrics_slices(sample_evaluation_data: pd.DataFrame) -> None:
    cfg = ExpandedMetricConfig(rolling_window=2, favorite_threshold=0.6, underdog_threshold=0.4)
    expanded = build_expanded_metrics(
        sample_evaluation_data,
        probability_column="p_home_win",
        label_column="label_home_win",
        config=cfg,
    )

    weekly_rows = expanded.loc[expanded[cfg.window_column] == "weekly"]
    assert {"overall", "favorite", "underdog", "toss_up"}.issuperset(set(weekly_rows[cfg.slice_column]))

    favorite_rows = weekly_rows.loc[weekly_rows[cfg.slice_column] == "favorite"]
    assert not favorite_rows.empty
    expected_subset = sample_evaluation_data.loc[
        (sample_evaluation_data["week"] == 1)
        & (sample_evaluation_data["p_home_win"] >= 0.6)
    ]
    expected_metrics = compute_classification_metrics(
        expected_subset,
        probability_column="p_home_win",
        label_column="label_home_win",
    ).iloc[0]
    favorite_week1 = favorite_rows.loc[favorite_rows[cfg.week_column] == 1].iloc[0]
    assert pytest.approx(favorite_week1["log_loss"], rel=1e-6) == expected_metrics["log_loss"]


def test_prepare_report_records_creates_metric_keys(sample_evaluation_data: pd.DataFrame) -> None:
    cfg = ExpandedMetricConfig(rolling_window=2)
    expanded = build_expanded_metrics(
        sample_evaluation_data,
        probability_column="p_home_win",
        label_column="label_home_win",
        config=cfg,
    )
    week3_rows = expanded.loc[expanded[cfg.week_column] == 3]
    records = prepare_report_records(
        week3_rows,
        asof_ts=pd.Timestamp("2024-01-05T00:00:00Z"),
        snapshot_at=pd.Timestamp("2024-01-06T00:00:00Z"),
        config=cfg,
    )

    assert not records.empty
    windows = {metric.split(".")[0] for metric in records["metric"]}
    assert windows.issubset({"weekly", "season_to_date", "rolling_2"})
    assert {"season", "week", "asof_ts", "metric", "value", "snapshot_at"} == set(records.columns)


def test_plot_and_save_expanded_metrics(tmp_path: Path, sample_evaluation_data: pd.DataFrame) -> None:
    cfg = ExpandedMetricConfig(rolling_window=2)
    expanded = build_expanded_metrics(
        sample_evaluation_data,
        probability_column="p_home_win",
        label_column="label_home_win",
        config=cfg,
    )

    csv_path = save_expanded_metrics(expanded, reports_dir=tmp_path, name="expanded.csv")
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").startswith("season,week")

    plot_path = plot_expanded_metric(
        expanded,
        metric="brier_score",
        window="weekly",
        season=2023,
        reports_dir=tmp_path,
        name="weekly_brier.png",
        config=cfg,
    )
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
