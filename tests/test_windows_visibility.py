import pandas as pd
from pandas.testing import assert_frame_equal

from nfl_pred.features.windows import (
    RollingMetric,
    compute_group_rolling_windows,
)
from nfl_pred.visibility import filter_visible_rows


def test_compute_group_rolling_windows_mean_and_rate() -> None:
    df = pd.DataFrame(
        {
            "season": [2023] * 5,
            "team": ["NE"] * 5,
            "week": [1, 2, 3, 4, 5],
            "yards": [100, 110, 90, 120, 130],
            "successes": [5, 6, 4, 7, 8],
            "plays": [10, 12, 8, 14, 16],
        }
    )

    metrics = [
        RollingMetric(name="yards", value_column="yards", statistic="mean"),
        RollingMetric(
            name="success_rate",
            value_column="successes",
            denominator_column="plays",
            statistic="rate",
        ),
    ]

    result = compute_group_rolling_windows(
        df,
        metrics=metrics,
        group_keys=["season", "team"],
        order_key="week",
        window_lengths={"w4": 4, "season": None},
    )

    expected = pd.DataFrame(
        {
            "season": [2023] * 5,
            "team": ["NE"] * 5,
            "week": [1, 2, 3, 4, 5],
            "yards_w4": [100.0, 105.0, 100.0, 105.0, 112.5],
            "yards_season": [100.0, 105.0, 100.0, 105.0, 110.0],
            "success_rate_w4": [0.5, 0.5, 0.5, 0.5, 0.5],
            "success_rate_season": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )

    assert_frame_equal(result, expected)


def test_compute_group_rolling_windows_respects_asof_timestamp() -> None:
    df = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023, 2023, 2023],
            "team": ["A", "A", "A", "B", "B", "B"],
            "week": [1, 2, 3, 1, 2, 3],
            "metric": [1, 2, 3, 4, 5, 6],
            "asof_ts": pd.to_datetime(
                [
                    "2023-09-05T12:00:00Z",
                    "2023-09-12T12:00:00Z",
                    "2023-09-26T12:00:00Z",
                    "2023-09-06T12:00:00Z",
                    "2023-09-13T12:00:00Z",
                    "2023-09-14T12:00:00Z",
                ]
            ),
        }
    )

    metric = RollingMetric(name="metric", value_column="metric", statistic="mean")

    result = compute_group_rolling_windows(
        df,
        metrics=[metric],
        group_keys=["season", "team"],
        order_key="week",
        window_lengths={"w2": 2},
        asof_ts=pd.Timestamp("2023-09-18T00:00:00Z"),
        asof_column="asof_ts",
    )

    expected = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023, 2023],
            "team": ["A", "A", "B", "B", "B"],
            "week": [1, 2, 1, 2, 3],
            "metric_w2": [1.0, 1.5, 4.0, 4.5, 5.5],
        }
    )

    assert_frame_equal(result, expected)


def test_filter_visible_rows_prefers_event_time_then_week() -> None:
    df = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023, 2022],
            "week": [1, 1, 2, 3, 17],
            "event_time": [
                pd.Timestamp("2023-09-10T17:00:00Z"),
                pd.NaT,
                pd.Timestamp("2023-09-17T17:00:00Z"),
                pd.Timestamp("2023-09-24T17:00:00Z"),
                pd.NaT,
            ],
            "value": [10, 20, 30, 40, 50],
        }
    )

    visible = filter_visible_rows(
        df,
        season=2023,
        week=2,
        asof_ts=pd.Timestamp("2023-09-18T00:00:00Z"),
        event_time_col="event_time",
        season_col="season",
        week_col="week",
    )

    expected_indices = [0, 1, 2, 4]
    assert visible.index.tolist() == expected_indices
    assert visible["value"].tolist() == [10, 20, 30, 50]
