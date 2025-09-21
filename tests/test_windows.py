import pandas as pd
import pytest

from nfl_pred.features.windows import RollingMetric, compute_group_rolling_windows


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023] * 8,
            "team": ["A"] * 5 + ["B"] * 3,
            "week": [1, 2, 3, 4, 5, 1, 2, 3],
            "value": [10, 20, 30, 40, 50, 5, 15, 35],
            "numerator": [1, 2, 1, 3, 2, 1, 3, 2],
            "denominator": [2, 4, 2, 6, 4, 2, 4, 5],
            "asof_ts": pd.to_datetime(
                [
                    "2023-09-01",
                    "2023-09-08",
                    "2023-09-15",
                    "2023-09-22",
                    "2023-09-29",
                    "2023-09-02",
                    "2023-09-09",
                    "2023-09-16",
                ]
            ),
        }
    )


def test_rolling_means_and_rates(sample_df: pd.DataFrame) -> None:
    metrics = [
        RollingMetric(name="value", value_column="value", statistic="mean"),
        RollingMetric(
            name="rate",
            value_column="numerator",
            denominator_column="denominator",
            statistic="rate",
        ),
    ]
    windows = {"w4": 4, "w8": 8, "season": None}

    result = compute_group_rolling_windows(
        sample_df,
        metrics=metrics,
        group_keys=["season", "team"],
        order_key="week",
        window_lengths=windows,
    )

    team_a = result[result["team"] == "A"]
    expected_a_w4 = [10.0, 15.0, 20.0, 25.0, 35.0]
    expected_a_w8 = [10.0, 15.0, 20.0, 25.0, 30.0]
    expected_a_season = [10.0, 15.0, 20.0, 25.0, 30.0]
    expected_a_rate = [0.5] * 5

    assert team_a["value_w4"].tolist() == expected_a_w4
    assert team_a["value_w8"].tolist() == expected_a_w8
    assert team_a["value_season"].tolist() == expected_a_season
    assert team_a["rate_w4"].tolist() == expected_a_rate
    assert team_a["rate_w8"].tolist() == expected_a_rate
    assert team_a["rate_season"].tolist() == expected_a_rate

    team_b = result[result["team"] == "B"]
    expected_b_w4 = [5.0, 10.0, 18.333333333333332]
    expected_b_w8 = [5.0, 10.0, 18.333333333333332]
    expected_b_season = [5.0, 10.0, 18.333333333333332]
    expected_b_rate = [0.5, 0.6666666666666666, 0.5454545454545454]

    assert team_b["value_w4"].tolist() == expected_b_w4
    assert team_b["value_w8"].tolist() == expected_b_w8
    assert team_b["value_season"].tolist() == expected_b_season
    assert team_b["rate_w4"].tolist() == expected_b_rate
    assert team_b["rate_w8"].tolist() == expected_b_rate
    assert team_b["rate_season"].tolist() == expected_b_rate


def test_asof_filter(sample_df: pd.DataFrame) -> None:
    metrics = [RollingMetric(name="value", value_column="value", statistic="mean")]
    windows = {"w4": 4, "season": None}

    cutoff = pd.Timestamp("2023-09-15")
    result = compute_group_rolling_windows(
        sample_df,
        metrics=metrics,
        group_keys=["season", "team"],
        order_key="week",
        window_lengths=windows,
        asof_ts=cutoff,
    )

    assert result["week"].max() == 3
    assert (result["week"] == 3).sum() == 1
    assert (result["value_season"].notna()).all()
