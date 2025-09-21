import pandas as pd
import pytest

from nfl_pred.visibility import compute_week_asof, filter_visible_rows


@pytest.fixture
def sample_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "start_time": [
                "2023-09-07T20:20:00Z",
                "2023-09-14T20:15:00Z",
                "2023-09-21T20:15:00Z",
            ],
        }
    )


def test_compute_week_asof_returns_latest_kickoff(sample_schedule: pd.DataFrame) -> None:
    asof = compute_week_asof(sample_schedule, season=2023, week=2)

    expected = pd.Timestamp("2023-09-14T20:15:00Z").tz_convert("UTC")
    assert asof == expected
    assert asof.tzinfo is not None and asof.tzinfo.utcoffset(asof) == pd.Timedelta(0)


def test_compute_week_asof_returns_none_without_timestamps(sample_schedule: pd.DataFrame) -> None:
    schedule = sample_schedule.copy()
    schedule.pop("start_time")

    result = compute_week_asof(schedule, season=2023, week=2)

    assert result is None


def test_filter_visible_rows_prefers_event_time(sample_schedule: pd.DataFrame) -> None:
    asof = compute_week_asof(sample_schedule, season=2023, week=2)
    data = pd.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "event_time": pd.to_datetime(
                [
                    "2023-09-10T17:00:00Z",
                    "2023-09-14T20:15:00Z",
                    "2023-09-24T17:00:00Z",
                ]
            ),
            "value": [10, 20, 30],
        }
    )

    filtered = filter_visible_rows(data, season=2023, week=2, asof_ts=asof)

    assert filtered["week"].tolist() == [1, 2]
    assert filtered["value"].tolist() == [10, 20]


def test_filter_visible_rows_falls_back_to_week_columns() -> None:
    data = pd.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )

    filtered = filter_visible_rows(data, season=2023, week=2, asof_ts=None)

    assert filtered["week"].tolist() == [1, 2]
    assert filtered["value"].tolist() == [10, 20]


def test_filter_visible_rows_uses_week_fallback_for_missing_event_time(sample_schedule: pd.DataFrame) -> None:
    asof = compute_week_asof(sample_schedule, season=2023, week=2)
    data = pd.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "event_time": [pd.NaT, pd.NaT, pd.NaT],
            "value": [10, 20, 30],
        }
    )

    filtered = filter_visible_rows(data, season=2023, week=2, asof_ts=asof)

    assert filtered["week"].tolist() == [1, 2]
    assert filtered["value"].tolist() == [10, 20]
