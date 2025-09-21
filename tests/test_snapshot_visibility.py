from __future__ import annotations

import pandas as pd
import pytest

from nfl_pred.snapshot.visibility import (
    VisibilityContext,
    filter_play_by_play,
    filter_schedule,
    filter_weekly_frame,
)


def test_filter_play_by_play_enforces_event_time_cutoff() -> None:
    context = VisibilityContext(asof_ts=pd.Timestamp("2023-09-15T00:00:00Z"))
    pbp = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 2, 2, 3],
            "event_time": [
                "2023-09-10T17:00:00Z",
                "2023-09-14T23:00:00Z",
                pd.NaT,
                pd.NaT,
            ],
            "asof_ts": [
                "2023-09-10T20:00:00Z",
                "2023-09-14T23:30:00Z",
                "2023-09-14T12:00:00Z",
                "2023-09-17T12:00:00Z",
            ],
            "value": [1, 2, 3, 4],
        }
    )

    filtered = filter_play_by_play(pbp, context=context)

    assert filtered["value"].tolist() == [1, 2, 3]
    assert 4 not in filtered["value"].tolist()


def test_filter_schedule_applies_asof_and_week() -> None:
    schedule = pd.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "start_time": [
                "2023-09-10T17:00:00Z",
                "2023-09-17T20:25:00Z",
                "2023-09-24T20:20:00Z",
            ],
            "game_id": ["g1", "g2", "g3"],
        }
    )

    context = VisibilityContext(season=2023, week=2, asof_ts=pd.Timestamp("2023-09-19T00:00:00Z"))
    filtered = filter_schedule(schedule, context=context)

    assert filtered["game_id"].tolist() == ["g1", "g2"]


def test_filter_weekly_frame_requires_columns() -> None:
    df = pd.DataFrame({"season": [2023], "value": [1]})

    with pytest.raises(KeyError):
        filter_weekly_frame(df, season=2023, week=1)
