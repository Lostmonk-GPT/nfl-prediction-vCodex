"""Unit tests verifying injury rollups respect visibility cutoffs."""

import pandas as pd
from pandas.testing import assert_frame_equal

from nfl_pred.features import build_injury_rollups
from nfl_pred.visibility import filter_visible_rows


def test_injury_rollups_exclude_rows_after_cutoff() -> None:
    injuries = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023, 2023],
            "week": [3, 3, 3, 3, 4],
            "team": ["NE", "NE", "NE", "NE", "NE"],
            "position": ["WR", "WR", "TE", "QB", "QB"],
            "practice_status": ["DNP", "LP", "FP", "LP", "DNP"],
            "event_time": pd.to_datetime(
                [
                    "2023-09-13T17:00:00Z",  # Visible: before cutoff
                    "2023-09-16T17:00:00Z",  # Hidden: after cutoff
                    "2023-09-14T17:00:00Z",  # Visible: before cutoff
                    None,  # Visible via week fallback (no timestamp)
                    None,  # Hidden via week fallback (week > target)
                ]
            ),
        }
    )

    asof_ts = pd.Timestamp("2023-09-15T12:00:00Z")

    visible = filter_visible_rows(
        injuries,
        season=2023,
        week=3,
        asof_ts=asof_ts,
        event_time_col="event_time",
        season_col="season",
        week_col="week",
    )

    assert not (visible["event_time"] > asof_ts).any()

    rollups = build_injury_rollups(visible)
    rollups = rollups.sort_values(["season", "week", "team", "position_group"]).reset_index(
        drop=True
    )
    rollups.columns.name = None

    expected = pd.DataFrame(
        {
            "season": [2023, 2023, 2023],
            "week": [3, 3, 3],
            "team": ["NE", "NE", "NE"],
            "position_group": ["QB", "TE", "WR"],
            "dnp": [0, 0, 1],
            "lp": [1, 0, 0],
            "fp": [0, 1, 0],
        }
    ).astype({"dnp": "int64", "lp": "int64", "fp": "int64"})

    assert_frame_equal(rollups, expected)
