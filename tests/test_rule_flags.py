"""Unit tests for rule flag feature helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_pred.features.build_features import _join_feature_components
from nfl_pred.features.rules import append_rule_flags, compute_rule_flags


def _schedule_meta_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2024, 2025, 2025],
            "week": [18, 1, 1, 20],
            "game_id": [
                "2023_18_KC_LV",
                "2024_01_NE_MIA",
                "2025_01_NE_BUF",
                "2025_20_NE_BUF",
            ],
            "team": ["KC", "NE", "NE", "NE"],
            "opponent": ["LV", "MIA", "BUF", "BUF"],
            "home_away": ["home", "home", "home", "home"],
            "start_time": [
                pd.Timestamp("2024-01-07 18:00:00+00:00"),
                pd.Timestamp("2024-09-08 17:00:00+00:00"),
                pd.Timestamp("2025-09-07 17:00:00+00:00"),
                pd.Timestamp("2026-01-18 21:30:00+00:00"),
            ],
            "rest_days": [7.0, 7.0, 7.0, 7.0],
            "short_week": [False, False, False, False],
            "kickoff_bucket": ["late", "early", "early", "late"],
        }
    )


def test_compute_rule_flags_activation_boundaries() -> None:
    schedule_meta = _schedule_meta_frame()

    flags = compute_rule_flags(schedule_meta)

    kc_row = flags.loc[flags["game_id"] == "2023_18_KC_LV"].iloc[0]
    assert bool(kc_row["kickoff_2024plus"]) is False
    assert bool(kc_row["ot_regular_2025plus"]) is False

    kickoff_row = flags.loc[flags["game_id"] == "2024_01_NE_MIA"].iloc[0]
    assert bool(kickoff_row["kickoff_2024plus"]) is True
    assert bool(kickoff_row["ot_regular_2025plus"]) is False

    ot_row = flags.loc[flags["game_id"] == "2025_01_NE_BUF"].iloc[0]
    assert bool(ot_row["kickoff_2024plus"]) is True
    assert bool(ot_row["ot_regular_2025plus"]) is True

    postseason_row = flags.loc[flags["game_id"] == "2025_20_NE_BUF"].iloc[0]
    assert bool(postseason_row["kickoff_2024plus"]) is True
    assert bool(postseason_row["ot_regular_2025plus"]) is False


def test_append_rule_flags_preserves_shape_and_dtypes() -> None:
    schedule_meta = _schedule_meta_frame()

    augmented = append_rule_flags(schedule_meta)

    assert augmented.shape[0] == schedule_meta.shape[0]
    assert pd.api.types.is_bool_dtype(augmented["kickoff_2024plus"])
    assert pd.api.types.is_bool_dtype(augmented["ot_regular_2025plus"])

    with pytest.raises(KeyError):
        compute_rule_flags(schedule_meta.drop(columns=["team"]))


def test_rule_flags_flow_through_feature_join() -> None:
    schedule_meta = append_rule_flags(_schedule_meta_frame())

    travel = schedule_meta[["season", "week", "game_id", "team"]].copy()
    travel["travel_miles"] = [0.0, 150.0, 200.0, 400.0]

    team_week = pd.DataFrame(
        {
            "season": [2023, 2024, 2025],
            "week": [18, 1, 1],
            "team": ["KC", "NE", "NE"],
            "plays_offense": [60, 58, 62],
        }
    )

    weather = schedule_meta[["season", "week", "game_id", "team"]].copy()
    weather["wx_temp"] = [35.0, 70.0, 68.0, 42.0]

    assembled = _join_feature_components(
        schedule_meta=schedule_meta,
        travel_features=travel,
        team_week_features=team_week,
        weather_features=weather,
    )

    assert {"kickoff_2024plus", "ot_regular_2025plus"} <= set(assembled.columns)

    ot_rows = assembled.loc[assembled["ot_regular_2025plus"]]
    assert ot_rows["week"].tolist() == [1]
