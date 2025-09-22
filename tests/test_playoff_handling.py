from __future__ import annotations

import pandas as pd
import pytest

from nfl_pred.features.playoffs import append_playoff_flags, compute_playoff_flags
from nfl_pred.pipeline.train import _apply_playoff_mode


def test_compute_playoff_flags_marks_postseason_games() -> None:
    schedule = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [18, 19],
            "game_id": ["2023_18_BUF_MIA", "2023_19_KC_BAL"],
            "game_type": ["REG", "POST"],
            "home_team": ["BUF", "BAL"],
            "away_team": ["MIA", "KC"],
        }
    )

    flags = compute_playoff_flags(schedule)

    expected = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "week": [18, 18, 19, 19],
            "game_id": [
                "2023_18_BUF_MIA",
                "2023_18_BUF_MIA",
                "2023_19_KC_BAL",
                "2023_19_KC_BAL",
            ],
            "team": ["BUF", "MIA", "BAL", "KC"],
            "is_postseason": [False, False, True, True],
            "season_phase": ["regular", "regular", "postseason", "postseason"],
        }
    )

    pd.testing.assert_frame_equal(flags, expected)


def test_compute_playoff_flags_fallback_without_game_type() -> None:
    schedule = pd.DataFrame(
        {
            "season": [2024],
            "week": [20],
            "game_id": ["2024_20_SB"],
            "home_team": ["AFC"],
            "away_team": ["NFC"],
        }
    )

    flags = compute_playoff_flags(schedule, postseason_game_types=())

    assert flags.loc[:, "is_postseason"].tolist() == [True, True]
    assert flags.loc[:, "season_phase"].tolist() == ["postseason", "postseason"]


def test_append_playoff_flags_merges_indicators() -> None:
    features = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [18, 19],
            "game_id": ["2023_18_BUF_MIA", "2023_19_KC_BAL"],
            "team": ["BUF", "BAL"],
            "metric": [1.0, 2.0],
        }
    )

    schedule = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [18, 19],
            "game_id": ["2023_18_BUF_MIA", "2023_19_KC_BAL"],
            "game_type": ["REG", "POST"],
            "home_team": ["BUF", "BAL"],
            "away_team": ["MIA", "KC"],
        }
    )

    enriched = append_playoff_flags(features, schedule)

    assert enriched["is_postseason"].tolist() == [False, True]
    assert enriched["season_phase"].tolist() == ["regular", "postseason"]


def test_apply_playoff_mode_filters_expected_rows() -> None:
    frame = pd.DataFrame(
        {
            "value": [1, 2, 3],
            "is_postseason": [True, False, True],
        }
    )

    included = _apply_playoff_mode(frame, mode="include")
    assert included.equals(frame)

    regular_only = _apply_playoff_mode(frame, mode="regular_only")
    assert regular_only["is_postseason"].tolist() == [False]

    postseason_only = _apply_playoff_mode(frame, mode="postseason_only")
    assert postseason_only["is_postseason"].tolist() == [True, True]

    with pytest.raises(ValueError):
        _apply_playoff_mode(frame, mode="unsupported")
