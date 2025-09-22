"""Boundary tests for rule and playoff flag helpers."""

import pandas as pd
from pandas.testing import assert_frame_equal

from nfl_pred.features.playoffs import compute_playoff_flags
from nfl_pred.features.rules import compute_rule_flags


def test_compute_rule_flags_cutovers_and_postseason_weeks() -> None:
    """Kickoff and overtime flags flip at the documented season boundaries."""

    schedule_meta = pd.DataFrame(
        {
            "season": [2023, 2024, 2024, 2025, 2025],
            "week": [10, 1, 19, 1, 19],
            "game_id": [
                "2023_10_NE_BUF",
                "2024_01_NE_BUF",
                "2024_19_NE_KC",
                "2025_01_NE_BUF",
                "2025_19_NE_KC",
            ],
            "team": ["NE", "NE", "KC", "NE", "KC"],
        }
    )

    result = compute_rule_flags(schedule_meta)

    expected = pd.DataFrame(
        {
            "season": [2023, 2024, 2024, 2025, 2025],
            "week": [10, 1, 19, 1, 19],
            "game_id": [
                "2023_10_NE_BUF",
                "2024_01_NE_BUF",
                "2024_19_NE_KC",
                "2025_01_NE_BUF",
                "2025_19_NE_KC",
            ],
            "team": ["NE", "NE", "KC", "NE", "KC"],
            "kickoff_2024plus": [False, True, True, True, True],
            "ot_regular_2025plus": [False, False, False, True, False],
        }
    )

    assert_frame_equal(result, expected)


def test_compute_playoff_flags_uses_game_type_and_week_fallback() -> None:
    """Playoff detection honours game type labels with a week fallback."""

    schedule = pd.DataFrame(
        {
            "season": [2024, 2024, 2025],
            "week": [18, 19, 19],
            "game_id": [
                "2024_18_BUF_NE",
                "2024_19_KC_BAL",
                "2025_19_KC_BAL",
            ],
            "home_team": ["BUF", "KC", "KC"],
            "away_team": ["NE", "BAL", "BAL"],
            "game_type": ["REG", "POST", "REG"],
        }
    )

    result = compute_playoff_flags(schedule)

    expected = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024, 2025, 2025],
            "week": [18, 18, 19, 19, 19, 19],
            "game_id": [
                "2024_18_BUF_NE",
                "2024_18_BUF_NE",
                "2024_19_KC_BAL",
                "2024_19_KC_BAL",
                "2025_19_KC_BAL",
                "2025_19_KC_BAL",
            ],
            "team": ["BUF", "NE", "BAL", "KC", "BAL", "KC"],
            "is_postseason": [False, False, True, True, True, True],
            "season_phase": [
                "regular",
                "regular",
                "postseason",
                "postseason",
                "postseason",
                "postseason",
            ],
        }
    )

    assert_frame_equal(result, expected)
