from __future__ import annotations

import pandas as pd
import pytest

from nfl_pred.picks import ConfidenceThresholds, assign_pick_confidence


def test_assign_pick_confidence_applies_expected_tiers() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "p_home_win": [0.70, 0.60, 0.54, 0.65],
            "p_away_win": [0.30, 0.40, 0.46, 0.35],
        }
    )

    result = assign_pick_confidence(frame)

    assert list(result["pick"]) == ["home", "home", "home", "home"]
    assert list(result["confidence"]) == ["Strong", "Lean", "Pass", "Strong"]


def test_assign_pick_confidence_handles_ties_deterministically() -> None:
    frame = pd.DataFrame(
        {
            "p_home_win": [0.50, 0.50],
            "p_away_win": [0.50, 0.50],
        }
    )

    home_pref = assign_pick_confidence(frame)
    away_pref = assign_pick_confidence(frame, prefer_home_on_tie=False)

    assert list(home_pref["pick"]) == ["home", "home"]
    assert list(away_pref["pick"]) == ["away", "away"]


def test_assign_pick_confidence_validates_columns() -> None:
    frame = pd.DataFrame({"p_home_win": [0.6]})

    with pytest.raises(KeyError):
        assign_pick_confidence(frame)


def test_threshold_configuration_enforces_ordering() -> None:
    with pytest.raises(ValueError):
        ConfidenceThresholds(strong=0.50, lean=0.55)

    thresholds = ConfidenceThresholds(strong=0.7, lean=0.6)

    frame = pd.DataFrame({"p_home_win": [0.65], "p_away_win": [0.35]})
    result = assign_pick_confidence(frame, thresholds=thresholds)
    assert result.loc[0, "confidence"] == "Lean"
