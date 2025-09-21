"""Tests for the injury rollup feature builder."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_pred.features.injury_rollups import build_injury_rollups


def test_build_injury_rollups_counts_practice_statuses() -> None:
    injuries = pd.DataFrame(
        [
            {"season": 2023, "week": 1, "team": "KC", "position": "QB", "practice_status": "DNP"},
            {"season": 2023, "week": 1, "team": "KC", "position": "QB", "practice_status": "Did Not Practice"},
            {"season": 2023, "week": 1, "team": "KC", "position": "FB", "practice_status": "Limited Participation"},
            {"season": 2023, "week": 1, "team": "KC", "position": "WR", "practice_status": "Full Participation"},
            {"season": 2023, "week": 1, "team": "KC", "position": "CB", "practice_status": "LP"},
            {"season": 2023, "week": 1, "team": "KC", "position": "CB", "practice_status": "LP"},
            {"season": 2023, "week": 1, "team": "KC", "position": "S", "practice_status": "DID NOT PARTICIPATE"},
            {"season": 2023, "week": 1, "team": "KC", "position": "LS", "practice_status": "Limited Practice"},
            {"season": 2023, "week": 1, "team": "KC", "position": "P", "practice_status": "FP"},
            {"season": 2023, "week": 1, "team": "KC", "position": "K", "practice_status": "DNP"},
            {"season": 2023, "week": 1, "team": "KC", "position": "DE", "practice_status": "DID NOT PRACTICE"},
            {"season": 2023, "week": 1, "team": "KC", "position": "QB", "practice_status": "Out"},
            {"season": 2023, "week": 1, "team": "KC", "position": "UNK", "practice_status": "DNP"},
            {"season": 2023, "week": 1, "team": "KC", "position": "RB", "practice_status": None},
            {"season": 2023, "week": 1, "team": "BUF", "position": "WR", "practice_status": "LP"},
            {"season": 2023, "week": 1, "team": "BUF", "position": "WR", "practice_status": "DNP"},
        ]
    )

    result = build_injury_rollups(injuries)

    kc_qb = result[(result["team"] == "KC") & (result["position_group"] == "QB")].iloc[0]
    assert kc_qb["dnp"] == 2
    assert kc_qb["lp"] == 0
    assert kc_qb["fp"] == 0

    kc_rb = result[(result["team"] == "KC") & (result["position_group"] == "RB")].iloc[0]
    assert kc_rb[["dnp", "fp"]].eq(0).all()
    assert kc_rb["lp"] == 1

    kc_db = result[(result["team"] == "KC") & (result["position_group"] == "DB")].iloc[0]
    assert kc_db["dnp"] == 1
    assert kc_db["lp"] == 2
    assert kc_db["fp"] == 0

    kc_st = result[(result["team"] == "KC") & (result["position_group"] == "ST")].iloc[0]
    assert kc_st.to_dict() == {"season": 2023, "week": 1, "team": "KC", "position_group": "ST", "dnp": 1, "lp": 1, "fp": 1}

    kc_dl = result[(result["team"] == "KC") & (result["position_group"] == "DL")].iloc[0]
    assert kc_dl["dnp"] == 1
    assert kc_dl["lp"] == 0
    assert kc_dl["fp"] == 0

    kc_wr = result[(result["team"] == "KC") & (result["position_group"] == "WR")].iloc[0]
    assert kc_wr["fp"] == 1
    assert kc_wr[["dnp", "lp"]].eq(0).all()

    buf_wr = result[(result["team"] == "BUF") & (result["position_group"] == "WR")].iloc[0]
    assert buf_wr.to_dict() == {"season": 2023, "week": 1, "team": "BUF", "position_group": "WR", "dnp": 1, "lp": 1, "fp": 0}

    assert set(result["position_group"]) == {"QB", "RB", "WR", "DB", "ST", "DL"}


def test_build_injury_rollups_handles_empty_frame() -> None:
    injuries = pd.DataFrame(columns=["season", "week", "team", "position", "practice_status"])

    result = build_injury_rollups(injuries)

    assert result.empty
    assert list(result.columns) == ["season", "week", "team", "position_group", "dnp", "lp", "fp"]


def test_build_injury_rollups_missing_columns() -> None:
    with pytest.raises(ValueError):
        build_injury_rollups(pd.DataFrame({"season": [2023]}))
