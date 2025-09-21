"""Tests for schedule metadata feature builder."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_pred.features.schedule_meta import compute_schedule_meta


def _schedule_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [
                2023,
                2023,
                2023,
                2023,
                2023,
                2023,
                2023,
            ],
            "week": [1, 2, 2, 4, 5, 1, 1],
            "game_id": [
                "2023_01_NE_MIA",
                "2023_02_PHI_NE",
                "2023_02_MIA_DEN",
                "2023_04_JAX_HOU",
                "2023_05_BUF_JAX",
                "2023_01_DAL_NYG",
                "2023_01_NO_CAR",
            ],
            "home_team": [
                "NE",
                "PHI",
                "MIA",
                "JAX",
                "BUF",
                "DAL",
                "CAR",
            ],
            "away_team": [
                "MIA",
                "NE",
                "DEN",
                "HOU",
                "JAX",
                "NYG",
                "NO",
            ],
            "start_time": [
                pd.Timestamp("2023-09-10 17:00:00+00:00"),
                pd.Timestamp("2023-09-15 00:20:00+00:00"),
                pd.Timestamp("2023-09-17 20:25:00+00:00"),
                pd.Timestamp("2023-10-01 17:00:00+00:00"),
                pd.Timestamp("2023-10-08 13:30:00+00:00"),
                pd.Timestamp("2023-09-11 00:20:00+00:00"),
                pd.Timestamp("2023-09-19 00:15:00+00:00"),
            ],
            "weekday": [
                "Sunday",
                "Thursday",
                "Sunday",
                "Sunday",
                "Sunday",
                "Sunday",
                "Monday",
            ],
            "neutral_site": [False, False, False, False, True, False, False],
        }
    )


def test_compute_schedule_meta_rest_and_short_week() -> None:
    schedule = _schedule_frame()

    meta = compute_schedule_meta(schedule)

    ne_week2 = meta.loc[(meta["team"] == "NE") & (meta["game_id"] == "2023_02_PHI_NE")]
    expected_rest = (
        pd.Timestamp("2023-09-15 00:20:00+00:00") - pd.Timestamp("2023-09-10 17:00:00+00:00")
    ).total_seconds() / 86400.0
    assert ne_week2["rest_days"].iloc[0] == pytest.approx(expected_rest, rel=1e-3)
    assert bool(ne_week2["short_week"].iloc[0]) is True

    mia_week2 = meta.loc[(meta["team"] == "MIA") & (meta["game_id"] == "2023_02_MIA_DEN")]
    assert bool(mia_week2["short_week"].iloc[0]) is False

    ne_week1 = meta.loc[(meta["team"] == "NE") & (meta["game_id"] == "2023_01_NE_MIA")]
    assert pd.isna(ne_week1["rest_days"].iloc[0])
    assert bool(ne_week1["short_week"].iloc[0]) is False


def test_compute_schedule_meta_kickoff_buckets_and_site() -> None:
    schedule = _schedule_frame()

    meta = compute_schedule_meta(schedule)

    snf = meta.loc[meta["game_id"] == "2023_01_DAL_NYG", "kickoff_bucket"].unique()
    assert snf.tolist() == ["snf"]

    mnf = meta.loc[meta["game_id"] == "2023_01_NO_CAR", "kickoff_bucket"].unique()
    assert mnf.tolist() == ["mnf"]

    early = meta.loc[meta["game_id"] == "2023_01_NE_MIA", "kickoff_bucket"].unique()
    assert early.tolist() == ["early"]

    late = meta.loc[meta["game_id"] == "2023_02_MIA_DEN", "kickoff_bucket"].unique()
    assert late.tolist() == ["late"]

    neutral = meta.loc[(meta["game_id"] == "2023_05_BUF_JAX") & (meta["team"] == "JAX")]
    assert neutral["home_away"].iloc[0] == "neutral"

    home = meta.loc[(meta["game_id"] == "2023_04_JAX_HOU") & (meta["team"] == "JAX")]
    assert home["home_away"].iloc[0] == "home"

    away = meta.loc[(meta["game_id"] == "2023_04_JAX_HOU") & (meta["team"] == "HOU")]
    assert away["home_away"].iloc[0] == "away"
