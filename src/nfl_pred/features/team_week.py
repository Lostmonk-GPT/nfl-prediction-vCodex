"""Team-week feature aggregation from play-by-play data.

This module provides utilities to aggregate nflfastR style play-by-play data
to the team/week level and derive both single-week and rolling-window metrics.

Output Schema
-------------
The primary public function :func:`compute_team_week_features` returns a
``pandas.DataFrame`` with one row per ``(season, week, team)`` triple. The
following columns are included:

``season`` (int)
    Season identifier from the source data.
``week`` (int)
    Week number (regular season and postseason supported).
``team`` (str)
    Offense/posteam abbreviation used by nflfastR.
``plays_offense`` (int)
    Number of offensive plays (pass + rush attempts).
``pass_plays`` / ``rush_plays`` (int)
    Counts of pass and rush attempts respectively.
``dropbacks`` (int)
    Quarterback dropbacks (includes sacks and scrambles).
``sacks`` (int)
    Recorded sacks against the offense.
``success_plays`` (float)
    Sum of ``success`` for offensive plays (acts as numerator for
    ``success_rate``).
``early_down_plays`` (int)
    Offensive plays on 1st/2nd down (supports early-down EPA metrics).
``play_action_plays`` / ``shotgun_plays`` / ``no_huddle_plays`` (int)
    Counts of plays flagged with each respective attribute.
``explosive_pass_plays`` / ``explosive_rush_plays`` (int)
    Counts of explosive gains (>=15 pass yards, >=10 rush yards).
``penalties`` (int)
    Offensive penalties charged to the team.
``st_plays`` (int)
    Special teams plays (punts, kickoffs, field goals, etc.).
``total_epa`` / ``early_down_epa`` / ``st_epa`` (float)
    EPA numerators for overall, early-down, and special teams aggregates.
``epa_per_play`` / ``early_down_epa_per_play`` / ``success_rate`` /
``pass_rate`` / ``rush_rate`` / ``play_action_rate`` / ``shotgun_rate`` /
``no_huddle_rate`` / ``sack_rate`` / ``explosive_pass_rate`` /
``explosive_rush_rate`` / ``penalty_rate`` / ``st_epa_per_play`` (float)
    Single-week efficiency metrics derived from the numerator/denominator
    columns above.

In addition to the single-week metrics, rolling windows are appended for each
rate/EPA metric using 4-week, 8-week, and season-to-date (expanding) windows.
For example ``epa_per_play_w4`` and ``epa_per_play_season`` provide the
corresponding rolling averages.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import pandas as pd

from nfl_pred.features.windows import RollingMetric, compute_group_rolling_windows

TEAM_WEEK_WINDOW_LENGTHS: Final[dict[str, int | None]] = {"w4": 4, "w8": 8, "season": None}

_REQUIRED_COLUMNS: Final[set[str]] = {
    "season",
    "week",
    "posteam",
    "epa",
    "success",
    "down",
    "pass",
    "rush",
    "qb_dropback",
    "sack",
    "play_action",
    "shotgun",
    "no_huddle",
    "yards_gained",
}


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(column for column in _REQUIRED_COLUMNS if column not in df.columns)
    if missing:
        raise KeyError(f"Play-by-play frame missing required columns: {missing}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while protecting against division-by-zero."""

    result = numerator.divide(denominator)
    return result.replace([np.inf, -np.inf], np.nan)


def _prepare_source_frame(pbp: pd.DataFrame, *, asof_ts: pd.Timestamp | None) -> pd.DataFrame:
    working = pbp.copy()
    if asof_ts is not None and "asof_ts" in working.columns:
        working = working.loc[working["asof_ts"] <= asof_ts].copy()

    _ensure_required_columns(working)

    working = working.loc[working["posteam"].notna()].copy()

    bool_like_columns = [
        "pass",
        "rush",
        "qb_dropback",
        "sack",
        "play_action",
        "shotgun",
        "no_huddle",
        "penalty",
        "special_teams_play",
    ]

    for column in bool_like_columns:
        if column in working.columns:
            working[column] = working[column].fillna(0).astype(int)

    working["success"] = working["success"].fillna(0).astype(float)
    working["epa"] = working["epa"].fillna(0).astype(float)
    working["yards_gained"] = working["yards_gained"].fillna(0).astype(float)

    working["is_pass"] = working["pass"] == 1
    working["is_rush"] = working["rush"] == 1
    working["is_offensive_play"] = working["is_pass"] | working["is_rush"]
    working["is_dropback"] = working["qb_dropback"] == 1
    working["is_sack"] = working["sack"] == 1
    working["is_play_action"] = working.get("play_action", 0) == 1
    working["is_shotgun"] = working.get("shotgun", 0) == 1
    working["is_no_huddle"] = working.get("no_huddle", 0) == 1
    working["is_special_teams"] = working.get("special_teams_play", 0) == 1

    working["is_early_down"] = working["down"].isin([1, 2])
    working["is_early_down_play"] = working["is_offensive_play"] & working["is_early_down"]

    yards = working["yards_gained"]
    working["is_explosive_pass"] = working["is_pass"] & (yards >= 15)
    working["is_explosive_rush"] = working["is_rush"] & (yards >= 10)

    if "penalty" in working.columns:
        penalty_indicator = working["penalty"] == 1
        if "penalty_team" in working.columns:
            offensive_penalty = penalty_indicator & (
                working["penalty_team"].fillna("") == working["posteam"].astype(str)
            )
        else:
            offensive_penalty = penalty_indicator & working["is_offensive_play"]
        working["is_penalty"] = offensive_penalty
    else:
        working["is_penalty"] = False

    working["epa_offense"] = np.where(working["is_offensive_play"], working["epa"], 0.0)
    working["epa_early_down"] = np.where(
        working["is_offensive_play"] & working["is_early_down"], working["epa"], 0.0
    )
    working["epa_special_teams"] = np.where(working["is_special_teams"], working["epa"], 0.0)

    working["success_offense"] = np.where(working["is_offensive_play"], working["success"], 0.0)

    working["season"] = working["season"].astype(int)
    working["week"] = working["week"].astype(int)

    return working


def _aggregate_team_week(pbp: pd.DataFrame, *, asof_ts: pd.Timestamp | None) -> pd.DataFrame:
    prepared = _prepare_source_frame(pbp, asof_ts=asof_ts)
    if prepared.empty:
        columns = [
            "season",
            "week",
            "team",
            "plays_offense",
            "pass_plays",
            "rush_plays",
            "dropbacks",
            "sacks",
            "success_plays",
            "early_down_plays",
            "play_action_plays",
            "shotgun_plays",
            "no_huddle_plays",
            "explosive_pass_plays",
            "explosive_rush_plays",
            "penalties",
            "st_plays",
            "total_epa",
            "early_down_epa",
            "st_epa",
            "epa_per_play",
            "early_down_epa_per_play",
            "success_rate",
            "pass_rate",
            "rush_rate",
            "play_action_rate",
            "shotgun_rate",
            "no_huddle_rate",
            "sack_rate",
            "explosive_pass_rate",
            "explosive_rush_rate",
            "penalty_rate",
            "st_epa_per_play",
        ]
        return pd.DataFrame(columns=columns)

    grouped = prepared.groupby(["season", "week", "posteam"], sort=False)

    summary = grouped.agg(
        plays_offense=("is_offensive_play", "sum"),
        pass_plays=("is_pass", "sum"),
        rush_plays=("is_rush", "sum"),
        dropbacks=("is_dropback", "sum"),
        sacks=("is_sack", "sum"),
        success_plays=("success_offense", "sum"),
        early_down_plays=("is_early_down_play", "sum"),
        play_action_plays=("is_play_action", "sum"),
        shotgun_plays=("is_shotgun", "sum"),
        no_huddle_plays=("is_no_huddle", "sum"),
        explosive_pass_plays=("is_explosive_pass", "sum"),
        explosive_rush_plays=("is_explosive_rush", "sum"),
        penalties=("is_penalty", "sum"),
        st_plays=("is_special_teams", "sum"),
        total_epa=("epa_offense", "sum"),
        early_down_epa=("epa_early_down", "sum"),
        st_epa=("epa_special_teams", "sum"),
    )

    summary = summary.reset_index().rename(columns={"posteam": "team"})

    # Ensure numeric dtypes for denominators.
    integer_like = [
        "plays_offense",
        "pass_plays",
        "rush_plays",
        "dropbacks",
        "sacks",
        "early_down_plays",
        "play_action_plays",
        "shotgun_plays",
        "no_huddle_plays",
        "explosive_pass_plays",
        "explosive_rush_plays",
        "penalties",
        "st_plays",
    ]
    for column in integer_like:
        summary[column] = summary[column].astype(int)

    # Success plays and EPA totals are floats by nature.
    float_like = ["success_plays", "total_epa", "early_down_epa", "st_epa"]
    for column in float_like:
        summary[column] = summary[column].astype(float)

    summary["epa_per_play"] = _safe_divide(summary["total_epa"], summary["plays_offense"])
    summary["early_down_epa_per_play"] = _safe_divide(
        summary["early_down_epa"], summary["early_down_plays"]
    )
    summary["success_rate"] = _safe_divide(summary["success_plays"], summary["plays_offense"])
    summary["pass_rate"] = _safe_divide(summary["pass_plays"], summary["plays_offense"])
    summary["rush_rate"] = _safe_divide(summary["rush_plays"], summary["plays_offense"])
    summary["play_action_rate"] = _safe_divide(
        summary["play_action_plays"], summary["pass_plays"]
    )
    summary["shotgun_rate"] = _safe_divide(summary["shotgun_plays"], summary["plays_offense"])
    summary["no_huddle_rate"] = _safe_divide(summary["no_huddle_plays"], summary["plays_offense"])
    summary["sack_rate"] = _safe_divide(summary["sacks"], summary["dropbacks"])
    summary["explosive_pass_rate"] = _safe_divide(
        summary["explosive_pass_plays"], summary["pass_plays"]
    )
    summary["explosive_rush_rate"] = _safe_divide(
        summary["explosive_rush_plays"], summary["rush_plays"]
    )
    summary["penalty_rate"] = _safe_divide(summary["penalties"], summary["plays_offense"])
    summary["st_epa_per_play"] = _safe_divide(summary["st_epa"], summary["st_plays"])

    summary = summary.sort_values(["season", "team", "week"]).reset_index(drop=True)

    return summary


def compute_team_week_features(
    pbp: pd.DataFrame,
    *,
    asof_ts: pd.Timestamp | None = None,
    window_lengths: Mapping[str, int | None] | None = None,
) -> pd.DataFrame:
    """Aggregate play-by-play to team-week features and compute rolling windows."""

    aggregated = _aggregate_team_week(pbp, asof_ts=asof_ts)
    if aggregated.empty:
        return aggregated

    windows = dict(window_lengths or TEAM_WEEK_WINDOW_LENGTHS)
    metrics = [
        RollingMetric(
            name="epa_per_play",
            value_column="total_epa",
            denominator_column="plays_offense",
            statistic="rate",
        ),
        RollingMetric(
            name="early_down_epa_per_play",
            value_column="early_down_epa",
            denominator_column="early_down_plays",
            statistic="rate",
        ),
        RollingMetric(
            name="success_rate",
            value_column="success_plays",
            denominator_column="plays_offense",
            statistic="rate",
        ),
        RollingMetric(
            name="pass_rate",
            value_column="pass_plays",
            denominator_column="plays_offense",
            statistic="rate",
        ),
        RollingMetric(
            name="rush_rate",
            value_column="rush_plays",
            denominator_column="plays_offense",
            statistic="rate",
        ),
        RollingMetric(
            name="play_action_rate",
            value_column="play_action_plays",
            denominator_column="pass_plays",
            statistic="rate",
        ),
        RollingMetric(
            name="shotgun_rate",
            value_column="shotgun_plays",
            denominator_column="plays_offense",
            statistic="rate",
        ),
        RollingMetric(
            name="no_huddle_rate",
            value_column="no_huddle_plays",
            denominator_column="plays_offense",
            statistic="rate",
        ),
        RollingMetric(
            name="sack_rate",
            value_column="sacks",
            denominator_column="dropbacks",
            statistic="rate",
        ),
        RollingMetric(
            name="explosive_pass_rate",
            value_column="explosive_pass_plays",
            denominator_column="pass_plays",
            statistic="rate",
        ),
        RollingMetric(
            name="explosive_rush_rate",
            value_column="explosive_rush_plays",
            denominator_column="rush_plays",
            statistic="rate",
        ),
        RollingMetric(
            name="penalty_rate",
            value_column="penalties",
            denominator_column="plays_offense",
            statistic="rate",
        ),
        RollingMetric(
            name="st_epa_per_play",
            value_column="st_epa",
            denominator_column="st_plays",
            statistic="rate",
        ),
    ]

    rolling = compute_group_rolling_windows(
        aggregated,
        metrics=metrics,
        group_keys=["season", "team"],
        order_key="week",
        window_lengths=windows,
    )

    result = aggregated.merge(rolling, on=["season", "team", "week"], how="left")
    return result


__all__ = ["compute_team_week_features", "TEAM_WEEK_WINDOW_LENGTHS"]

