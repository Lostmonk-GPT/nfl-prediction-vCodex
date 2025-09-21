"""Schedule-derived metadata features.

This module produces per-team schedule metadata that can be joined with
team-week features. The goal is to expose context that depends purely on the
league schedule such as rest days, short-week indicators, kickoff timing
categories, and whether the game is effectively home, away, or neutral.

The primary entry point :func:`compute_schedule_meta` expects the raw schedule
frame produced by :mod:`nfl_pred.ingest.schedules` and returns a
``pandas.DataFrame`` with one row per ``(season, week, game_id, team)`` tuple
containing the following columns:

``season`` (int)
    Season identifier.
``week`` (int)
    Week number from the source schedule.
``game_id`` (str)
    nflverse game identifier.
``team`` / ``opponent`` (str)
    Participating teams for the row.
``home_away`` (str)
    One of ``{"home", "away", "neutral"}`` reflecting the game site for the
    team. Neutral-site games are flagged regardless of the designated home team
    from the schedule.
``start_time`` (:class:`pandas.Timestamp`)
    Kickoff timestamp (UTC). The timestamp is passed through directly from the
    schedule but coerced to UTC-aware datetimes for consistent math.
``rest_days`` (float)
    Days between this kickoff and the team's previous game in the same season.
    ``NaN`` when no previous game exists (e.g., Week 1 or missing kickoffs).
``short_week`` (bool)
    ``True`` when ``rest_days`` is strictly less than seven, indicating a short
    turnaround relative to a standard Sunday-to-Sunday week.
``kickoff_bucket`` (str)
    One of ``{"early", "late", "snf", "mnf"}`` describing the kickoff window
    using Eastern Time heuristics. Prime-time games take precedence over the
    generic early/late buckets.

The implementation intentionally keeps timezone handling simple for the MVP by
deriving all buckets from the schedule ``start_time`` column converted to
Eastern Time. Stadium-specific timezones will be incorporated in a later phase.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS: Final[set[str]] = {
    "season",
    "week",
    "game_id",
    "start_time",
    "home_team",
    "away_team",
}


def compute_schedule_meta(schedule: pd.DataFrame) -> pd.DataFrame:
    """Derive per-team schedule metadata features.

    Args:
        schedule: Raw schedule frame containing at least the required columns
            listed in ``_REQUIRED_COLUMNS`` and optionally ``weekday``,
            ``neutral_site``, or ``location``.

    Returns:
        ``DataFrame`` with one row per team/game including rest, kickoff
        buckets, and home/away context.
    """

    _validate_schedule(schedule)
    working = schedule.copy()

    working["season"] = working["season"].astype(int)
    working["week"] = working["week"].astype(int)

    kickoff_utc = pd.to_datetime(working["start_time"], utc=True, errors="coerce")
    working["start_time"] = kickoff_utc

    weekday = None
    if "weekday" in working.columns:
        weekday = working["weekday"].astype(str).str.lower()

    neutral_flag = _extract_neutral_flag(working)

    base_columns = ["season", "week", "game_id", "start_time"]

    extra_columns = ["home_team", "away_team"]
    if weekday is not None:
        extra_columns.append("weekday")

    home = working[base_columns + extra_columns].copy()
    home["team"] = home["home_team"].astype(str)
    home["opponent"] = home["away_team"].astype(str)
    home["home_away"] = np.where(neutral_flag, "neutral", "home")

    away = working[base_columns + extra_columns].copy()
    away["team"] = away["away_team"].astype(str)
    away["opponent"] = away["home_team"].astype(str)
    away["home_away"] = np.where(neutral_flag, "neutral", "away")

    combined = pd.concat([home, away], ignore_index=True, sort=False)
    if "home_team" in combined.columns:
        combined = combined.drop(columns=["home_team", "away_team"])

    combined = combined.sort_values(["team", "season", "start_time", "game_id"]).reset_index(drop=True)

    combined["rest_days"] = _compute_rest_days(combined)
    combined["short_week"] = combined["rest_days"].lt(7.0).fillna(False).astype(bool)

    weekday_series = combined.get("weekday")
    combined["kickoff_bucket"] = _classify_kickoff_bucket(
        combined["start_time"], weekday=weekday_series
    )
    if "weekday" in combined.columns:
        combined = combined.drop(columns=["weekday"])

    ordered_columns = [
        "season",
        "week",
        "game_id",
        "team",
        "opponent",
        "home_away",
        "start_time",
        "rest_days",
        "short_week",
        "kickoff_bucket",
    ]

    combined = combined[ordered_columns].sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)

    return combined


def _validate_schedule(schedule: pd.DataFrame) -> None:
    missing = sorted(column for column in _REQUIRED_COLUMNS if column not in schedule.columns)
    if missing:
        raise KeyError(f"Schedule frame missing required columns: {missing}")


def _extract_neutral_flag(schedule: pd.DataFrame) -> pd.Series:
    """Return a boolean Series indicating neutral-site games."""

    if "neutral_site" in schedule.columns:
        neutral_raw = schedule["neutral_site"]
    elif "location" in schedule.columns:
        neutral_raw = schedule["location"].astype(str)
        return neutral_raw.str.lower().str.strip().eq("neutral")
    else:
        return pd.Series(False, index=schedule.index, dtype=bool)

    if pd.api.types.is_bool_dtype(neutral_raw):
        return neutral_raw.fillna(False)

    if pd.api.types.is_numeric_dtype(neutral_raw):
        return neutral_raw.fillna(0).astype(int).astype(bool)

    neutral_str = neutral_raw.astype(str).str.lower().str.strip()
    truthy = {"true", "t", "yes", "y", "1", "neutral"}
    return neutral_str.isin(truthy)


def _compute_rest_days(team_games: pd.DataFrame) -> pd.Series:
    """Compute rest days relative to the previous team game in the same season."""

    prev_start = (
        team_games.groupby(["team", "season"], group_keys=False)["start_time"].shift(1)
    )

    rest = (team_games["start_time"] - prev_start).dt.total_seconds() / 86400.0
    return rest


def _classify_kickoff_bucket(
    kickoff: pd.Series, *, weekday: pd.Series | None
) -> pd.Series:
    """Classify kickoff timestamps into coarse buckets."""

    if kickoff.dt.tz is not None:
        kickoff_et = kickoff.dt.tz_convert("US/Eastern")
    else:
        kickoff_et = kickoff

    if weekday is not None:
        weekday_norm = weekday.fillna("").astype(str).str.lower()
    else:
        weekday_norm = kickoff_et.dt.day_name().str.lower()

    hours = kickoff_et.dt.hour + kickoff_et.dt.minute / 60.0

    bucket = np.full(len(kickoff_et), "late", dtype=object)
    bucket = np.where(hours < 16.0, "early", bucket)
    bucket = np.where((weekday_norm == "sunday") & (hours >= 19.0), "snf", bucket)
    bucket = np.where(weekday_norm == "monday", "mnf", bucket)

    return pd.Series(bucket, index=kickoff.index, dtype="object")


__all__ = ["compute_schedule_meta"]
