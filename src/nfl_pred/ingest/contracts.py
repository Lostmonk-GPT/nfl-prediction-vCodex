"""Dataframe contract helpers for ingestion datasets.

This module provides simple column-level assertions for the raw datasets
retrieved via :mod:`nflreadpy`.  The goal is to guard downstream code against
upstream schema drift by ensuring the minimum viable column sets required by
feature builders remain available before any heavy processing begins.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


SCHEDULE_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "gametime",
        "weekday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "location",
        "stadium",
        "surface",
        "roof",
        "neutral_site",
    }
)


PBP_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "game_id",
        "play_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "play_type",
        "yards_gained",
        "epa",
        "air_yards",
        "success",
        "down",
        "pass",
        "rush",
        "play_action",
        "shotgun",
        "no_huddle",
        "sack",
        "penalty",
        "touchdown",
        "posteam_score",
        "defteam_score",
        "score_differential",
        "half_seconds_remaining",
    }
)


ROSTER_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "player_id",
        "gsis_id",
        "season",
        "team",
        "position",
        "depth_chart_position",
        "status",
        "full_name",
    }
)


TEAM_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "team_abbr",
        "team_name",
        "team_id",
        "conference",
        "division",
        "full_name",
    }
)


def _assert_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    dataset_name: str,
) -> None:
    """Raise ``ValueError`` if ``required`` columns are missing from ``df``.

    Args:
        df: Dataframe returned by an ingestion routine.
        required: Iterable of column names that must exist on ``df``.
        dataset_name: Human-friendly dataset label used in error messages.

    Raises:
        ValueError: When any required column is absent.
    """

    present_columns = set(df.columns)
    missing = set(required) - present_columns
    if not missing:
        return

    sample_present = sorted(present_columns)[:5]
    sample_display = ", ".join(sample_present) if sample_present else "<no columns>"
    missing_display = ", ".join(sorted(missing))
    raise ValueError(
        f"{dataset_name} missing required columns: {missing_display}. "
        f"Present columns include: {sample_display}."
    )


def assert_schedule_contract(df: pd.DataFrame) -> None:
    """Validate that schedule dataframes include critical columns.

    Parameters
    ----------
    df:
        Schedule dataframe returned by :func:`nflreadpy.load_schedules`.

    Raises
    ------
    ValueError
        If any of the minimum viable schedule columns are missing.
    """

    _assert_columns(df, SCHEDULE_REQUIRED_COLUMNS, "Schedule dataset")


def assert_pbp_contract(df: pd.DataFrame) -> None:
    """Validate that play-by-play dataframes include critical columns."""

    _assert_columns(df, PBP_REQUIRED_COLUMNS, "Play-by-play dataset")


def assert_roster_contract(df: pd.DataFrame) -> None:
    """Validate that roster dataframes include critical columns."""

    _assert_columns(df, ROSTER_REQUIRED_COLUMNS, "Roster dataset")


def assert_team_contract(df: pd.DataFrame) -> None:
    """Validate that team metadata dataframes include critical columns."""

    _assert_columns(df, TEAM_REQUIRED_COLUMNS, "Team metadata dataset")


__all__ = [
    "SCHEDULE_REQUIRED_COLUMNS",
    "PBP_REQUIRED_COLUMNS",
    "ROSTER_REQUIRED_COLUMNS",
    "TEAM_REQUIRED_COLUMNS",
    "assert_schedule_contract",
    "assert_pbp_contract",
    "assert_roster_contract",
    "assert_team_contract",
]

