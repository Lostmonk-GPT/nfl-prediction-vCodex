"""Playoff handling utilities for feature assembly.

This module exposes helpers that tag schedule-derived feature rows with
postseason indicators. The functions operate on the raw schedule frame used by
the MVP feature builder and return per-team rows that can be merged with other
feature components. The implementation avoids mutating the input frames and
defaults to the regular-season-only behaviour required by the MVP until a
configuration toggle opts into postseason data.
"""

from __future__ import annotations

from typing import Final, Iterable

import numpy as np
import pandas as pd

_MERGE_KEYS: Final[list[str]] = ["season", "week", "game_id", "team"]
_REQUIRED_SCHEDULE_COLUMNS: Final[set[str]] = {
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
}
_DEFAULT_POSTSEASON_GAME_TYPES: Final[frozenset[str]] = frozenset({"POST", "SB"})


def compute_playoff_flags(
    schedule: pd.DataFrame,
    *,
    postseason_game_types: Iterable[str] = _DEFAULT_POSTSEASON_GAME_TYPES,
    fallback_week_threshold: int | None = 18,
) -> pd.DataFrame:
    """Return per-team postseason flags for the provided schedule frame.

    Args:
        schedule: Raw schedule frame with at least the columns in
            :data:`_REQUIRED_SCHEDULE_COLUMNS`.
        postseason_game_types: Game type labels that should be treated as
            postseason contests when the schedule includes a ``game_type``
            column. Values are upper-cased before comparison. Defaults to the
            nflverse ``POST`` convention and the Super Bowl ``SB`` label.
        fallback_week_threshold: Numeric week strictly above which rows are
            considered postseason when ``game_type`` is missing. ``None``
            disables the fallback.

    Returns:
        ``DataFrame`` with columns ``season``, ``week``, ``game_id``, ``team``,
        ``is_postseason`` (boolean), and ``season_phase`` (string).

    Raises:
        KeyError: When required schedule columns are missing.
    """

    missing = sorted(column for column in _REQUIRED_SCHEDULE_COLUMNS if column not in schedule.columns)
    if missing:
        raise KeyError(f"Schedule frame missing required columns for playoff flags: {missing}")

    if schedule.empty:
        return pd.DataFrame(columns=_MERGE_KEYS + ["is_postseason", "season_phase"])

    working = schedule.copy()

    season_numeric = pd.to_numeric(working["season"], errors="coerce")
    week_numeric = pd.to_numeric(working["week"], errors="coerce")
    valid_mask = season_numeric.notna() & week_numeric.notna()
    if not valid_mask.all():
        working = working.loc[valid_mask].copy()
        season_numeric = season_numeric.loc[valid_mask]
        week_numeric = week_numeric.loc[valid_mask]

    working["season"] = season_numeric.astype(int)
    working["week"] = week_numeric.astype(int)
    working["game_id"] = working["game_id"].astype(str)

    postseason_types = {str(value).strip().upper() for value in postseason_game_types}

    if "game_type" in working.columns and postseason_types:
        raw_types = working["game_type"]
        game_type = raw_types.astype(str).str.upper().str.strip()
        is_postseason = game_type.isin(postseason_types)
    else:
        is_postseason = pd.Series(False, index=working.index, dtype=bool)

    if fallback_week_threshold is not None:
        fallback_flag = week_numeric.gt(int(fallback_week_threshold)).fillna(False)
        is_postseason = is_postseason.fillna(False) | fallback_flag

    flags = is_postseason.fillna(False).astype(bool)
    working["is_postseason"] = flags

    base_columns = ["season", "week", "game_id", "is_postseason"]

    home = working[base_columns + ["home_team"]].copy()
    home.rename(columns={"home_team": "team"}, inplace=True)

    away = working[base_columns + ["away_team"]].copy()
    away.rename(columns={"away_team": "team"}, inplace=True)

    combined = pd.concat([home, away], ignore_index=True, sort=False)
    combined["team"] = combined["team"].astype(str)
    combined["season_phase"] = np.where(combined["is_postseason"], "postseason", "regular")

    combined = combined[_MERGE_KEYS + ["is_postseason", "season_phase"]]
    combined = combined.sort_values(_MERGE_KEYS).reset_index(drop=True)
    return combined


def append_playoff_flags(
    features: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    postseason_game_types: Iterable[str] = _DEFAULT_POSTSEASON_GAME_TYPES,
    fallback_week_threshold: int | None = 18,
) -> pd.DataFrame:
    """Return ``features`` with postseason indicators merged in.

    Args:
        features: Feature frame with at least the merge keys defined in
            :data:`_MERGE_KEYS`.
        schedule: Schedule frame used to derive postseason flags.
        postseason_game_types: Passed through to
            :func:`compute_playoff_flags`.
        fallback_week_threshold: Passed through to
            :func:`compute_playoff_flags`.

    Returns:
        Copy of ``features`` including ``is_postseason`` and ``season_phase``
        columns.
    """

    missing = [column for column in _MERGE_KEYS if column not in features.columns]
    if missing:
        raise KeyError(f"Feature frame missing required columns for playoff merge: {missing}")

    if features.empty:
        empty = features.copy()
        empty["is_postseason"] = pd.Series(dtype=bool)
        empty["season_phase"] = pd.Series(dtype="string")
        return empty

    playoff_flags = compute_playoff_flags(
        schedule,
        postseason_game_types=postseason_game_types,
        fallback_week_threshold=fallback_week_threshold,
    )

    if playoff_flags.empty:
        merged = features.copy()
        merged["is_postseason"] = False
        merged["season_phase"] = "regular"
        return merged

    merged = features.merge(
        playoff_flags,
        on=_MERGE_KEYS,
        how="left",
        validate="one_to_one",
    )

    merged["is_postseason"] = merged["is_postseason"].fillna(False).astype(bool)
    if "season_phase" in merged.columns:
        merged["season_phase"] = merged["season_phase"].fillna("regular").astype(str)
    else:
        merged["season_phase"] = np.where(merged["is_postseason"], "postseason", "regular")

    return merged


__all__ = ["append_playoff_flags", "compute_playoff_flags"]

