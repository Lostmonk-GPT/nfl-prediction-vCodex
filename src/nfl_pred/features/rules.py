"""Rule-change guard features for the MVP feature matrix.

This module centralises the logic that tags rows in the feature matrix when
league rule changes take effect. The current scope covers two binary switches
requested in the PRD:

* ``kickoff_2024plus`` – flags seasons impacted by the 2024 kickoff overhaul.
* ``ot_regular_2025plus`` – flags regular-season games from 2025 onward when
  the updated overtime format is in place.

The helpers operate on per-team schedule metadata (``season``, ``week``,
``game_id``, ``team``) and avoid mutating the input frame. They intentionally
use the season identifier as the primary activation key so postseason games
played in the following calendar year continue to inherit the rule state of the
season they belong to. Regular-season detection is handled via the numeric week
value with the assumption that standard nflverse schedules encode Weeks 1–18 as
the regular season. The behaviour is documented so future playoff-specific
logic can refine the boundary when postseason handling lands in [AI-502].
"""

from __future__ import annotations

from typing import Final

import pandas as pd

_MERGE_KEYS: Final[list[str]] = ["season", "week", "game_id", "team"]
_KICKOFF_RULE_SEASON: Final[int] = 2024
_OT_RULE_SEASON: Final[int] = 2025
_REGULAR_SEASON_MAX_WEEK: Final[int] = 18


def compute_rule_flags(
    schedule_meta: pd.DataFrame,
    *,
    kickoff_cutover_season: int = _KICKOFF_RULE_SEASON,
    ot_regular_cutover_season: int = _OT_RULE_SEASON,
    regular_season_max_week: int = _REGULAR_SEASON_MAX_WEEK,
) -> pd.DataFrame:
    """Return per-team rule flag indicators for the supplied schedule frame.

    Args:
        schedule_meta: Per-team schedule metadata frame containing at least the
            merge keys defined in :data:`_MERGE_KEYS`. The function does not
            mutate the input and only inspects the key columns.
        kickoff_cutover_season: Season (inclusive) when the reimagined kickoff
            rules are considered active. Defaults to ``2024`` per PRD guidance.
        ot_regular_cutover_season: Season (inclusive) when the updated
            regular-season overtime procedure activates. Defaults to ``2025``.
        regular_season_max_week: Highest week number treated as regular season
            when deriving the overtime flag. Defaults to ``18`` which aligns
            with the current NFL schedule format.

    Returns:
        ``DataFrame`` with the merge keys and two boolean columns named
        ``kickoff_2024plus`` and ``ot_regular_2025plus``.

    Boundary notes:
        * ``kickoff_2024plus`` flips to ``True`` for every row whose season is
          greater than or equal to the configured cutoff, including postseason
          games assigned to that season.
        * ``ot_regular_2025plus`` flips to ``True`` for rows at or beyond the
          overtime cutoff season **and** whose numeric week is within the
          regular-season range. Postseason rows therefore stay ``False`` until a
          dedicated postseason flag is introduced in [AI-502].
    """

    missing = [column for column in _MERGE_KEYS if column not in schedule_meta.columns]
    if missing:
        raise KeyError(f"Schedule frame missing required columns for rule flags: {missing}")

    working = schedule_meta[_MERGE_KEYS].copy()

    season_numeric = pd.to_numeric(working["season"], errors="coerce")
    week_numeric = pd.to_numeric(working["week"], errors="coerce")

    kickoff_flag = season_numeric.ge(kickoff_cutover_season)
    kickoff_flag = kickoff_flag.fillna(False)

    ot_flag = season_numeric.ge(ot_regular_cutover_season)
    regular_season_mask = week_numeric.le(regular_season_max_week)
    ot_flag = ot_flag & regular_season_mask.fillna(False)

    working["kickoff_2024plus"] = kickoff_flag.astype(bool)
    working["ot_regular_2025plus"] = ot_flag.astype(bool)

    return working


def append_rule_flags(
    features: pd.DataFrame,
    *,
    kickoff_cutover_season: int = _KICKOFF_RULE_SEASON,
    ot_regular_cutover_season: int = _OT_RULE_SEASON,
    regular_season_max_week: int = _REGULAR_SEASON_MAX_WEEK,
) -> pd.DataFrame:
    """Return a copy of ``features`` with rule flag columns appended.

    The helper merges the output of :func:`compute_rule_flags` back into the
    provided frame. Existing columns remain untouched and the shape of the
    returned frame matches the input.
    """

    if features.empty:
        empty = features.copy()
        empty["kickoff_2024plus"] = pd.Series(dtype=bool)
        empty["ot_regular_2025plus"] = pd.Series(dtype=bool)
        return empty

    rule_flags = compute_rule_flags(
        features,
        kickoff_cutover_season=kickoff_cutover_season,
        ot_regular_cutover_season=ot_regular_cutover_season,
        regular_season_max_week=regular_season_max_week,
    )

    merged = features.merge(rule_flags, on=_MERGE_KEYS, how="left", validate="one_to_one")
    return merged


__all__ = ["append_rule_flags", "compute_rule_flags"]

