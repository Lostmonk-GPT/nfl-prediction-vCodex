"""Week-level visibility helpers for MVP feature builds.

This module implements a coarse snapshot approximation that treats the end of
an NFL week as the latest scheduled kickoff for that week. The utilities here
allow callers to derive an ``asof_ts`` timestamp for a ``(season, week)`` pair
and to filter arbitrary dataframes so that only rows visible at that instant are
retained.

The approximation is intentionally simple for the MVP: we assume that all
required inputs (play-by-play, schedule, travel metadata) are either timestamped
at the play level or keyed by ``season``/``week``. Injuries, inactives, and
other intra-week updates are not modelled; rows without event timestamps fall
back to season/week filtering. Future phases will replace this proxy with
granular snapshot captures aligned to the project runbook.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["compute_week_asof", "filter_visible_rows"]


def compute_week_asof(
    schedule: pd.DataFrame,
    *,
    season: int,
    week: int,
    kickoff_column: str = "start_time",
) -> pd.Timestamp | None:
    """Return the latest scheduled kickoff for ``(season, week)``.

    Args:
        schedule: Schedule dataframe containing at least ``season`` and
            ``week`` columns. Kickoff information is read from
            ``kickoff_column`` when present.
        season: Season identifier for the requested week.
        week: Week number for the requested season.
        kickoff_column: Column containing the scheduled kickoff timestamps.
            Defaults to ``"start_time"`` which matches the ingestion contract.

    Returns:
        The maximum kickoff timestamp for the requested week as a timezone-aware
        ``pandas.Timestamp`` in UTC. ``None`` is returned when no valid kickoff
        timestamps are available (e.g., missing column or all null values).

    Raises:
        ValueError: When ``schedule`` is empty or no rows match the requested
            ``season``/``week`` combination.
    """

    if schedule.empty:
        raise ValueError("Schedule dataframe is empty; cannot compute asof timestamp.")

    required_columns = {"season", "week"}
    missing = sorted(required_columns - set(schedule.columns))
    if missing:
        raise KeyError(f"Schedule dataframe missing required columns: {missing}")

    working = schedule.copy()
    working["season"] = working["season"].astype(int)
    working["week"] = working["week"].astype(int)

    mask = (working["season"] == season) & (working["week"] == week)
    week_rows = working.loc[mask]
    if week_rows.empty:
        raise ValueError(f"No schedule rows found for season {season}, week {week}.")

    if kickoff_column not in week_rows.columns:
        return None

    kickoff_times = pd.to_datetime(week_rows[kickoff_column], utc=True, errors="coerce")
    if kickoff_times.notna().any():
        return kickoff_times.max()

    return None


def filter_visible_rows(
    df: pd.DataFrame,
    *,
    season: int,
    week: int,
    asof_ts: pd.Timestamp | None,
    event_time_col: str = "event_time",
    season_col: str = "season",
    week_col: str = "week",
) -> pd.DataFrame:
    """Filter ``df`` to rows visible at ``asof_ts``.

    The helper prefers precise timestamp-based filtering when ``event_time_col``
    contains valid timestamps. Rows lacking event timestamps fall back to a
    coarse ``season``/``week`` filter that retains data up to and including the
    target week.

    Args:
        df: Source dataframe to filter.
        season: Target season for the visibility window.
        week: Target week (inclusive) for the visibility window.
        asof_ts: Cutoff timestamp. When ``None`` the function degrades to
            season/week filtering.
        event_time_col: Column holding event timestamps for precise filtering.
        season_col: Column representing the season identifiers.
        week_col: Column representing the week numbers.

    Returns:
        A dataframe containing only rows visible at the requested snapshot.

    Raises:
        KeyError: When fallback filtering is required but ``season_col`` or
            ``week_col`` is missing from ``df``.
    """

    if df.empty:
        return df.copy()

    working = df.copy()
    frames: list[pd.DataFrame] = []
    fallback_source: pd.DataFrame | None = working

    if asof_ts is not None and event_time_col in working.columns:
        asof_utc = _ensure_utc_timestamp(asof_ts)
        event_times = pd.to_datetime(working[event_time_col], utc=True, errors="coerce")
        valid_event_mask = event_times.notna()

        if valid_event_mask.any():
            visible_event = working.loc[valid_event_mask & (event_times <= asof_utc)].copy()
            if not visible_event.empty:
                frames.append(visible_event)

            # Only rows without usable event timestamps require fallback logic.
            fallback_source = working.loc[~valid_event_mask].copy()
        else:
            fallback_source = working
    else:
        fallback_source = working

    if fallback_source is not None and not fallback_source.empty:
        frames.append(
            _filter_by_week(
                fallback_source,
                season=season,
                week=week,
                season_col=season_col,
                week_col=week_col,
            )
        )

    if not frames:
        return working.iloc[0:0].copy()

    combined = pd.concat(frames).sort_index()
    return combined


def _filter_by_week(
    df: pd.DataFrame,
    *,
    season: int,
    week: int,
    season_col: str,
    week_col: str,
) -> pd.DataFrame:
    if season_col not in df.columns or week_col not in df.columns:
        raise KeyError(
            "Dataframe missing season/week columns required for fallback filtering."
        )

    filtered = df.copy()
    filtered[season_col] = filtered[season_col].astype(int)
    filtered[week_col] = filtered[week_col].astype(int)

    mask = (filtered[season_col] < season) | (
        (filtered[season_col] == season) & (filtered[week_col] <= week)
    )

    return filtered.loc[mask]


def _ensure_utc_timestamp(value: Any) -> pd.Timestamp:
    """Return ``value`` as a UTC normalised ``Timestamp``."""

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Cannot normalise NaT/NaN timestamp to UTC.")

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")
