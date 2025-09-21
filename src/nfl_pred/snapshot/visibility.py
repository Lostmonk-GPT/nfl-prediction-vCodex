"""Snapshot visibility utilities for feature builders.

The helpers defined here centralise the enforcement of ``asof_ts`` cut-offs
across the various feature builders. Each function accepts a pandas
``DataFrame`` and returns a filtered copy where events occurring strictly after
``asof_ts`` have been removed. When precise event timestamps are unavailable the
helpers degrade gracefully by falling back to coarser season/week filters so we
avoid introducing silent leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


__all__ = [
    "VisibilityContext",
    "filter_play_by_play",
    "filter_schedule",
    "filter_weekly_frame",
]


@dataclass(frozen=True)
class VisibilityContext:
    """Container describing the snapshot cut-off for a build."""

    season: int | None = None
    week: int | None = None
    asof_ts: pd.Timestamp | None = None


def filter_play_by_play(
    pbp: pd.DataFrame,
    *,
    context: VisibilityContext,
    event_time_col: str = "event_time",
    ingestion_time_col: str = "asof_ts",
) -> pd.DataFrame:
    """Return play-by-play rows visible at ``context.asof_ts``.

    The function first enforces a hard cut on ``event_time_col``. Rows with a
    valid event timestamp strictly greater than ``asof_ts`` are dropped even if
    ingestion timestamps indicate earlier availability. For rows lacking precise
    event timestamps we fall back to ``ingestion_time_col`` when present. If
    neither timestamp exists we further degrade to season/week filtering when
    the context provides those values.
    """

    if pbp.empty:
        return pbp.copy()

    working = pbp.copy()
    cutoff = _ensure_optional_cutoff(context.asof_ts)

    mask = pd.Series(True, index=working.index, dtype=bool)
    fallback_candidates = pd.Series(True, index=working.index, dtype=bool)

    if cutoff is not None and event_time_col in working.columns:
        event_times = pd.to_datetime(working[event_time_col], utc=True, errors="coerce")
        event_known = event_times.notna()
        mask &= (~event_known) | (event_times <= cutoff)
        fallback_candidates = fallback_candidates & (~event_known)

    if cutoff is not None and ingestion_time_col in working.columns:
        ingestion_times = pd.to_datetime(working[ingestion_time_col], utc=True, errors="coerce")
        ingestion_known = ingestion_times.notna()
        mask &= (~fallback_candidates) | (~ingestion_known) | (ingestion_times <= cutoff)
        fallback_candidates = fallback_candidates & (~ingestion_known)

    if cutoff is not None:
        working = working.loc[mask].copy()

    if context.season is not None and context.week is not None:
        working = filter_weekly_frame(
            working,
            season=context.season,
            week=context.week,
            season_column="season",
            week_column="week",
        )

    return working


def filter_schedule(
    schedule: pd.DataFrame,
    *,
    context: VisibilityContext,
    kickoff_column: str = "start_time",
) -> pd.DataFrame:
    """Return schedule rows visible at ``context.asof_ts``."""

    if schedule.empty:
        return schedule.copy()

    working = schedule.copy()
    cutoff = _ensure_optional_cutoff(context.asof_ts)

    if kickoff_column in working.columns and cutoff is not None:
        kickoff = pd.to_datetime(working[kickoff_column], utc=True, errors="coerce")
        mask = kickoff.isna() | (kickoff <= cutoff)
        working = working.loc[mask].copy()

    if context.season is not None and context.week is not None:
        working = filter_weekly_frame(
            working,
            season=context.season,
            week=context.week,
            season_column="season",
            week_column="week",
        )

    return working


def filter_weekly_frame(
    df: pd.DataFrame,
    *,
    season: int,
    week: int,
    season_column: str = "season",
    week_column: str = "week",
    columns_required: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Restrict a frame to rows with ``season/week`` up to the target week."""

    if df.empty:
        return df.copy()

    if columns_required is not None:
        missing = sorted(set(columns_required) - set(df.columns))
        if missing:
            raise KeyError(f"Dataframe missing required columns for visibility filtering: {missing}")

    if season_column not in df.columns or week_column not in df.columns:
        raise KeyError("Dataframe must contain season/week columns for fallback visibility filtering.")

    filtered = df.copy()
    filtered[season_column] = filtered[season_column].astype(int)
    filtered[week_column] = filtered[week_column].astype(int)

    mask = (filtered[season_column] < season) | (
        (filtered[season_column] == season) & (filtered[week_column] <= week)
    )

    return filtered.loc[mask].copy()


def _ensure_optional_cutoff(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
