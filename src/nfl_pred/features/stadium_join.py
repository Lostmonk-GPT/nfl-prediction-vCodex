"""Join utilities for enriching schedules with stadium metadata."""

from __future__ import annotations

from typing import Final

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

_REQUIRED_SCHEDULE_COLUMNS: Final[set[str]] = {
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
}

def join_stadium_metadata(schedule: pd.DataFrame, stadiums: pd.DataFrame) -> pd.DataFrame:
    """Join authoritative stadium information onto a schedule frame.

    The returned frame contains one row per game with the authoritative roof,
    surface, time zone, coordinates, and neutral-site designation. When the
    schedule provides conflicting values for roof or surface, the authoritative
    values take precedence and a warning is logged summarising the mismatch
    counts.

    Args:
        schedule: NFL schedule data including at least the required columns.
        stadiums: Authoritative stadium reference as produced by
            :func:`nfl_pred.ref.stadiums.load_stadiums`.

    Returns:
        ``pandas.DataFrame`` keyed by ``season``, ``week``, and ``game_id`` with
        authoritative stadium attributes.
    """

    _validate_schedule(schedule)
    if stadiums.empty:
        raise ValueError("Authoritative stadium table is empty.")

    working = schedule.copy()
    working["venue"] = _extract_venue(working)

    # Normalise columns used for mismatch reporting.
    schedule_roof = working.get("roof")
    schedule_surface = working.get("surface")
    schedule_neutral = _extract_schedule_neutral(working)

    authoritative = _prepare_authoritative_table(stadiums)
    merged = working.merge(
        authoritative,
        how="left",
        left_on=["venue_norm", "home_team_norm"],
        right_on=["venue_norm", "team"],
        suffixes=("", "_auth"),
    )

    unresolved = merged["venue"].notna() & merged["roof_auth"].isna()
    if unresolved.any():
        count = int(unresolved.sum())
        examples = ", ".join(merged.loc[unresolved, "game_id"].head(3))
        LOGGER.warning(
            "Missing authoritative stadium rows for %s games (examples: %s)",
            count,
            examples,
        )

    result = merged[["season", "week", "game_id", "venue"]].copy()

    schedule_roof_aligned = None if schedule_roof is None else schedule_roof.reset_index(drop=True)
    if schedule_roof_aligned is not None:
        result["roof"] = merged["roof_auth"].where(
            merged["roof_auth"].notna(),
            schedule_roof_aligned,
        )
    else:
        result["roof"] = merged["roof_auth"]

    schedule_surface_aligned = None if schedule_surface is None else schedule_surface.reset_index(drop=True)
    if schedule_surface_aligned is not None:
        result["surface"] = merged["surface_auth"].where(
            merged["surface_auth"].notna(),
            schedule_surface_aligned,
        )
    else:
        result["surface"] = merged["surface_auth"]

    result["tz"] = merged.get("tz")
    result["lat"] = merged.get("lat")
    result["lon"] = merged.get("lon")

    schedule_neutral_aligned = schedule_neutral.reset_index(drop=True)
    result["neutral_site"] = merged["neutral_site_auth"].where(
        merged["neutral_site_auth"].notna(),
        schedule_neutral_aligned,
    )

    _log_mismatches(
        merged,
        schedule_roof=schedule_roof_aligned,
        schedule_surface=schedule_surface_aligned,
        schedule_neutral=schedule_neutral_aligned,
    )

    result = result.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return result


def _validate_schedule(schedule: pd.DataFrame) -> None:
    missing = sorted(column for column in _REQUIRED_SCHEDULE_COLUMNS if column not in schedule.columns)
    if missing:
        raise KeyError(f"Schedule frame missing required columns: {missing}")


def _extract_venue(schedule: pd.DataFrame) -> pd.Series:
    if "stadium" in schedule.columns:
        venue = schedule["stadium"].astype(str)
    elif "venue" in schedule.columns:
        venue = schedule["venue"].astype(str)
    else:
        raise KeyError("Schedule frame must include either 'stadium' or 'venue' column.")

    venue_norm = venue.str.strip()
    schedule["venue_norm"] = venue_norm.str.lower()
    schedule["home_team_norm"] = schedule["home_team"].astype(str).str.strip().str.upper()
    return venue_norm


def _extract_schedule_neutral(schedule: pd.DataFrame) -> pd.Series:
    if "neutral_site" in schedule.columns:
        neutral_raw = schedule["neutral_site"]
        if pd.api.types.is_bool_dtype(neutral_raw):
            return neutral_raw.fillna(False)
        if pd.api.types.is_numeric_dtype(neutral_raw):
            return neutral_raw.fillna(0).astype(int).astype(bool)
        return neutral_raw.astype(str).str.lower().isin({"true", "t", "yes", "y", "1"})

    if "location" in schedule.columns:
        return schedule["location"].astype(str).str.strip().str.lower().eq("neutral")

    return pd.Series(False, index=schedule.index, dtype=bool)


def _prepare_authoritative_table(stadiums: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "venue",
        "teams",
        "roof",
        "surface",
        "tz",
        "lat",
        "lon",
        "neutral_site",
    }
    missing = sorted(column for column in required_columns if column not in stadiums.columns)
    if missing:
        raise KeyError(f"Stadium reference missing columns: {missing}")

    exploded = stadiums.copy()
    exploded["venue_norm"] = exploded["venue"].astype(str).str.strip().str.lower()
    exploded = exploded.explode("teams")
    exploded["team"] = exploded["teams"].astype(str).str.strip().str.upper()
    exploded = exploded.drop(columns=["teams"])
    exploded = exploded.rename(
        columns={
            "roof": "roof_auth",
            "surface": "surface_auth",
            "neutral_site": "neutral_site_auth",
        }
    )
    return exploded


def _log_mismatches(
    merged: pd.DataFrame,
    *,
    schedule_roof: pd.Series | None,
    schedule_surface: pd.Series | None,
    schedule_neutral: pd.Series,
) -> None:
    if schedule_roof is not None:
        mismatch_roof = _series_mismatch(schedule_roof, merged["roof_auth"])
        if mismatch_roof.any():
            _log_warning("roof", mismatch_roof, merged)

    if schedule_surface is not None:
        mismatch_surface = _series_mismatch(schedule_surface, merged["surface_auth"])
        if mismatch_surface.any():
            _log_warning("surface", mismatch_surface, merged)

    mismatch_neutral = _series_mismatch(schedule_neutral, merged["neutral_site_auth"])
    if mismatch_neutral.any():
        _log_warning("neutral_site", mismatch_neutral, merged)


def _series_mismatch(schedule_values: pd.Series | None, authoritative_values: pd.Series) -> pd.Series:
    if schedule_values is None:
        return pd.Series(False, index=authoritative_values.index)

    sched = schedule_values.astype(str).str.strip().str.lower()
    auth = authoritative_values.astype(str).str.strip().str.lower()
    mask = authoritative_values.notna() & schedule_values.notna() & (sched != auth)
    return mask


def _log_warning(field: str, mask: pd.Series, merged: pd.DataFrame) -> None:
    count = int(mask.sum())
    examples = ", ".join(merged.loc[mask, "game_id"].head(3))
    LOGGER.warning(
        "Schedule %s differs from authoritative data for %s games (examples: %s).", field, count, examples
    )
