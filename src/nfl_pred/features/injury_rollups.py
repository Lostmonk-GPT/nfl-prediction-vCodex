"""Utilities for aggregating injury report practice statuses by position group."""

from __future__ import annotations

from typing import Final

import pandas as pd


# Canonical mapping of roster positions to the nine position groups we expose as features.
# The mapping intentionally covers common abbreviations for both offensive and defensive roles.
_POSITION_TO_GROUP: Final[dict[str, str]] = {
    # Offense
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "OL": "OL",
    "LT": "OL",
    "RT": "OL",
    "LG": "OL",
    "RG": "OL",
    "C": "OL",
    "G": "OL",
    "T": "OL",
    # Defensive front
    "DL": "DL",
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "EDGE": "DL",
    # Linebackers
    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    # Secondary
    "DB": "DB",
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "NB": "DB",
    # Specialists
    "K": "ST",
    "P": "ST",
    "LS": "ST",
    "KR": "ST",
    "PR": "ST",
    "KOS": "ST",
}

_VALID_STATUSES: Final[frozenset[str]] = frozenset({"DNP", "LP", "FP"})
_STATUS_NORMALIZATION: Final[dict[str, str]] = {
    "DID NOT PRACTICE": "DNP",
    "DID NOT PARTICIPATE": "DNP",
    "LIMITED PARTICIPATION": "LP",
    "LIMITED PARTICIPATION IN PRACTICE": "LP",
    "LIMITED PRACTICE": "LP",
    "FULL PARTICIPATION": "FP",
    "FULL PARTICIPATION IN PRACTICE": "FP",
    "FULL PRACTICE": "FP",
}
_REQUIRED_COLUMNS: Final[set[str]] = {"season", "week", "team", "position", "practice_status"}


def _normalize_position(position: object) -> str | None:
    """Return the canonical position group for ``position`` or ``None`` when unknown."""

    if position is None:
        return None
    if isinstance(position, str):
        key = position.strip().upper()
    else:  # pragma: no cover - defensive guard for unexpected types
        key = str(position).strip().upper()
    return _POSITION_TO_GROUP.get(key)


def _normalize_status(status: object) -> str | None:
    """Normalize raw practice status strings to ``DNP``/``LP``/``FP`` codes."""

    if status is None:
        return None
    if isinstance(status, str):
        key = status.strip().upper()
    else:  # pragma: no cover - defensive guard
        key = str(status).strip().upper()
    key = _STATUS_NORMALIZATION.get(key, key)
    if key in _VALID_STATUSES:
        return key
    return None


def build_injury_rollups(injuries: pd.DataFrame) -> pd.DataFrame:
    """Aggregate practice participation counts by team, week, and position group.

    Parameters
    ----------
    injuries:
        DataFrame containing injury report rows with at least ``season``, ``week``,
        ``team``, ``position``, and ``practice_status`` columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with counts of DNP/LP/FP for each position group by team and
        week. Entries without a recognized position group or practice status are
        excluded, yielding zero counts for those scenarios.

    Notes
    -----
    Rows missing either a usable position mapping or practice status code are
    dropped rather than imputed. This keeps the resulting features faithful to
    the available injury intelligence for the snapshot.
    """

    missing = _REQUIRED_COLUMNS.difference(injuries.columns)
    if missing:
        raise ValueError(f"Injuries frame is missing required columns: {sorted(missing)}")

    if injuries.empty:
        empty = pd.DataFrame(columns=["season", "week", "team", "position_group", "dnp", "lp", "fp"])
        return empty.astype({"dnp": "int64", "lp": "int64", "fp": "int64"})

    normalized = injuries.copy()
    normalized["position_group"] = normalized["position"].map(_normalize_position)
    normalized["status_code"] = normalized["practice_status"].map(_normalize_status)

    filtered = normalized.dropna(subset=["position_group", "status_code", "season", "week", "team"])
    if filtered.empty:
        empty = pd.DataFrame(columns=["season", "week", "team", "position_group", "dnp", "lp", "fp"])
        return empty.astype({"dnp": "int64", "lp": "int64", "fp": "int64"})

    counts = (
        filtered.assign(count=1)
        .groupby(["season", "week", "team", "position_group", "status_code"], dropna=False)["count"]
        .sum()
        .unstack(fill_value=0)
    )

    for status in _VALID_STATUSES:
        if status not in counts.columns:
            counts[status] = 0

    counts = counts.reset_index().rename(columns={"DNP": "dnp", "LP": "lp", "FP": "fp"})
    counts = counts[["season", "week", "team", "position_group", "dnp", "lp", "fp"]]
    return counts.astype({"dnp": "int64", "lp": "int64", "fp": "int64"})


__all__ = ["build_injury_rollups"]
