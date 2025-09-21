"""Utilities for deriving picks and confidence tiers from win probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

STRONG_THRESHOLD: Final[float] = 0.65
LEAN_THRESHOLD: Final[float] = 0.55


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Configuration container for pick confidence tiers."""

    strong: float = STRONG_THRESHOLD
    lean: float = LEAN_THRESHOLD

    def __post_init__(self) -> None:
        if not np.isfinite(self.strong) or not np.isfinite(self.lean):
            raise ValueError("Confidence thresholds must be finite numeric values.")
        if self.strong < self.lean:
            raise ValueError("Strong threshold must be greater than or equal to the lean threshold.")
        if self.lean <= 0:
            raise ValueError("Lean threshold must be positive.")


def assign_pick_confidence(
    frame: pd.DataFrame,
    *,
    home_column: str = "p_home_win",
    away_column: str = "p_away_win",
    thresholds: ConfidenceThresholds | None = None,
    prefer_home_on_tie: bool = True,
) -> pd.DataFrame:
    """Return a new DataFrame with pick and confidence tier columns.

    Parameters
    ----------
    frame:
        DataFrame containing home/away win probabilities for each game.
    home_column, away_column:
        Column names holding the win probabilities for the home and away teams.
    thresholds:
        Optional :class:`ConfidenceThresholds` to customize Strong/Lean cutoffs.
    prefer_home_on_tie:
        When ``True`` (default) ties are resolved in favour of the home team; when
        ``False`` ties favour the away team.

    Returns
    -------
    pandas.DataFrame
        Copy of ``frame`` with two new columns:

        ``pick``
            Either ``"home"`` or ``"away"`` indicating the selected side.
        ``confidence``
            One of ``"Strong"``, ``"Lean"`` or ``"Pass"`` describing the pick tier.

    Notes
    -----
    Probabilities that are missing or non-finite default to 0.5 so that the game is
    treated as a coin flip. Ties are broken deterministically according to
    ``prefer_home_on_tie`` to satisfy the repository's repeatability requirement.
    """

    if thresholds is None:
        thresholds = ConfidenceThresholds()

    missing_columns = {home_column, away_column} - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Frame missing required probability columns: {missing}.")

    result = frame.copy()

    home_probs = pd.to_numeric(result[home_column], errors="coerce").to_numpy(dtype=float)
    away_probs = pd.to_numeric(result[away_column], errors="coerce").to_numpy(dtype=float)

    home_probs = np.nan_to_num(home_probs, nan=0.5, posinf=0.5, neginf=0.5)
    away_probs = np.nan_to_num(away_probs, nan=0.5, posinf=0.5, neginf=0.5)

    if prefer_home_on_tie:
        pick = np.where(home_probs >= away_probs, "home", "away")
    else:
        pick = np.where(home_probs > away_probs, "home", "away")

    max_probs = np.maximum(home_probs, away_probs)

    confidence = np.full_like(max_probs, fill_value="Pass", dtype=object)
    confidence[max_probs >= thresholds.lean] = "Lean"
    confidence[max_probs >= thresholds.strong] = "Strong"

    result["pick"] = pick
    result["confidence"] = confidence

    return result


__all__ = ["ConfidenceThresholds", "assign_pick_confidence"]
