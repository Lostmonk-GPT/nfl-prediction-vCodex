"""Helpers for loading and validating stadium reference data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from nfl_pred.config import load_config

__all__ = ["StadiumsValidationError", "load_stadiums"]


DATA_SUBPATH = Path("ref") / "stadiums.csv"
REQUIRED_COLUMNS = (
    "venue",
    "teams",
    "lat",
    "lon",
    "tz",
    "altitude_ft",
    "surface",
    "roof",
    "neutral_site",
)
ALLOWED_SURFACES = {"natural_grass", "artificial_turf"}
ALLOWED_ROOF_TYPES = {"open", "dome", "retractable", "indoors"}


@dataclass(frozen=True)
class StadiumsValidationError(ValueError):
    """Raised when the stadium reference table fails validation."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - dataclass convenience
        return self.message


def load_stadiums(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the authoritative stadium table and validate its contents."""
    base_dir = Path(data_dir) if data_dir is not None else Path(load_config().paths.data_dir)
    csv_path = base_dir / DATA_SUBPATH

    if not csv_path.exists():
        raise StadiumsValidationError(f"Stadium reference file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_schema(df)

    df = df.copy()
    df["teams"] = df["teams"].apply(_parse_teams)
    df["neutral_site"] = df["neutral_site"].apply(_coerce_bool)

    _validate_rows(df)

    return df.sort_values("venue").reset_index(drop=True)


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise StadiumsValidationError(
            "Stadium reference is missing required columns: " + ", ".join(sorted(missing))
        )


def _parse_teams(raw: str) -> tuple[str, ...]:
    parts = [segment.strip() for segment in str(raw).split("|") if segment.strip()]
    if not parts:
        raise StadiumsValidationError("Stadium entry must include at least one team code.")
    return tuple(parts)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise StadiumsValidationError(f"Invalid neutral_site flag: {value}")


def _validate_rows(df: pd.DataFrame) -> None:
    for column in ("lat", "lon"):
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise StadiumsValidationError(f"Column '{column}' must be numeric.")

    if (df["lat"].abs() > 90).any():
        raise StadiumsValidationError("Latitude must be within [-90, 90].")
    if (df["lon"].abs() > 180).any():
        raise StadiumsValidationError("Longitude must be within [-180, 180].")

    if not pd.api.types.is_numeric_dtype(df["altitude_ft"]):
        raise StadiumsValidationError("Column 'altitude_ft' must be numeric.")

    if (df["altitude_ft"] < -100).any():
        raise StadiumsValidationError("Altitude appears invalid (below -100 ft).")

    invalid_surfaces = set(df["surface"].unique()) - ALLOWED_SURFACES
    if invalid_surfaces:
        raise StadiumsValidationError(
            "Unexpected surface types: " + ", ".join(sorted(invalid_surfaces))
        )

    invalid_roofs = set(df["roof"].unique()) - ALLOWED_ROOF_TYPES
    if invalid_roofs:
        raise StadiumsValidationError(
            "Unexpected roof types: " + ", ".join(sorted(invalid_roofs))
        )

    invalid_tz: list[str] = []
    for tz in df["tz"].unique():
        try:
            ZoneInfo(tz)
        except ZoneInfoNotFoundError:
            invalid_tz.append(tz)
    if invalid_tz:
        raise StadiumsValidationError(
            "Invalid time zone identifiers: " + ", ".join(sorted(invalid_tz))
        )

    duplicate_venues = df["venue"].duplicated().any()
    if duplicate_venues:
        raise StadiumsValidationError("Duplicate venue names detected in stadium reference.")

    duplicate_rows = df.duplicated(subset=["venue", "teams"]).any()
    if duplicate_rows:
        raise StadiumsValidationError("Duplicate venue/team combinations detected.")

    empty_roof = df["roof"].astype(str).str.strip() == ""
    if empty_roof.any():
        raise StadiumsValidationError("Roof type cannot be blank.")

    empty_surface = df["surface"].astype(str).str.strip() == ""
    if empty_surface.any():
        raise StadiumsValidationError("Surface cannot be blank.")

    for venue, teams in zip(df["venue"], df["teams"]):
        if any(len(team) == 0 for team in teams):
            raise StadiumsValidationError(f"Venue '{venue}' has an empty team code.")

