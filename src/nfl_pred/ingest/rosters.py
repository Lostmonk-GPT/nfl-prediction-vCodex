"""Roster and team ingestion via ``nflreadpy``.

This module pulls raw rosters for requested seasons as well as team metadata
from ``nflreadpy``. The returned structures are preserved with minimal
transformation: ingestion metadata columns are appended before each dataset is
persisted to Parquet and registered with DuckDB for downstream access.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

import nflreadpy
import pandas as pd

from nfl_pred.config import load_config
from nfl_pred.storage.duckdb_client import DuckDBClient


LOGGER = logging.getLogger(__name__)
_RAW_SUBDIR = "raw"
_ROSTERS_FILENAME = "rosters.parquet"
_TEAMS_FILENAME = "teams.parquet"
_ROSTERS_VIEW = "rosters_raw"
_TEAMS_VIEW = "teams_raw"
_SOURCE_NAME = "nflreadpy"


def ingest_rosters(seasons: list[int]) -> Path:
    """Pull rosters for ``seasons`` and persist them to a single Parquet file."""

    if not seasons:
        raise ValueError("'seasons' must contain at least one season to ingest.")

    config = load_config()
    data_dir = Path(config.paths.data_dir)
    raw_dir = data_dir / _RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    pulled_at = datetime.now(timezone.utc)
    source_version = getattr(nflreadpy, "__version__", None)

    frames: list[pd.DataFrame] = []
    for season in seasons:
        LOGGER.info("Loading roster for season %s via nflreadpy", season)
        season_df = _load_roster(season)
        if season_df.empty:
            LOGGER.warning("No roster rows returned for season %s", season)
            continue
        frames.append(season_df)

    if not frames:
        raise RuntimeError("No roster data was retrieved for the requested seasons.")

    combined = pd.concat(frames, ignore_index=True)
    combined["pulled_at"] = pulled_at
    combined["source"] = _SOURCE_NAME
    combined["source_version"] = source_version

    output_path = raw_dir / _ROSTERS_FILENAME
    combined.to_parquet(output_path, index=False)

    LOGGER.info(
        "Wrote rosters to %s with shape %s and columns %s",
        output_path,
        combined.shape,
        list(combined.columns),
    )

    _register_with_duckdb(output_path, config.paths.duckdb_path, _ROSTERS_VIEW)

    return output_path


def ingest_teams() -> Path:
    """Pull team metadata and persist it to Parquet."""

    config = load_config()
    data_dir = Path(config.paths.data_dir)
    raw_dir = data_dir / _RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    pulled_at = datetime.now(timezone.utc)
    source_version = getattr(nflreadpy, "__version__", None)

    LOGGER.info("Loading team metadata via nflreadpy")
    teams_df = _load_teams()
    if teams_df.empty:
        raise RuntimeError("No team metadata was retrieved from nflreadpy.")

    teams_df["pulled_at"] = pulled_at
    teams_df["source"] = _SOURCE_NAME
    teams_df["source_version"] = source_version

    output_path = raw_dir / _TEAMS_FILENAME
    teams_df.to_parquet(output_path, index=False)

    LOGGER.info(
        "Wrote teams to %s with shape %s and columns %s",
        output_path,
        teams_df.shape,
        list(teams_df.columns),
    )

    _register_with_duckdb(output_path, config.paths.duckdb_path, _TEAMS_VIEW)

    return output_path


def _load_roster(season: int) -> pd.DataFrame:
    """Load a single season roster as a pandas ``DataFrame``."""

    polars_df = nflreadpy.load_rosters(season)
    pdf = polars_df.to_pandas(use_pyarrow_extension_array=True)
    return pdf


def _load_teams() -> pd.DataFrame:
    """Load team metadata as a pandas ``DataFrame``."""

    polars_df = nflreadpy.load_teams()
    pdf = polars_df.to_pandas(use_pyarrow_extension_array=True)
    return pdf


def _register_with_duckdb(parquet_path: Path, duckdb_path: str, view_name: str) -> None:
    """Register the Parquet file as a DuckDB view for downstream consumption."""

    try:
        with DuckDBClient(duckdb_path) as client:
            client.register_parquet(str(parquet_path), view_name)
            LOGGER.info("Registered DuckDB view '%s' for %s", view_name, parquet_path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to register DuckDB view '%s': %s", view_name, exc)


__all__ = ["ingest_rosters", "ingest_teams"]
