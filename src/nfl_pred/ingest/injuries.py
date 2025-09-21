"""Injury report ingestion via ``nflreadpy``.

This module mirrors the other ingestion utilities in the project by fetching
raw data from ``nflreadpy``, attaching minimal ingestion metadata, persisting
to Parquet, and optionally registering the output with DuckDB for
downstream access. The injury feed includes a ``date_modified`` field which
is surfaced as ``event_time`` to support future visibility filtering.
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
_INJURIES_FILENAME = "injuries.parquet"
_DUCKDB_VIEW = "injuries_raw"
_SOURCE_NAME = "nflreadpy"


def ingest_injuries(seasons: list[int]) -> Path:
    """Pull injuries for ``seasons`` and persist them to a Parquet file."""

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
        LOGGER.info("Loading injuries for season %s via nflreadpy", season)
        season_df = _load_injuries(season)
        if season_df.empty:
            LOGGER.warning("No injury rows returned for season %s", season)
            continue
        frames.append(season_df)

    if not frames:
        raise RuntimeError("No injury data was retrieved for the requested seasons.")

    combined = pd.concat(frames, ignore_index=True)
    combined["event_time"] = pd.to_datetime(
        combined.get("date_modified"), utc=True, errors="coerce"
    )
    combined["pulled_at"] = pulled_at
    combined["source"] = _SOURCE_NAME
    combined["source_version"] = source_version

    output_path = raw_dir / _INJURIES_FILENAME
    combined.to_parquet(output_path, index=False)

    LOGGER.info(
        "Wrote injuries to %s with shape %s and columns %s",
        output_path,
        combined.shape,
        list(combined.columns),
    )

    _register_with_duckdb(output_path, config.paths.duckdb_path)

    return output_path


def _load_injuries(season: int) -> pd.DataFrame:
    """Load a single season of injury data as a pandas ``DataFrame``."""

    polars_df = nflreadpy.load_injuries(season)
    pdf = polars_df.to_pandas(use_pyarrow_extension_array=True)
    return pdf


def _register_with_duckdb(parquet_path: Path, duckdb_path: str) -> None:
    """Register the Parquet file as a DuckDB view for downstream consumption."""

    try:
        with DuckDBClient(duckdb_path) as client:
            client.register_parquet(str(parquet_path), _DUCKDB_VIEW)
            LOGGER.info("Registered DuckDB view '%s' for %s", _DUCKDB_VIEW, parquet_path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to register DuckDB view '%s': %s", _DUCKDB_VIEW, exc)


__all__ = ["ingest_injuries"]

