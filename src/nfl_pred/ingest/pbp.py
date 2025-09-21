"""Play-by-play ingestion via ``nflreadpy``.

This module mirrors the schedule ingestion pattern but keeps each season in a
dedicated Parquet file. The raw structure returned by ``nflreadpy`` is
preserved; only ingestion metadata columns are appended before persistence.
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
_FILENAME_TEMPLATE = "pbp_{season}.parquet"
_DUCKDB_VIEW_TEMPLATE = "pbp_raw_{season}"
_SOURCE_NAME = "nflreadpy"


def ingest_pbp(seasons: list[int]) -> list[Path]:
    """Pull play-by-play for ``seasons`` and persist each season separately."""

    if not seasons:
        raise ValueError("'seasons' must contain at least one season to ingest.")

    config = load_config()
    data_dir = Path(config.paths.data_dir)
    raw_dir = data_dir / _RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    pulled_at = datetime.now(timezone.utc)
    source_version = getattr(nflreadpy, "__version__", None)

    written_paths: list[Path] = []
    for season in seasons:
        LOGGER.info("Loading play-by-play for season %s via nflreadpy", season)
        season_df = _load_pbp(season)
        if season_df.empty:
            LOGGER.warning("No play-by-play rows returned for season %s", season)
            continue

        season_df["pulled_at"] = pulled_at
        season_df["source"] = _SOURCE_NAME
        season_df["source_version"] = source_version

        output_path = raw_dir / _FILENAME_TEMPLATE.format(season=season)
        season_df.to_parquet(output_path, index=False)
        written_paths.append(output_path)

        LOGGER.info(
            "Wrote play-by-play for season %s to %s with shape %s",  # noqa: TRY400
            season,
            output_path,
            season_df.shape,
        )

        _register_with_duckdb(output_path, config.paths.duckdb_path, season)

    if not written_paths:
        raise RuntimeError("No play-by-play data was retrieved for the requested seasons.")

    return written_paths


def _load_pbp(season: int) -> pd.DataFrame:
    """Load a single season of play-by-play data as a pandas ``DataFrame``."""

    polars_df = nflreadpy.load_pbp(season)
    pdf = polars_df.to_pandas(use_pyarrow_extension_array=True)
    return pdf


def _register_with_duckdb(parquet_path: Path, duckdb_path: str, season: int) -> None:
    """Register the Parquet file as a DuckDB view for downstream consumption."""

    view_name = _DUCKDB_VIEW_TEMPLATE.format(season=season)
    try:
        with DuckDBClient(duckdb_path) as client:
            client.register_parquet(str(parquet_path), view_name)
            LOGGER.info("Registered DuckDB view '%s' for %s", view_name, parquet_path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to register DuckDB view '%s': %s", view_name, exc)


__all__ = ["ingest_pbp"]

