"""Ingestion entrypoints for nfl_pred."""

from .pbp import ingest_pbp
from .schedules import ingest_schedules

__all__ = ["ingest_schedules", "ingest_pbp"]
