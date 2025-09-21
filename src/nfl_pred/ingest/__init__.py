"""Ingestion entrypoints for nfl_pred."""

from .pbp import ingest_pbp
from .rosters import ingest_rosters, ingest_teams
from .schedules import ingest_schedules

__all__ = ["ingest_schedules", "ingest_pbp", "ingest_rosters", "ingest_teams"]
