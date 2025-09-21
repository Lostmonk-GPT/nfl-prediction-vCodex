"""Ingestion entrypoints and data contracts for :mod:`nfl_pred`."""

from .contracts import (
    assert_pbp_contract,
    assert_roster_contract,
    assert_schedule_contract,
    assert_team_contract,
)
from .pbp import ingest_pbp
from .injuries import ingest_injuries
from .rosters import ingest_rosters, ingest_teams
from .schedules import ingest_schedules

__all__ = [
    "assert_schedule_contract",
    "assert_pbp_contract",
    "assert_roster_contract",
    "assert_team_contract",
    "ingest_schedules",
    "ingest_pbp",
    "ingest_rosters",
    "ingest_teams",
    "ingest_injuries",
]
