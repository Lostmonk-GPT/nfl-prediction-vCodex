"""Unit-like checks for ingestion data contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the package source is importable without installing the project.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nfl_pred.ingest.contracts import (
    PBP_REQUIRED_COLUMNS,
    ROSTER_REQUIRED_COLUMNS,
    SCHEDULE_REQUIRED_COLUMNS,
    TEAM_REQUIRED_COLUMNS,
    assert_pbp_contract,
    assert_roster_contract,
    assert_schedule_contract,
    assert_team_contract,
)


def _frame_with_columns(columns: set[str]) -> pd.DataFrame:
    """Return a tiny dataframe containing ``columns``."""

    return pd.DataFrame({column: [0] for column in columns})


@pytest.mark.parametrize(
    ("columns", "validator"),
    [
        (SCHEDULE_REQUIRED_COLUMNS, assert_schedule_contract),
        (PBP_REQUIRED_COLUMNS, assert_pbp_contract),
        (ROSTER_REQUIRED_COLUMNS, assert_roster_contract),
        (TEAM_REQUIRED_COLUMNS, assert_team_contract),
    ],
)
def test_contract_validators_accept_complete_frames(columns: set[str], validator) -> None:
    """Validators should no-op when provided all required columns."""

    df = _frame_with_columns(set(columns) | {"extra"})
    validator(df)


@pytest.mark.parametrize(
    ("columns", "missing", "validator"),
    [
        (SCHEDULE_REQUIRED_COLUMNS, "game_id", assert_schedule_contract),
        (PBP_REQUIRED_COLUMNS, "play_id", assert_pbp_contract),
        (ROSTER_REQUIRED_COLUMNS, "player_id", assert_roster_contract),
        (TEAM_REQUIRED_COLUMNS, "team_abbr", assert_team_contract),
    ],
)
def test_contract_validators_raise_on_missing(columns: set[str], missing: str, validator) -> None:
    """Validators should raise ``ValueError`` when required columns are missing."""

    partial_columns = set(columns) - {missing}
    df = _frame_with_columns(partial_columns)
    with pytest.raises(ValueError) as exc:
        validator(df)

    message = str(exc.value)
    assert missing in message
    # Provide context by ensuring a present column snippet is included.
    assert "Present columns" in message

