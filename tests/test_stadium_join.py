from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nfl_pred.features.stadium_join import join_stadium_metadata


def _authoritative_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "venue": ["Sample Stadium", "Alternate Field"],
            "teams": [("HOM", "ALT"), ("NUT",)],
            "lat": [40.0, 30.0],
            "lon": [-75.0, -90.0],
            "tz": ["America/New_York", "America/Chicago"],
            "altitude_ft": [100, 200],
            "surface": ["artificial_turf", "natural_grass"],
            "roof": ["dome", "open"],
            "neutral_site": [False, True],
        }
    )


def test_join_stadium_metadata_prefers_authoritative_fields(caplog) -> None:
    schedule = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "game_id": ["2024_01_HOM_ALT"],
            "stadium": ["Sample Stadium"],
            "start_time": ["2024-09-07T17:00:00Z"],
            "home_team": ["hom"],
            "away_team": ["ALT"],
            "surface": ["natural_grass"],
            "roof": ["open"],
            "neutral_site": [True],
        }
    )

    with caplog.at_level("WARNING"):
        result = join_stadium_metadata(schedule, _authoritative_table())

    expected = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "game_id": ["2024_01_HOM_ALT"],
            "venue": ["Sample Stadium"],
            "roof": ["dome"],
            "surface": ["artificial_turf"],
            "tz": ["America/New_York"],
            "lat": [40.0],
            "lon": [-75.0],
            "neutral_site": [False],
        }
    )

    assert_frame_equal(result, expected)
    assert "Schedule roof differs" in caplog.text
    assert "Schedule surface differs" in caplog.text
    assert "Schedule neutral_site differs" in caplog.text


def test_join_stadium_metadata_warns_when_missing_authority(caplog) -> None:
    schedule = pd.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "game_id": ["2024_02_NEU_VIS"],
            "stadium": ["Unknown Venue"],
            "home_team": ["NEU"],
            "away_team": ["VIS"],
        }
    )

    with caplog.at_level("WARNING"):
        result = join_stadium_metadata(schedule, _authoritative_table())

    expected = pd.DataFrame(
        {
            "season": [2024],
            "week": [2],
            "game_id": ["2024_02_NEU_VIS"],
            "venue": ["Unknown Venue"],
            "roof": [float("nan")],
            "surface": [float("nan")],
            "tz": [float("nan")],
            "lat": [float("nan")],
            "lon": [float("nan")],
            "neutral_site": [False],
        }
    )

    assert_frame_equal(result, expected, check_dtype=False)
    assert "Missing authoritative stadium rows" in caplog.text
