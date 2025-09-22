from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_pred.docs.feature_spec import generate_feature_spec
from nfl_pred.storage.duckdb_client import DuckDBClient


def _build_payload(team: str, opponent: str, *, base_rate_scale: float) -> dict[str, object]:
    payload: dict[str, object] = {
        "team": team,
        "opponent": opponent,
        "rest_days": 9.0 * base_rate_scale,
        "short_week": base_rate_scale < 1.0,
        "kickoff_bucket": "early" if base_rate_scale < 1.0 else "mnf",
        "neutral_site": False,
        "travel_miles": 500.0 * base_rate_scale,
        "days_since_last": 9.0 * base_rate_scale,
        "venue_latitude": 40.0 + base_rate_scale,
        "venue_longitude": -90.0 - base_rate_scale,
        "plays_offense": int(60 * base_rate_scale + 50),
        "pass_plays": int(32 * base_rate_scale + 25),
        "rush_plays": int(28 * base_rate_scale + 25),
        "dropbacks": int(35 * base_rate_scale + 28),
        "sacks": int(1 * base_rate_scale + 1),
        "success_plays": 30.0 * base_rate_scale + 20.0,
        "early_down_plays": int(35 * base_rate_scale + 28),
        "play_action_plays": int(8 * base_rate_scale + 6),
        "shotgun_plays": int(20 * base_rate_scale + 15),
        "no_huddle_plays": int(4 * base_rate_scale + 3),
        "explosive_pass_plays": int(5 * base_rate_scale + 3),
        "explosive_rush_plays": int(3 * base_rate_scale + 2),
        "penalties": int(6 * base_rate_scale + 4),
        "st_plays": int(10 * base_rate_scale + 8),
        "total_epa": 5.0 * base_rate_scale + 3.0,
        "early_down_epa": 2.0 * base_rate_scale + 1.5,
        "st_epa": 0.4 * base_rate_scale + 0.3,
        "wx_temp": 70.0 + 5.0 * base_rate_scale,
        "wx_wind": 3.0 + base_rate_scale,
        "precip": 0.2 * base_rate_scale,
        "is_postseason": False,
        "season_phase": "regular",
        "kickoff_2024plus": False,
        "ot_regular_2025plus": False,
        "team_score": 24.0 + 3.0 * base_rate_scale,
        "opponent_score": 20.0,
        "label_team_win": 1.0,
    }

    rate_metrics = {
        "epa_per_play": 0.10 * base_rate_scale + 0.02,
        "early_down_epa_per_play": 0.08 * base_rate_scale + 0.01,
        "success_rate": 0.55 * base_rate_scale,
        "pass_rate": 0.52 * base_rate_scale,
        "rush_rate": 0.48 * base_rate_scale,
        "play_action_rate": 0.28 * base_rate_scale,
        "shotgun_rate": 0.35 * base_rate_scale,
        "no_huddle_rate": 0.06 * base_rate_scale,
        "sack_rate": 0.05 * base_rate_scale,
        "explosive_pass_rate": 0.18 * base_rate_scale,
        "explosive_rush_rate": 0.12 * base_rate_scale,
        "penalty_rate": 0.09 * base_rate_scale,
        "st_epa_per_play": 0.04 * base_rate_scale,
    }

    for name, value in rate_metrics.items():
        payload[name] = value
        payload[f"{name}_w4"] = value * 0.9
        payload[f"{name}_w8"] = value * 0.95
        payload[f"{name}_season"] = value

    return payload


def test_generate_feature_spec_creates_markdown(tmp_path: Path) -> None:
    duckdb_path = tmp_path / "features.duckdb"
    rows = []

    payload_home = _build_payload("KAN", "DET", base_rate_scale=1.0)
    payload_away = _build_payload("DET", "KAN", base_rate_scale=0.8)

    rows.append(
        {
            "season": 2024,
            "week": 1,
            "game_id": "2024_01_KC_DET",
            "team_side": "home",
            "asof_ts": pd.Timestamp("2024-09-05T23:30:00Z"),
            "snapshot_at": pd.Timestamp("2024-09-05T20:00:00Z"),
            "feature_set": "mvp_v1",
            "created_at": pd.Timestamp("2024-09-05T20:00:00Z"),
            "payload_json": json.dumps(payload_home),
        }
    )
    rows.append(
        {
            "season": 2024,
            "week": 1,
            "game_id": "2024_01_KC_DET",
            "team_side": "away",
            "asof_ts": pd.Timestamp("2024-09-05T23:30:00Z"),
            "snapshot_at": pd.Timestamp("2024-09-05T20:00:00Z"),
            "feature_set": "mvp_v1",
            "created_at": pd.Timestamp("2024-09-05T20:00:00Z"),
            "payload_json": json.dumps(payload_away),
        }
    )

    frame = pd.DataFrame(rows)
    frame = frame[
        [
            "season",
            "week",
            "game_id",
            "team_side",
            "asof_ts",
            "snapshot_at",
            "feature_set",
            "payload_json",
            "created_at",
        ]
    ]
    with DuckDBClient(str(duckdb_path)) as client:
        client.apply_schema()
        client.write_df(frame, table="features", mode="append")

    output_path = tmp_path / "feature_spec.md"
    generated_path = generate_feature_spec(
        duckdb_path=duckdb_path,
        feature_set="mvp_v1",
        output_path=output_path,
    )

    assert generated_path == output_path
    contents = generated_path.read_text(encoding="utf-8")
    assert "Feature Specification" in contents
    assert "`plays_offense`" in contents
    assert "Numeric Feature Summary" in contents
    assert "`epa_per_play_w4`" in contents
