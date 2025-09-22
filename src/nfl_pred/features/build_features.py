"""Utilities for assembling the MVP modeling feature matrix.

This module joins the previously derived feature components into a single
team/week level table that can be persisted to DuckDB. The resulting feature
payload contains:

* Play-by-play aggregates and rolling windows from
  :mod:`nfl_pred.features.team_week`.
* Schedule context (rest, kickoff buckets, home/away) from
  :mod:`nfl_pred.features.schedule_meta`.
* Travel metrics derived in :mod:`nfl_pred.features.travel`.
* Weather context derived in :mod:`nfl_pred.features.weather` when available.
* A minimal training label (`label_team_win`) computed from final scores.

Null policy
-----------
* ``rest_days`` / ``days_since_last`` are ``NaN`` for the first team game of a
  season or when the necessary kickoff timestamps are unavailable.
* ``travel_miles`` is ``NaN`` when venue coordinates cannot be resolved or the
  previous game is missing.
* ``label_team_win`` is ``NaN`` until final scores are known. Completed games
  have values in ``{0.0, 0.5, 1.0}``, with ``0.5`` representing ties.

The public :func:`build_and_store_features` function keeps feature computation
leakage-free by respecting an ``asof_ts`` cutoff, serialises each feature row as
JSON, and writes the result to the canonical ``features`` DuckDB table defined
in :mod:`nfl_pred.storage.schema`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import numpy as np
import pandas as pd

from nfl_pred.config import load_config
from nfl_pred.features.playoffs import append_playoff_flags
from nfl_pred.features.rules import append_rule_flags
from nfl_pred.features.schedule_meta import compute_schedule_meta
from nfl_pred.features.team_week import compute_team_week_features
from nfl_pred.features.travel import compute_travel_features
from nfl_pred.features.weather import compute_weather_features
from nfl_pred.snapshot.visibility import VisibilityContext, filter_schedule
from nfl_pred.storage.duckdb_client import DuckDBClient
from nfl_pred.ref.stadiums import load_stadiums

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from nfl_pred.weather import MeteostatClient, NWSClient


@dataclass(slots=True)
class FeatureBuildResult:
    """Container for feature assembly outputs.

    Attributes:
        features_df: DataFrame with one row per ``(season, week, game_id, team)``
            combination containing the assembled feature columns and label.
        payload_df: DataFrame matching the DuckDB ``features`` schema with the
            JSON serialised payload ready for persistence.
    """

    features_df: pd.DataFrame
    payload_df: pd.DataFrame


def build_and_store_features(
    pbp: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    team_locations: pd.DataFrame | None = None,
    stadiums: pd.DataFrame | None = None,
    nws_client: NWSClient | None = None,
    meteostat_client: MeteostatClient | None = None,
    asof_ts: pd.Timestamp | None = None,
    snapshot_at: pd.Timestamp | None = None,
    feature_set: str = "mvp_v1",
    write_mode: str = "replace",
    duckdb_path: str | None = None,
) -> FeatureBuildResult:
    """Assemble MVP features and persist them to DuckDB.

    Args:
        pbp: Raw play-by-play frame used for team-week aggregations. Only rows
            with ``posteam`` populated are considered.
        schedule: League schedule (as ingested via
            :mod:`nfl_pred.ingest.schedules`) containing kickoff timestamps and
            final scores.
        team_locations: Optional coordinates used by
            :func:`~nfl_pred.features.travel.compute_travel_features` when venue
            latitude/longitude are missing.
        stadiums: Optional authoritative stadium reference. When omitted, the
            table is loaded via :func:`nfl_pred.ref.stadiums.load_stadiums`.
        nws_client: Optional :class:`~nfl_pred.weather.NWSClient` for weather
            forecasts.
        meteostat_client: Optional :class:`~nfl_pred.weather.MeteostatClient`
            for historical weather backfill.
        asof_ts: Optional cutoff timestamp. Play-by-play rows and schedule
            entries strictly after this instant are ignored to avoid leakage.
            When omitted, all available data are used.
        snapshot_at: Timestamp representing when the feature build ran. Defaults
            to ``pd.Timestamp.utcnow()`` (UTC aware).
        feature_set: Identifier stored alongside the payload for downstream
            selection. Defaults to ``"mvp_v1"``.
        write_mode: Passed directly to :meth:`DuckDBClient.write_df` allowing
            ``{"create", "replace", "append"}``.
        duckdb_path: Optional override for the DuckDB database path. Falls back
            to the configured path from :func:`load_config` when omitted.

    Returns:
        :class:`FeatureBuildResult` containing both the assembled feature frame
        and the DuckDB payload frame.
    """

    if snapshot_at is None:
        snapshot_at = pd.Timestamp.utcnow().tz_localize("UTC")

    snapshot_at = _ensure_utc_timestamp(snapshot_at)

    if asof_ts is not None:
        asof_ts = _ensure_utc_timestamp(asof_ts)
    else:
        asof_ts = None

    schedule_context = VisibilityContext(asof_ts=asof_ts)
    schedule_filtered = _filter_schedule(schedule, context=schedule_context)

    team_week_features = compute_team_week_features(pbp, asof_ts=asof_ts)
    schedule_meta = compute_schedule_meta(schedule_filtered, asof_ts=asof_ts)
    schedule_meta = append_playoff_flags(schedule_meta, schedule_filtered)
    schedule_meta = append_rule_flags(schedule_meta)
    travel_features = compute_travel_features(
        schedule_filtered,
        team_locations=team_locations,
        asof_ts=asof_ts,
    )

    if stadiums is None:
        stadiums = load_stadiums()

    weather_features = compute_weather_features(
        schedule_filtered,
        stadiums,
        nws_client=nws_client,
        meteostat_client=meteostat_client,
        asof_ts=asof_ts,
    )

    assembled = _join_feature_components(
        schedule_meta=schedule_meta,
        travel_features=travel_features,
        team_week_features=team_week_features,
        weather_features=weather_features,
    )

    scores = _extract_scores(schedule_filtered)
    features_df = assembled.merge(
        scores,
        on=["season", "week", "game_id", "team"],
        how="left",
    )

    _validate_uniqueness(features_df, keys=["season", "week", "game_id", "team"])

    payload_df = _to_duckdb_payload(
        features_df,
        asof_ts=asof_ts or snapshot_at,
        snapshot_at=snapshot_at,
        feature_set=feature_set,
    )

    if duckdb_path is None:
        config = load_config()
        duckdb_path = config.paths.duckdb_path

    with DuckDBClient(duckdb_path) as client:
        client.write_df(payload_df, table="features", mode=write_mode)

    return FeatureBuildResult(features_df=features_df, payload_df=payload_df)


def _filter_schedule(schedule: pd.DataFrame, *, context: VisibilityContext) -> pd.DataFrame:
    """Return a schedule frame restricted to data visible at ``context.asof_ts``."""

    working = schedule.copy()
    working = filter_schedule(working, context=context, kickoff_column="start_time")

    if working.empty:
        return working

    working["season"] = working["season"].astype(int)
    working["week"] = working["week"].astype(int)
    working["start_time"] = pd.to_datetime(working["start_time"], utc=True, errors="coerce")

    return working


def _join_feature_components(
    *,
    schedule_meta: pd.DataFrame,
    travel_features: pd.DataFrame,
    team_week_features: pd.DataFrame,
    weather_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join schedule, travel, and team-week frames into a single feature table."""

    travel_trimmed = travel_features.drop(columns=["opponent", "home_away", "start_time"], errors="ignore")

    merged = schedule_meta.merge(
        travel_trimmed,
        on=["season", "week", "game_id", "team"],
        how="left",
    )

    if not team_week_features.empty:
        merged = merged.merge(
            team_week_features,
            on=["season", "week", "team"],
            how="left",
        )

    merged = merged.merge(
        weather_features,
        on=["season", "week", "game_id", "team"],
        how="left",
    )

    return merged


def _extract_scores(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return per-team scores and win labels from the raw schedule frame."""

    required = {"home_score", "away_score", "home_team", "away_team", "game_id"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise KeyError(f"Schedule frame missing required score columns: {missing}")

    base_columns = ["season", "week", "game_id", "home_score", "away_score"]

    home = schedule[base_columns + ["home_team", "away_team"]].copy()
    home.rename(columns={"home_team": "team", "away_team": "opponent"}, inplace=True)
    home["team_score"] = pd.to_numeric(home["home_score"], errors="coerce")
    home["opponent_score"] = pd.to_numeric(home["away_score"], errors="coerce")

    away = schedule[base_columns + ["home_team", "away_team"]].copy()
    away.rename(columns={"away_team": "team", "home_team": "opponent"}, inplace=True)
    away["team_score"] = pd.to_numeric(away["away_score"], errors="coerce")
    away["opponent_score"] = pd.to_numeric(away["home_score"], errors="coerce")

    combined = pd.concat([home, away], ignore_index=True, sort=False)
    combined = combined.drop(columns=["home_score", "away_score"])
    combined["label_team_win"] = _compute_win_label(combined["team_score"], combined["opponent_score"])

    combined = combined.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    return combined


def _compute_win_label(team_score: pd.Series, opponent_score: pd.Series) -> pd.Series:
    """Return win indicator values in ``{0.0, 0.5, 1.0}`` with ``NaN`` when unknown."""

    team_vals = pd.to_numeric(team_score, errors="coerce")
    opp_vals = pd.to_numeric(opponent_score, errors="coerce")

    missing = team_vals.isna() | opp_vals.isna()

    label = np.where(team_vals > opp_vals, 1.0, 0.0)
    label = np.where(team_vals < opp_vals, 0.0, label)
    label = np.where(team_vals.eq(opp_vals) & team_vals.notna(), 0.5, label)
    label = pd.Series(label, index=team_vals.index, dtype="float64")
    label = label.mask(missing)
    return label


def _validate_uniqueness(df: pd.DataFrame, *, keys: Iterable[str]) -> None:
    """Ensure the assembled features have a single row per key combination."""

    duplicated = df.duplicated(subset=list(keys), keep=False)
    if duplicated.any():
        dup_rows = df.loc[duplicated, list(keys)].drop_duplicates()
        raise ValueError(
            "Duplicate feature rows detected for keys: " f"{dup_rows.to_dict(orient='records')}"
        )


def _to_duckdb_payload(
    features_df: pd.DataFrame,
    *,
    asof_ts: pd.Timestamp,
    snapshot_at: pd.Timestamp,
    feature_set: str,
) -> pd.DataFrame:
    """Serialise feature rows into the DuckDB ``features`` table schema."""

    asof_ts = _ensure_utc_timestamp(asof_ts)
    snapshot_at = _ensure_utc_timestamp(snapshot_at)

    payload_columns = [
        column
        for column in features_df.columns
        if column not in {"season", "week", "game_id", "home_away"}
    ]

    payload_json = features_df[payload_columns].apply(_serialise_row, axis=1)

    payload = pd.DataFrame(
        {
            "season": features_df["season"].astype(int),
            "week": features_df["week"].astype(int),
            "game_id": features_df["game_id"].astype(str),
            "team_side": features_df["home_away"].astype(str),
            "asof_ts": asof_ts,
            "snapshot_at": snapshot_at,
            "feature_set": feature_set,
            "payload_json": payload_json,
        }
    )

    return payload


def _serialise_row(row: pd.Series) -> str:
    """Convert a feature row into a JSON string suitable for storage."""

    record: dict[str, object] = {}
    for key, value in row.items():
        if pd.isna(value):
            record[key] = None
            continue

        if isinstance(value, pd.Timestamp):
            if value.tzinfo is None:
                value = value.tz_localize("UTC")
            record[key] = value.tz_convert("UTC").isoformat()
        elif isinstance(value, (np.floating, np.integer)):
            record[key] = value.item()
        else:
            record[key] = value

    return json.dumps(record, sort_keys=True)


def _ensure_utc_timestamp(value: pd.Timestamp | str | np.datetime64) -> pd.Timestamp:
    """Return a UTC-normalised pandas ``Timestamp`` from assorted inputs."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


__all__ = ["FeatureBuildResult", "build_and_store_features"]

