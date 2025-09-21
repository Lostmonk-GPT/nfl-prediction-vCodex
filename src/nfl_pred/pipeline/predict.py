"""Inference pipeline for generating NFL game win probabilities."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from nfl_pred.config import load_config
from nfl_pred.logging_setup import setup_logging
from nfl_pred.storage.duckdb_client import DuckDBClient

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceResult:
    """Container for inference outputs."""

    predictions_df: pd.DataFrame
    model_path: Path
    model_id: str
    season: int
    week: int


def run_inference_pipeline(
    *,
    model_path: str | Path,
    season: int,
    week: int,
    model_id: str | None = None,
    feature_set: str | None = None,
    feature_snapshot_at: str | pd.Timestamp | None = None,
    snapshot_at: str | pd.Timestamp | None = None,
    config_path: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    write_mode: str = "append",
) -> InferenceResult:
    """Execute the end-to-end prediction workflow for a single week."""

    setup_logging()
    config = load_config(config_path)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact '{model_path}' does not exist.")

    artifact = joblib.load(model_path)
    try:
        calibrator = artifact["calibrator"]
        feature_columns = list(artifact["feature_columns"])
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError("Model artifact missing required keys: 'calibrator'/'feature_columns'.") from exc

    metadata = artifact.get("metadata", {})

    if feature_set is None:
        feature_set = metadata.get("feature_set")
    if feature_set is None:
        raise ValueError("feature_set must be supplied explicitly or present in the model metadata.")

    if model_id is None:
        model_id = metadata.get("model_id") or model_path.stem

    snapshot_at_ts = _ensure_utc_timestamp(snapshot_at) if snapshot_at is not None else _utcnow()
    feature_snapshot_ts = (
        _ensure_utc_timestamp(feature_snapshot_at) if feature_snapshot_at is not None else None
    )

    db_path = Path(duckdb_path) if duckdb_path is not None else Path(config.paths.duckdb_path)

    features_df = _load_feature_rows_for_week(
        duckdb_path=db_path,
        feature_set=feature_set,
        season=season,
        week=week,
        snapshot_at=feature_snapshot_ts,
    )
    if features_df.empty:
        raise ValueError(
            f"No feature rows found for season={season}, week={week}, feature_set='{feature_set}'."
        )

    missing = [column for column in feature_columns if column not in features_df.columns]
    if missing:
        raise KeyError(f"Feature payloads missing required columns: {', '.join(sorted(missing))}")

    probs = calibrator.predict_proba(features_df[feature_columns])
    positive_index = _positive_class_index(calibrator)
    team_probs = _clip_probabilities(probs[:, positive_index])

    features_df = features_df.assign(p_team_win=team_probs)

    predictions_df = _to_game_level_predictions(
        features_df,
        model_id=model_id,
        snapshot_at=snapshot_at_ts,
    )

    with DuckDBClient(str(db_path)) as client:
        client.apply_schema()
        client.write_df(predictions_df, table="predictions", mode=write_mode)

    LOGGER.info(
        "Generated %s predictions for season=%s week=%s using model '%s'.",
        len(predictions_df),
        season,
        week,
        model_id,
    )

    return InferenceResult(
        predictions_df=predictions_df,
        model_path=model_path,
        model_id=model_id,
        season=season,
        week=week,
    )


def _load_feature_rows_for_week(
    *,
    duckdb_path: Path,
    feature_set: str,
    season: int,
    week: int,
    snapshot_at: pd.Timestamp | None,
) -> pd.DataFrame:
    where = ["feature_set = ?", "season = ?", "week = ?"]
    params: list[object] = [feature_set, season, week]

    if snapshot_at is not None:
        where.append("snapshot_at <= ?")
        params.append(snapshot_at)

    where_clause = " AND ".join(where)

    query = f"""
        SELECT
            season,
            week,
            game_id,
            team_side,
            asof_ts,
            snapshot_at,
            payload_json
        FROM features
        WHERE {where_clause}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY season, week, game_id, team_side
            ORDER BY snapshot_at DESC
        ) = 1
        ORDER BY game_id, team_side
    """

    with DuckDBClient(str(duckdb_path)) as client:
        client.apply_schema()
        frame = client.read_sql(query, params)

    if frame.empty:
        return pd.DataFrame()

    payload_df = pd.DataFrame.from_records(frame["payload_json"].map(json.loads))
    payload_df = payload_df.fillna(value=np.nan)

    combined = pd.concat(
        [frame.drop(columns=["payload_json"]).reset_index(drop=True), payload_df],
        axis=1,
    )

    combined["team_side"] = combined["team_side"].astype(str).str.lower()
    combined["season"] = combined["season"].astype(int)
    combined["week"] = combined["week"].astype(int)
    combined["game_id"] = combined["game_id"].astype(str)
    combined["asof_ts"] = pd.to_datetime(combined["asof_ts"], utc=True, errors="coerce")
    combined["snapshot_at"] = pd.to_datetime(combined["snapshot_at"], utc=True, errors="coerce")
    if "home_away" not in combined.columns:
        combined["home_away"] = combined["team_side"].astype(str)

    return combined


def _to_game_level_predictions(
    df: pd.DataFrame,
    *,
    model_id: str,
    snapshot_at: pd.Timestamp,
) -> pd.DataFrame:
    required = {"season", "week", "game_id", "team_side", "asof_ts", "p_team_win"}
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Feature frame missing required columns: {missing_str}")

    grouped = df.groupby(["season", "week", "game_id"], sort=False)

    records: list[dict[str, object]] = []
    for (season, week, game_id), group in grouped:
        sides = set(group["team_side"].astype(str).str.lower())
        if not {"home", "away"}.issubset(sides):
            raise ValueError(f"Game '{game_id}' missing home/away feature rows.")

        asof_values = group["asof_ts"].dropna().unique()
        if asof_values.size != 1:
            raise ValueError(f"Game '{game_id}' has inconsistent asof_ts values.")
        asof_ts = _ensure_utc_timestamp(asof_values[0])

        home_prob = float(group.loc[group["team_side"] == "home", "p_team_win"].iloc[0])
        away_prob = float(group.loc[group["team_side"] == "away", "p_team_win"].iloc[0])

        total = home_prob + away_prob
        if not np.isfinite(total) or total <= 0:
            home_norm = away_norm = 0.5
        else:
            home_norm = home_prob / total
            away_norm = 1.0 - home_norm

        pick = "home" if home_norm >= away_norm else "away"
        confidence = float(max(home_norm, away_norm))

        records.append(
            {
                "game_id": game_id,
                "season": int(season),
                "week": int(week),
                "asof_ts": asof_ts,
                "p_home_win": float(home_norm),
                "p_away_win": float(away_norm),
                "pick": pick,
                "confidence": confidence,
                "model_id": model_id,
                "snapshot_at": snapshot_at,
            }
        )

    predictions_df = pd.DataFrame.from_records(records)
    predictions_df["snapshot_at"] = predictions_df["snapshot_at"].apply(_ensure_utc_timestamp)
    predictions_df["asof_ts"] = predictions_df["asof_ts"].apply(_ensure_utc_timestamp)

    return predictions_df


def _positive_class_index(model: object) -> int:
    classes = getattr(model, "classes_", np.array([0, 1]))
    classes = np.asarray(classes)
    if classes.ndim != 1 or classes.size != 2:
        raise ValueError("Expected binary classifier with exactly two classes.")

    if 1 in classes:
        return int(np.where(classes == 1)[0][0])
    return 1


def _clip_probabilities(values: Sequence[float], *, eps: float = 1e-6) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.clip(array, eps, 1 - eps)


def _ensure_utc_timestamp(value: pd.Timestamp | str | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def main(args: Sequence[str] | None = None) -> InferenceResult:  # pragma: no cover - CLI entrypoint
    parser = argparse.ArgumentParser(description="Run inference to generate weekly predictions.")
    parser.add_argument("--model-path", dest="model_path", required=True, help="Path to the model artifact")
    parser.add_argument("--season", dest="season", type=int, required=True, help="Season identifier")
    parser.add_argument("--week", dest="week", type=int, required=True, help="Week number")
    parser.add_argument("--model-id", dest="model_id", default=None, help="Model identifier for persistence")
    parser.add_argument("--feature-set", dest="feature_set", default=None, help="Feature set to query")
    parser.add_argument(
        "--feature-snapshot",
        dest="feature_snapshot_at",
        default=None,
        help="Maximum feature snapshot timestamp (ISO format)",
    )
    parser.add_argument(
        "--snapshot-at",
        dest="snapshot_at",
        default=None,
        help="Timestamp recorded with the generated predictions (ISO format)",
    )
    parser.add_argument("--config", dest="config_path", default=None, help="Path to config YAML")

    parsed = parser.parse_args(args=args)

    return run_inference_pipeline(
        model_path=parsed.model_path,
        season=parsed.season,
        week=parsed.week,
        model_id=parsed.model_id,
        feature_set=parsed.feature_set,
        feature_snapshot_at=parsed.feature_snapshot_at,
        snapshot_at=parsed.snapshot_at,
        config_path=parsed.config_path,
    )


__all__ = [
    "InferenceResult",
    "run_inference_pipeline",
]

