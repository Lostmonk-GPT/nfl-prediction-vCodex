"""Monitoring report assembly helpers for CLI consumption."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from nfl_pred.monitoring.psi import PSISummary, compute_psi_summary
from nfl_pred.monitoring.triggers import (
    RetrainTriggerConfig,
    RetrainTriggerDecision,
    evaluate_retrain_triggers,
)
from nfl_pred.reporting.expanded import ExpandedMetricConfig
from nfl_pred.storage.duckdb_client import DuckDBClient


_FEATURE_METADATA_COLUMNS = {
    "season",
    "week",
    "game_id",
    "team_side",
    "home_away",
    "asof_ts",
    "snapshot_at",
    "feature_set",
}


@dataclass(frozen=True)
class MonitoringComputation:
    """Container describing monitoring summary outputs."""

    summary: dict[str, object]
    decision: RetrainTriggerDecision
    recent_brier_scores: list[float]
    baseline_brier: float


def load_feature_payloads(
    duckdb_path: str | Path,
    *,
    feature_set: str,
) -> pd.DataFrame:
    """Load the latest feature payload rows for the requested feature set."""

    query = """
        SELECT
            season,
            week,
            game_id,
            team_side,
            asof_ts,
            snapshot_at,
            feature_set,
            payload_json
        FROM features
        WHERE feature_set = ?
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY season, week, game_id, team_side
            ORDER BY snapshot_at DESC
        ) = 1
        ORDER BY season, week, team_side
    """

    with DuckDBClient(str(duckdb_path)) as client:
        client.apply_schema()
        frame = client.read_sql(query, (feature_set,))

    if frame.empty:
        raise ValueError(f"Feature table returned no rows for feature_set='{feature_set}'.")

    payload_df = pd.DataFrame.from_records(frame["payload_json"].map(json.loads))
    payload_df = payload_df.fillna(value=np.nan)

    combined = pd.concat(
        [frame.drop(columns=["payload_json"]).reset_index(drop=True), payload_df], axis=1
    )

    combined["season"] = combined["season"].astype(int)
    combined["week"] = combined["week"].astype(int)
    combined["team_side"] = combined["team_side"].astype(str)

    if "home_away" not in combined.columns:
        combined["home_away"] = combined["team_side"].astype(str)

    return combined


def compute_monitoring_psi_from_features(
    features_df: pd.DataFrame,
    *,
    season: int,
    week: int,
    psi_threshold: float,
    bins: int = 10,
    reference_weeks: Sequence[int] | None = None,
) -> PSISummary:
    """Compute a PSI summary for monitoring given a features dataframe."""

    if features_df.empty:
        raise ValueError("Features dataframe is empty; cannot compute PSI.")

    if "season" not in features_df.columns or "week" not in features_df.columns:
        raise KeyError("Features dataframe must include 'season' and 'week' columns.")

    current_mask = (features_df["season"] == season) & (features_df["week"] == week)
    current_frame = features_df.loc[current_mask].copy()
    if current_frame.empty:
        raise ValueError(
            f"No feature rows available for season {season} week {week} to use as current frame."
        )

    reference_mask = features_df["season"] < season
    if reference_weeks is not None:
        reference_mask |= (
            (features_df["season"] == season)
            & (features_df["week"].isin(list(reference_weeks)))
            & (features_df["week"] < week)
        )
    else:
        reference_mask |= (
            (features_df["season"] == season)
            & (features_df["week"] < week)
        )

    reference_frame = features_df.loc[reference_mask].copy()
    if reference_frame.empty:
        raise ValueError(
            "Reference frame is empty; provide earlier weeks or previous seasons for comparison."
        )

    candidate_columns = [
        column
        for column in features_df.columns
        if column not in _FEATURE_METADATA_COLUMNS
    ]

    numeric_columns: list[str] = []
    for column in candidate_columns:
        dtype = features_df[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_columns.append(column)

    if not numeric_columns:
        raise ValueError("No numeric feature columns available for PSI computation.")

    reference_numeric = reference_frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    current_numeric = current_frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    usable_columns = [
        column
        for column in numeric_columns
        if not reference_numeric[column].isna().all() or not current_numeric[column].isna().all()
    ]

    if not usable_columns:
        raise ValueError("All candidate PSI columns are empty after coercion.")

    summary = compute_psi_summary(
        reference_numeric[usable_columns],
        current_numeric[usable_columns],
        usable_columns,
        bins=bins,
        threshold=psi_threshold,
    )

    return summary


def prepare_brier_inputs(
    expanded_metrics: pd.DataFrame,
    *,
    season: int,
    week: int,
    window: int,
    config: ExpandedMetricConfig | None = None,
) -> tuple[list[float], float]:
    """Extract recent Brier scores and a historical baseline for trigger evaluation."""

    if window <= 0:
        raise ValueError("Rolling window must be positive for Brier preparation.")

    cfg = config or ExpandedMetricConfig()
    required_columns = {
        cfg.season_column,
        cfg.week_column,
        cfg.window_column,
        cfg.slice_column,
        "brier_score",
    }
    missing = required_columns.difference(expanded_metrics.columns)
    if missing:
        raise KeyError(f"Expanded metrics missing required columns: {sorted(missing)}")

    weekly_mask = (
        (expanded_metrics[cfg.season_column] == season)
        & (expanded_metrics[cfg.window_column] == "weekly")
        & (expanded_metrics[cfg.slice_column] == "overall")
        & (expanded_metrics[cfg.week_column] <= week)
    )
    weekly = expanded_metrics.loc[weekly_mask].copy()
    if weekly.empty:
        raise ValueError(
            f"No weekly metrics available for season {season} up to week {week}."
        )

    weekly.sort_values(cfg.week_column, inplace=True)
    recent = weekly.tail(window)
    recent_scores = [float(score) for score in recent["brier_score"].tolist()]

    if len(weekly) > len(recent_scores):
        history = weekly.iloc[: len(weekly) - len(recent_scores)]
    else:
        history = weekly.iloc[:1]

    baseline_brier = float(history["brier_score"].mean())

    return recent_scores, baseline_brier


def build_monitoring_summary(
    *,
    season: int,
    week: int,
    generated_at: datetime,
    asof_ts: pd.Timestamp | None,
    weekly_metrics: pd.Series,
    expanded_metrics: pd.DataFrame,
    psi_summary: PSISummary,
    trigger_config: RetrainTriggerConfig | Mapping[str, object] | None = None,
    previous_rule_flags: Mapping[str, bool] | None = None,
    current_rule_flags: Mapping[str, bool] | None = None,
    expanded_config: ExpandedMetricConfig | None = None,
) -> MonitoringComputation:
    """Construct the monitoring summary payload and trigger decision."""

    cfg = expanded_config or ExpandedMetricConfig()
    retrain_cfg = (
        trigger_config
        if isinstance(trigger_config, RetrainTriggerConfig)
        else RetrainTriggerConfig.from_mapping(trigger_config)
    )

    recent_scores, baseline_brier = prepare_brier_inputs(
        expanded_metrics,
        season=season,
        week=week,
        window=retrain_cfg.brier_window_weeks,
        config=cfg,
    )

    decision = evaluate_retrain_triggers(
        recent_brier_scores=recent_scores,
        baseline_brier=baseline_brier,
        psi_summary=psi_summary,
        previous_rule_flags=previous_rule_flags,
        current_rule_flags=current_rule_flags,
        config=retrain_cfg,
    )

    summary = _construct_summary_payload(
        season=season,
        week=week,
        generated_at=generated_at,
        asof_ts=asof_ts,
        weekly_metrics=weekly_metrics,
        expanded_metrics=expanded_metrics,
        psi_summary=psi_summary,
        decision=decision,
        retrain_cfg=retrain_cfg,
        recent_scores=recent_scores,
        baseline_brier=baseline_brier,
        expanded_config=cfg,
    )

    return MonitoringComputation(
        summary=summary,
        decision=decision,
        recent_brier_scores=[float(score) for score in recent_scores],
        baseline_brier=float(baseline_brier),
    )


def plot_psi_barchart(
    psi_summary: PSISummary,
    *,
    path: Path,
    top_n: int = 10,
) -> Path:
    """Render a horizontal bar chart of top PSI features and persist to ``path``."""

    feature_frame = psi_summary.feature_psi.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("PSI")
    ax.set_title("Population Stability Index — Top Features")

    if feature_frame.empty:
        ax.text(0.5, 0.5, "No features available", ha="center", va="center")
        ax.set_yticks([])
    else:
        colors = [
            "#d62728" if breached else "#1f77b4"
            for breached in feature_frame["breached"].tolist()
        ]
        ax.barh(
            feature_frame["feature"],
            feature_frame["psi"],
            color=colors,
        )
        ax.invert_yaxis()

    ax.grid(True, axis="x", alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def _construct_summary_payload(
    *,
    season: int,
    week: int,
    generated_at: datetime,
    asof_ts: pd.Timestamp | None,
    weekly_metrics: pd.Series,
    expanded_metrics: pd.DataFrame,
    psi_summary: PSISummary,
    decision: RetrainTriggerDecision,
    retrain_cfg: RetrainTriggerConfig,
    recent_scores: Sequence[float],
    baseline_brier: float,
    expanded_config: ExpandedMetricConfig,
) -> dict[str, object]:
    metrics_section = {
        "weekly": _extract_metric_values(weekly_metrics),
        "season_to_date": _extract_metric_values(
            _get_window_row(
                expanded_metrics,
                season=season,
                week=week,
                window_label="season_to_date",
                config=expanded_config,
            )
        ),
        "rolling": _extract_metric_values(
            _get_window_row(
                expanded_metrics,
                season=season,
                week=week,
                window_label=f"rolling_{expanded_config.rolling_window}",
                config=expanded_config,
            )
        ),
    }

    recent_clean = [float(score) for score in recent_scores if not math.isnan(float(score))]
    rolling_mean = float(np.mean(recent_clean)) if recent_clean else math.nan

    psi_frame = psi_summary.feature_psi.copy()
    top_features = [
        {
            "feature": str(row["feature"]),
            "psi": _safe_float(row.get("psi")),
            "breached": bool(row.get("breached", False)),
        }
        for _, row in psi_frame.iterrows()
    ]

    summary: dict[str, object] = {
        "season": season,
        "week": week,
        "generated_at": generated_at.isoformat(),
        "asof_ts": asof_ts.isoformat() if asof_ts is not None else None,
        "metrics": metrics_section,
        "rolling_window_weeks": retrain_cfg.brier_window_weeks,
        "rolling_brier_mean": _safe_float(rolling_mean),
        "recent_brier_scores": recent_clean,
        "baseline_brier": _safe_float(baseline_brier),
        "psi": {
            "threshold": retrain_cfg.psi_threshold,
            "breach_count": psi_summary.breach_count,
            "breached_features": psi_summary.breached_features,
            "top_features": top_features,
            "feature_count_trigger": retrain_cfg.psi_feature_count,
        },
        "retrain_triggers": {
            "triggered": decision.triggered,
            "brier_deterioration": decision.brier_deterioration,
            "psi_breach": decision.psi_breach,
            "rule_change": decision.rule_change,
            "reasons": list(decision.reasons),
        },
    }

    return summary


def _extract_metric_values(row: pd.Series | None) -> dict[str, object]:
    if row is None or row.empty:
        return {}

    values: dict[str, object] = {}
    if "brier_score" in row:
        values["brier_score"] = _safe_float(row.get("brier_score"))
    if "log_loss" in row:
        values["log_loss"] = _safe_float(row.get("log_loss"))
    if "n_observations" in row and not pd.isna(row.get("n_observations")):
        values["n_observations"] = int(row.get("n_observations"))
    return values


def _get_window_row(
    expanded_metrics: pd.DataFrame,
    *,
    season: int,
    week: int,
    window_label: str,
    config: ExpandedMetricConfig,
) -> pd.Series | None:
    mask = (
        (expanded_metrics[config.season_column] == season)
        & (expanded_metrics[config.week_column] == week)
        & (expanded_metrics[config.window_column] == window_label)
        & (expanded_metrics[config.slice_column] == "overall")
    )
    subset = expanded_metrics.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


__all__ = [
    "MonitoringComputation",
    "build_monitoring_summary",
    "compute_monitoring_psi_from_features",
    "load_feature_payloads",
    "plot_psi_barchart",
    "prepare_brier_inputs",
]

