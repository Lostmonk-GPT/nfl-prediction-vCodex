from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_pred.monitoring.psi import PSISummary
from nfl_pred.monitoring.triggers import RetrainTriggerConfig
from nfl_pred.reporting.expanded import ExpandedMetricConfig
from nfl_pred.reporting.monitoring_report import (
    build_monitoring_summary,
    compute_monitoring_psi_from_features,
    plot_psi_barchart,
    prepare_brier_inputs,
)


def test_prepare_brier_inputs_returns_recent_and_baseline() -> None:
    expanded = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4, 5],
            "window": ["weekly"] * 5,
            "slice": ["overall"] * 5,
            "brier_score": [0.09, 0.095, 0.1, 0.11, 0.12],
            "log_loss": np.linspace(0.4, 0.44, 5),
            "n_observations": [4, 4, 4, 4, 4],
        }
    )

    recent, baseline = prepare_brier_inputs(
        expanded,
        season=2024,
        week=5,
        window=3,
        config=ExpandedMetricConfig(),
    )

    assert recent == [0.1, 0.11, 0.12]
    assert baseline == np.mean([0.09, 0.095])


def test_build_monitoring_summary_combines_metrics_and_triggers() -> None:
    expanded_metrics = pd.DataFrame(
        [
            {"season": 2024, "week": 1, "window": "weekly", "slice": "overall", "brier_score": 0.09, "log_loss": 0.40, "n_observations": 4},
            {"season": 2024, "week": 2, "window": "weekly", "slice": "overall", "brier_score": 0.095, "log_loss": 0.41, "n_observations": 4},
            {"season": 2024, "week": 3, "window": "weekly", "slice": "overall", "brier_score": 0.11, "log_loss": 0.42, "n_observations": 4},
            {"season": 2024, "week": 3, "window": "season_to_date", "slice": "overall", "brier_score": 0.098, "log_loss": 0.41, "n_observations": 12},
            {"season": 2024, "week": 3, "window": "rolling_4", "slice": "overall", "brier_score": 0.105, "log_loss": 0.415, "n_observations": 8},
        ]
    )

    psi_frame = pd.DataFrame(
        {
            "feature": ["feat_a", "feat_b", "feat_c"],
            "psi": [0.25, 0.22, 0.05],
        }
    )
    psi_frame["threshold"] = 0.2
    psi_frame["breached"] = psi_frame["psi"] >= psi_frame["threshold"]
    psi_summary = PSISummary(feature_psi=psi_frame, threshold=0.2)

    weekly_metrics = pd.Series(
        {"brier_score": 0.11, "log_loss": 0.42, "n_observations": 4}
    )

    trigger_config = RetrainTriggerConfig(
        brier_window_weeks=2,
        brier_deterioration_pct=0.0,
        psi_threshold=0.2,
        psi_feature_count=2,
    )

    computation = build_monitoring_summary(
        season=2024,
        week=3,
        generated_at=datetime.now(timezone.utc),
        asof_ts=pd.Timestamp("2024-10-01T12:00:00Z"),
        weekly_metrics=weekly_metrics,
        expanded_metrics=expanded_metrics,
        psi_summary=psi_summary,
        trigger_config=trigger_config,
        previous_rule_flags={"kickoff_rule": False},
        current_rule_flags={"kickoff_rule": True},
        expanded_config=ExpandedMetricConfig(),
    )

    summary = computation.summary
    assert summary["season"] == 2024
    assert summary["week"] == 3
    assert summary["psi"]["breach_count"] == 2
    assert set(summary["psi"]["breached_features"]) == {"feat_a", "feat_b"}
    assert summary["retrain_triggers"]["triggered"]
    assert len(summary["retrain_triggers"]["reasons"]) >= 2
    assert summary["rolling_brier_mean"] is not None
    assert summary["baseline_brier"] == 0.09


def test_compute_monitoring_psi_from_features_uses_reference_history() -> None:
    features_df = pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "week": [17, 17, 1, 1],
            "game_id": ["G1", "G1", "G2", "G2"],
            "team_side": ["home", "away", "home", "away"],
            "asof_ts": pd.Timestamp("2024-01-01T12:00:00Z"),
            "snapshot_at": pd.Timestamp("2024-01-01T12:00:00Z"),
            "feature_set": ["mvp_v1", "mvp_v1", "mvp_v1", "mvp_v1"],
            "feature_one": [0.1, 0.2, 0.4, 0.5],
            "feature_two": [1.0, 1.2, 0.8, 0.9],
        }
    )

    summary = compute_monitoring_psi_from_features(
        features_df,
        season=2024,
        week=1,
        psi_threshold=0.2,
    )

    assert isinstance(summary, PSISummary)
    assert summary.threshold == 0.2
    assert summary.feature_psi.shape[0] == 2
    assert summary.breach_count >= 0


def test_plot_psi_barchart_creates_file(tmp_path: Path) -> None:
    psi_frame = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "psi": [0.3, 0.1],
        }
    )
    psi_frame["threshold"] = 0.2
    psi_frame["breached"] = psi_frame["psi"] >= psi_frame["threshold"]
    psi_summary = PSISummary(feature_psi=psi_frame, threshold=0.2)

    output_path = tmp_path / "psi_plot.png"
    result_path = plot_psi_barchart(psi_summary, path=output_path)

    assert result_path.exists()
