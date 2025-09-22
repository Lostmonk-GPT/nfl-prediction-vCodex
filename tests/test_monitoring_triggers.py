"""Unit tests for retrain trigger evaluation utilities."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nfl_pred.monitoring.psi import PSISummary
from nfl_pred.monitoring.triggers import (
    RetrainTriggerConfig,
    evaluate_retrain_triggers,
    check_brier_deterioration,
    check_psi_feature_drift,
    check_rule_flag_changes,
)


def _make_psi_summary(values: list[tuple[str, float]], threshold: float = 0.2) -> PSISummary:
    frame = pd.DataFrame(values, columns=["feature", "psi"])
    return PSISummary(feature_psi=frame, threshold=threshold)


def test_check_brier_deterioration_respects_threshold_boundary() -> None:
    scores = [0.088, 0.088, 0.088, 0.088]
    triggered, rolling_average, deterioration = check_brier_deterioration(
        scores,
        baseline_brier=0.08,
        window=4,
        deterioration_pct=0.10,
    )

    assert triggered
    assert math.isclose(rolling_average, 0.088)
    assert math.isclose(deterioration, 0.10, rel_tol=1e-6)


def test_check_psi_feature_drift_requires_minimum_feature_count() -> None:
    summary = _make_psi_summary(
        [
            ("f1", 0.21),
            ("f2", 0.19),
            ("f3", 0.24),
            ("f4", 0.27),
            ("f5", 0.2),
            ("f6", 0.22),
        ]
    )

    triggered, features = check_psi_feature_drift(
        summary, psi_threshold=0.2, feature_count=5
    )

    assert triggered
    assert set(features) == {"f1", "f3", "f4", "f5", "f6"}


def test_check_rule_flag_changes_detects_flips() -> None:
    previous = {"kickoff_rule": False, "extra_point": True}
    current = {"kickoff_rule": True, "extra_point": True}

    triggered, reasons = check_rule_flag_changes(previous, current)

    assert triggered
    assert reasons == ["Rule flag 'kickoff_rule' flipped from False to True."]


def test_evaluate_retrain_triggers_combines_signals() -> None:
    psi_summary = _make_psi_summary(
        [
            ("f1", 0.25),
            ("f2", 0.18),
            ("f3", 0.22),
            ("f4", 0.21),
            ("f5", 0.23),
            ("f6", 0.20),
        ]
    )

    decision = evaluate_retrain_triggers(
        recent_brier_scores=[0.09, 0.094, 0.095, 0.092],
        baseline_brier=0.08,
        psi_summary=psi_summary,
        previous_rule_flags={"kickoff_rule": False},
        current_rule_flags={"kickoff_rule": True},
        config=RetrainTriggerConfig(),
    )

    assert decision.triggered
    assert decision.brier_deterioration
    assert decision.psi_breach
    assert decision.rule_change
    assert len(decision.reasons) == 3
