"""Boundary condition tests for PSI monitoring and retrain triggers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nfl_pred.monitoring.psi import PSISummary, compute_feature_psi
from nfl_pred.monitoring.triggers import (
    RetrainTriggerConfig,
    check_brier_deterioration,
    check_psi_feature_drift,
    check_rule_flag_changes,
    evaluate_retrain_triggers,
)


def _make_psi_summary(values: dict[str, float], *, threshold: float = 0.2) -> PSISummary:
    feature_frame = pd.DataFrame(
        {"feature": list(values.keys()), "psi": list(values.values())}
    )
    return PSISummary(feature_psi=feature_frame, threshold=threshold)


def test_check_brier_deterioration_triggers_at_exact_threshold() -> None:
    scores = [0.22, 0.22, 0.22, 0.22]
    triggered, rolling_average, deterioration = check_brier_deterioration(
        scores, 0.20, window=4, deterioration_pct=0.10
    )

    assert triggered
    assert rolling_average == pytest.approx(0.22)
    assert deterioration == pytest.approx(0.10)


def test_check_brier_deterioration_triggers_when_above_threshold() -> None:
    scores = [0.25, 0.24, 0.23, 0.22]
    triggered, rolling_average, deterioration = check_brier_deterioration(
        scores, 0.20, window=4, deterioration_pct=0.10
    )

    assert triggered
    assert rolling_average == pytest.approx(0.235)
    assert deterioration == pytest.approx(0.175)


def test_check_brier_deterioration_not_triggered_when_below_threshold() -> None:
    scores = [0.19, 0.21, 0.20, 0.19]
    triggered, rolling_average, deterioration = check_brier_deterioration(
        scores, 0.20, window=4, deterioration_pct=0.10
    )

    assert not triggered
    assert rolling_average == pytest.approx(0.1975)
    assert deterioration == pytest.approx(-0.0125)


def test_check_brier_deterioration_returns_nan_when_window_insufficient() -> None:
    triggered, rolling_average, deterioration = check_brier_deterioration(
        [0.23, 0.24], 0.20, window=4, deterioration_pct=0.10
    )

    assert not triggered
    assert math.isnan(rolling_average)
    assert math.isnan(deterioration)


def test_check_brier_deterioration_raises_for_non_positive_baseline() -> None:
    with pytest.raises(ValueError):
        check_brier_deterioration([0.21, 0.22, 0.23, 0.24], 0.0)


def test_check_brier_deterioration_triggers_with_rounding_atol() -> None:
    scores = [0.22, 0.22, 0.22, 0.22000000001]
    triggered, _, deterioration = check_brier_deterioration(
        scores, 0.20, window=4, deterioration_pct=0.10
    )

    assert triggered
    assert deterioration == pytest.approx(0.10, rel=0, abs=1e-10)


def test_check_psi_feature_drift_triggers_at_feature_count_threshold() -> None:
    values = {f"feature_{idx}": 0.2 for idx in range(5)}
    summary = _make_psi_summary(values, threshold=0.2)

    triggered, breached = check_psi_feature_drift(
        summary, psi_threshold=0.2, feature_count=5
    )

    assert triggered
    assert breached == [f"feature_{idx}" for idx in range(5)]


def test_check_psi_feature_drift_not_triggered_when_feature_count_not_met() -> None:
    values = {f"feature_{idx}": 0.21 for idx in range(4)}
    summary = _make_psi_summary(values, threshold=0.2)

    triggered, breached = check_psi_feature_drift(
        summary, psi_threshold=0.2, feature_count=5
    )

    assert not triggered
    assert breached == [f"feature_{idx}" for idx in range(4)]


def test_check_psi_feature_drift_respects_custom_threshold() -> None:
    values = {
        "feature_a": 0.14,
        "feature_b": 0.15,
        "feature_c": 0.16,
        "feature_d": 0.17,
        "feature_e": 0.18,
    }
    summary = _make_psi_summary(values, threshold=0.15)

    triggered, breached = check_psi_feature_drift(
        summary, psi_threshold=0.15, feature_count=3
    )

    assert triggered
    assert breached == ["feature_b", "feature_c", "feature_d", "feature_e"]


def test_check_psi_feature_drift_returns_breached_features_in_input_order() -> None:
    values = {
        "feature_z": 0.30,
        "feature_a": 0.25,
        "feature_m": 0.40,
        "feature_k": 0.10,
        "feature_q": 0.22,
    }
    summary = _make_psi_summary(values, threshold=0.2)

    _, breached = check_psi_feature_drift(summary, psi_threshold=0.2, feature_count=2)

    assert breached == ["feature_z", "feature_a", "feature_m", "feature_q"]


def test_check_psi_feature_drift_handles_no_breaches() -> None:
    summary = _make_psi_summary({"feature_a": 0.05, "feature_b": 0.07})

    triggered, breached = check_psi_feature_drift(
        summary, psi_threshold=0.2, feature_count=5
    )

    assert not triggered
    assert breached == []


def test_check_psi_feature_drift_raises_when_columns_missing() -> None:
    feature_frame = pd.DataFrame({"psi": [0.2, 0.3]})
    summary = PSISummary(feature_psi=feature_frame, threshold=0.2)

    with pytest.raises(ValueError):
        check_psi_feature_drift(summary)


def test_compute_feature_psi_matches_manual_two_bin_example() -> None:
    reference = pd.Series([0.0, 0.0, 1.0, 1.0], dtype="float64")
    current = pd.Series([0.0, 1.0, 1.0, 1.0], dtype="float64")

    psi_value, detail = compute_feature_psi(reference, current, bins=2)

    # Manual calculation replicating the PSI formula.
    ref_distribution = np.array([0.5, 0.5, 1e-6])
    cur_distribution = np.array([0.25, 0.75, 1e-6])
    expected = float(
        np.sum((cur_distribution - ref_distribution) * np.log(cur_distribution / ref_distribution))
    )

    assert psi_value == pytest.approx(expected)
    assert detail.loc[detail["bin"] == "<NULL>", "ref_proportion"].iat[0] == pytest.approx(1e-6)


def test_check_rule_flag_changes_detects_new_true_flag() -> None:
    triggered, messages = check_rule_flag_changes({"kickoff_rule": False}, {"kickoff_rule": True})

    assert triggered
    assert messages == ["Rule flag 'kickoff_rule' flipped from False to True."]


def test_check_rule_flag_changes_detects_cleared_flag() -> None:
    triggered, messages = check_rule_flag_changes({"onside_rule": True}, {"onside_rule": False})

    assert triggered
    assert messages == ["Rule flag 'onside_rule' flipped from True to False."]


def test_check_rule_flag_changes_returns_false_when_no_changes() -> None:
    triggered, messages = check_rule_flag_changes({"clock": True}, {"clock": True})

    assert not triggered
    assert messages == []


def test_check_rule_flag_changes_handles_none_inputs() -> None:
    triggered, messages = check_rule_flag_changes(None, None)

    assert not triggered
    assert messages == []


def test_evaluate_retrain_triggers_combines_all_reasons() -> None:
    decision = evaluate_retrain_triggers(
        recent_brier_scores=[0.22, 0.22, 0.22, 0.22],
        baseline_brier=0.20,
        psi_summary=_make_psi_summary({f"f{idx}": 0.25 for idx in range(5)}),
        previous_rule_flags={"extra_point": False},
        current_rule_flags={"extra_point": True},
        config=RetrainTriggerConfig(
            brier_window_weeks=4,
            brier_deterioration_pct=0.10,
            psi_threshold=0.2,
            psi_feature_count=5,
        ),
    )

    assert decision.triggered
    assert decision.brier_deterioration
    assert decision.psi_breach
    assert decision.rule_change
    assert len(decision.reasons) == 3
    assert "features exceeded PSI threshold" in decision.reasons[1]


def test_evaluate_retrain_triggers_accepts_config_mapping() -> None:
    decision = evaluate_retrain_triggers(
        recent_brier_scores=[0.19, 0.20, 0.21, 0.22],
        baseline_brier=0.20,
        psi_summary=_make_psi_summary({f"f{idx}": 0.16 for idx in range(3)}),
        previous_rule_flags={"neutral_site": False},
        current_rule_flags={"neutral_site": False},
        config={
            "brier_window_weeks": 2,
            "brier_deterioration_pct": 0.05,
            "psi_threshold": 0.15,
            "psi_feature_count": 3,
        },
    )

    assert decision.brier_deterioration
    assert decision.psi_breach
    assert not decision.rule_change
    assert decision.triggered


def test_evaluate_retrain_triggers_returns_false_when_no_conditions_met() -> None:
    decision = evaluate_retrain_triggers(
        recent_brier_scores=[0.18, 0.19, 0.18, 0.19],
        baseline_brier=0.20,
        psi_summary=_make_psi_summary({"feature_a": 0.05, "feature_b": 0.10}),
        previous_rule_flags={"clock": True},
        current_rule_flags={"clock": True},
    )

    assert not decision.triggered
    assert decision.reasons == ()

