"""Retrain trigger evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .psi import PSISummary


@dataclass(frozen=True)
class RetrainTriggerConfig:
    """Configuration controlling retrain trigger thresholds."""

    brier_window_weeks: int = 4
    brier_deterioration_pct: float = 0.10
    psi_threshold: float = 0.2
    psi_feature_count: int = 5

    def __post_init__(self) -> None:
        if self.brier_window_weeks <= 0:
            raise ValueError("brier_window_weeks must be positive.")
        if self.brier_deterioration_pct < 0.0:
            raise ValueError("brier_deterioration_pct must be non-negative.")
        if self.psi_threshold < 0.0:
            raise ValueError("psi_threshold must be non-negative.")
        if self.psi_feature_count <= 0:
            raise ValueError("psi_feature_count must be positive.")

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, object] | None = None
    ) -> "RetrainTriggerConfig":
        """Create a configuration object from a mapping of values."""

        if mapping is None:
            return cls()

        if isinstance(mapping, cls):
            return mapping

        defaults = cls()
        return cls(
            brier_window_weeks=int(
                mapping.get("brier_window_weeks", defaults.brier_window_weeks)
            ),
            brier_deterioration_pct=float(
                mapping.get(
                    "brier_deterioration_pct", defaults.brier_deterioration_pct
                )
            ),
            psi_threshold=float(
                mapping.get("psi_threshold", defaults.psi_threshold)
            ),
            psi_feature_count=int(
                mapping.get("psi_feature_count", defaults.psi_feature_count)
            ),
        )


@dataclass(frozen=True)
class RetrainTriggerDecision:
    """Result of evaluating retrain triggers for the monitoring period."""

    brier_deterioration: bool
    psi_breach: bool
    rule_change: bool
    reasons: tuple[str, ...]

    @property
    def triggered(self) -> bool:
        """Return whether any retrain trigger has fired."""

        return self.brier_deterioration or self.psi_breach or self.rule_change


def check_brier_deterioration(
    recent_brier_scores: Sequence[float],
    baseline_brier: float,
    *,
    window: int = 4,
    deterioration_pct: float = 0.10,
) -> tuple[bool, float, float]:
    """Evaluate whether rolling Brier deterioration breaches the threshold."""

    if baseline_brier <= 0.0:
        raise ValueError("baseline_brier must be greater than zero for comparison.")

    if len(recent_brier_scores) < window:
        return False, float("nan"), float("nan")

    window_scores = np.asarray(recent_brier_scores[-window:], dtype="float64")
    rolling_average = float(np.mean(window_scores))
    deterioration = (rolling_average - baseline_brier) / baseline_brier
    triggered = deterioration >= deterioration_pct or np.isclose(
        deterioration, deterioration_pct, rtol=1e-9, atol=1e-9
    )

    return triggered, rolling_average, deterioration


def check_psi_feature_drift(
    psi_summary: PSISummary,
    *,
    psi_threshold: float = 0.2,
    feature_count: int = 5,
) -> tuple[bool, list[str]]:
    """Evaluate whether PSI drift breaches the configured tolerance."""

    feature_frame = psi_summary.feature_psi
    if "psi" not in feature_frame.columns or "feature" not in feature_frame.columns:
        raise ValueError("PSI summary must include 'feature' and 'psi' columns.")

    breaches = feature_frame.loc[feature_frame["psi"] >= psi_threshold, "feature"]
    breached_features = breaches.tolist()
    triggered = len(breached_features) >= feature_count
    return triggered, breached_features


def check_rule_flag_changes(
    previous_flags: Mapping[str, bool] | None,
    current_flags: Mapping[str, bool] | None,
) -> tuple[bool, list[str]]:
    """Evaluate whether any rule flag flipped state."""

    previous_flags = previous_flags or {}
    current_flags = current_flags or {}

    changed_messages: list[str] = []
    for name in sorted(set(previous_flags) | set(current_flags)):
        old = bool(previous_flags.get(name, False))
        new = bool(current_flags.get(name, False))
        if old != new:
            changed_messages.append(
                f"Rule flag '{name}' flipped from {old} to {new}."
            )

    return bool(changed_messages), changed_messages


def evaluate_retrain_triggers(
    *,
    recent_brier_scores: Sequence[float],
    baseline_brier: float,
    psi_summary: PSISummary,
    previous_rule_flags: Mapping[str, bool] | None,
    current_rule_flags: Mapping[str, bool] | None,
    config: RetrainTriggerConfig | Mapping[str, object] | None = None,
) -> RetrainTriggerDecision:
    """Evaluate retrain triggers across Brier, PSI, and rule flag signals."""

    cfg = (
        config
        if isinstance(config, RetrainTriggerConfig)
        else RetrainTriggerConfig.from_mapping(config)  # type: ignore[arg-type]
    )

    brier_trigger, rolling_average, deterioration = check_brier_deterioration(
        recent_brier_scores,
        baseline_brier,
        window=cfg.brier_window_weeks,
        deterioration_pct=cfg.brier_deterioration_pct,
    )

    psi_trigger, breached_features = check_psi_feature_drift(
        psi_summary,
        psi_threshold=cfg.psi_threshold,
        feature_count=cfg.psi_feature_count,
    )

    rule_trigger, rule_reasons = check_rule_flag_changes(
        previous_rule_flags, current_rule_flags
    )

    reasons: list[str] = []
    if brier_trigger:
        reasons.append(
            "{}-week rolling Brier {:.4f} deteriorated {:.1%} vs baseline {:.4f}.".format(
                cfg.brier_window_weeks,
                rolling_average,
                deterioration,
                baseline_brier,
            )
        )

    if psi_trigger:
        formatted_features = ", ".join(breached_features)
        reasons.append(
            "{} features exceeded PSI threshold {:.3f}: {}.".format(
                len(breached_features),
                cfg.psi_threshold,
                formatted_features,
            )
        )

    if rule_trigger:
        reasons.extend(rule_reasons)

    return RetrainTriggerDecision(
        brier_deterioration=brier_trigger,
        psi_breach=psi_trigger,
        rule_change=rule_trigger,
        reasons=tuple(reasons),
    )


__all__ = [
    "RetrainTriggerConfig",
    "RetrainTriggerDecision",
    "check_brier_deterioration",
    "check_psi_feature_drift",
    "check_rule_flag_changes",
    "evaluate_retrain_triggers",
]
