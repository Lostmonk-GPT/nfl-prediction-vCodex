"""Reporting utilities for evaluation metrics and reliability analysis."""

from .expanded import (
    ExpandedMetricConfig,
    build_expanded_metrics,
    plot_expanded_metric,
    prepare_report_records,
    save_expanded_metrics,
)
from .metrics import (
    MetricsResult,
    ReliabilityBin,
    compute_classification_metrics,
    compute_reliability_table,
    plot_reliability_curve,
    save_metrics_report,
    save_reliability_report,
)

__all__ = [
    "ExpandedMetricConfig",
    "MetricsResult",
    "ReliabilityBin",
    "build_expanded_metrics",
    "compute_classification_metrics",
    "compute_reliability_table",
    "plot_expanded_metric",
    "plot_reliability_curve",
    "prepare_report_records",
    "save_expanded_metrics",
    "save_metrics_report",
    "save_reliability_report",
]
