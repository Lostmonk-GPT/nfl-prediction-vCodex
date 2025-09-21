"""Pipeline entrypoints for NFL prediction workflows."""

from .train import FoldMetrics, TrainingResult, run_training_pipeline

__all__ = [
    "FoldMetrics",
    "TrainingResult",
    "run_training_pipeline",
]
