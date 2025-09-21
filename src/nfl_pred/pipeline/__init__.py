"""Pipeline entrypoints for NFL prediction workflows."""

from .predict import InferenceResult, run_inference_pipeline
from .train import FoldMetrics, TrainingResult, run_training_pipeline

__all__ = [
    "FoldMetrics",
    "InferenceResult",
    "TrainingResult",
    "run_inference_pipeline",
    "run_training_pipeline",
]
