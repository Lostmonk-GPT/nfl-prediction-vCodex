"""Snapshot orchestration utilities for the NFL prediction pipeline."""

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_SNAPSHOT_STAGES",
    "SnapshotRunner",
    "SnapshotStage",
    "StageExecution",
    "run_snapshot_workflow",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin re-export shim
    if name in __all__:
        module = import_module("nfl_pred.snapshot.runner")
        return getattr(module, name)
    raise AttributeError(f"module 'nfl_pred.snapshot' has no attribute {name!r}")
