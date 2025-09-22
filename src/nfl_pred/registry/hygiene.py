"""MLflow hygiene utilities for tagging runs and enforcing retention policies."""

from __future__ import annotations

import dataclasses
import logging
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse

import mlflow
from mlflow.entities import Run, ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger(__name__)


def _coerce_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_sequence(values: Iterable[int | str]) -> list[str]:
    normalized: set[str] = set()
    for item in values:
        if isinstance(item, bool):  # pragma: no cover - defensive
            normalized.add(str(int(item)))
            continue
        if isinstance(item, (int, float)):
            normalized.add(str(int(item)))
            continue

        text = str(item).strip()
        if text.startswith("-") and text[1:].isdigit():
            normalized.add(str(int(text)))
            continue
        if text.isdigit():
            normalized.add(str(int(text)))
            continue
        normalized.add(text)
    return sorted(normalized)


def _active_run() -> mlflow.ActiveRun:
    active_run = mlflow.active_run()
    if active_run is None:  # pragma: no cover - defensive guard
        raise RuntimeError("An active MLflow run is required to apply tags.")
    return active_run


@dataclass(slots=True, frozen=True)
class StandardTagPayload:
    """Normalized payload for standard MLflow tags."""

    seasons: Sequence[str] = dataclasses.field(default_factory=tuple)
    weeks: Sequence[str] = dataclasses.field(default_factory=tuple)
    snapshot_at: Sequence[str] = dataclasses.field(default_factory=tuple)
    model_id: str | None = None
    promoted: bool | None = None
    lineage: str | None = None

    def to_tags(self) -> dict[str, str]:
        tags: dict[str, str] = {}
        if self.seasons:
            tags["season"] = "|".join(self.seasons)
        if self.weeks:
            tags["week"] = "|".join(self.weeks)
        if self.snapshot_at:
            tags["snapshot_at"] = "|".join(self.snapshot_at)
        if self.model_id:
            tags["model_id"] = self.model_id
        if self.promoted is not None:
            tags["promoted"] = "true" if self.promoted else "false"
        if self.lineage:
            tags["lineage"] = self.lineage
        return tags


def build_standard_tags(
    *,
    seasons: Iterable[int | str] | None = None,
    weeks: Iterable[int | str] | None = None,
    snapshot_ats: Iterable[str | datetime] | str | datetime | None = None,
    model_id: str | None = None,
    promoted: bool | None = None,
    lineage: str | None = None,
) -> StandardTagPayload:
    """Create a :class:`StandardTagPayload` for downstream application."""

    season_values: list[str] = []
    if seasons is not None:
        season_values = _normalize_sequence(seasons)

    week_values: list[str] = []
    if weeks is not None:
        week_values = _normalize_sequence(weeks)

    snapshot_values: list[str] = []
    if snapshot_ats is not None:
        if isinstance(snapshot_ats, (str, datetime)):
            snapshot_iter: Iterable[str | datetime] = [snapshot_ats]
        else:
            snapshot_iter = snapshot_ats
        snapshot_values = [
            _format_timestamp(_coerce_timestamp(value)) for value in snapshot_iter
        ]
        snapshot_values = sorted(set(snapshot_values))

    return StandardTagPayload(
        seasons=tuple(season_values),
        weeks=tuple(week_values),
        snapshot_at=tuple(snapshot_values),
        model_id=model_id,
        promoted=promoted,
        lineage=lineage,
    )


def apply_standard_tags(
    payload: StandardTagPayload,
    extra_tags: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply standardized tags to the active MLflow run."""

    active_run = _active_run()
    tags = payload.to_tags()
    if extra_tags:
        tags.update(extra_tags)

    if not tags:
        LOGGER.debug("No tags provided for MLflow run %s; skipping.", active_run.info.run_id)
        return {}

    mlflow.set_tags(tags)
    LOGGER.info(
        "Applied MLflow tags for run %s: %s",
        active_run.info.run_id,
        ", ".join(f"{key}={value}" for key, value in sorted(tags.items())),
    )
    return tags


@dataclass(slots=True, frozen=True)
class RetentionPolicy:
    """Retention policy for pruning MLflow runs."""

    max_age_days: int | None = 180
    keep_last_runs: int = 20
    keep_top_runs: int = 10
    metric: str = "holdout_brier"
    metric_goal: str = "min"
    protect_promoted: bool = True
    min_metric_value: float | None = None

    def __post_init__(self) -> None:
        goal = self.metric_goal.lower()
        if goal not in {"min", "max"}:
            raise ValueError(
                "metric_goal must be either 'min' or 'max', "
                f"received '{self.metric_goal}'."
            )


@dataclass(slots=True)
class HygieneReport:
    """Summary of retention enforcement."""

    experiment: str
    scanned_runs: int
    deleted_runs: list[str] = field(default_factory=list)
    protected_runs: list[str] = field(default_factory=list)
    kept_runs: list[str] = field(default_factory=list)
    dry_run: bool = True
    delete_artifacts: bool = False


def enforce_retention_policy(
    *,
    tracking_uri: str | Path,
    experiment: str,
    policy: RetentionPolicy,
    dry_run: bool = True,
    delete_artifacts: bool = False,
) -> HygieneReport:
    """Apply the configured MLflow retention policy and optionally delete artifacts."""

    mlflow.set_tracking_uri(str(tracking_uri))
    client = MlflowClient(tracking_uri=str(tracking_uri))

    experiment_obj = client.get_experiment_by_name(experiment)
    if experiment_obj is None:
        raise MlflowException(f"Experiment '{experiment}' not found at {tracking_uri}.")

    runs = client.search_runs(
        [experiment_obj.experiment_id],
        max_results=5000,
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    LOGGER.info(
        "Evaluating %s runs in experiment '%s' for retention enforcement.",
        len(runs),
        experiment,
    )

    protected: set[str] = set()
    candidates: list[tuple[Run, datetime | None, float | None]] = []
    for run in runs:
        tags = run.data.tags or {}
        metric_value = run.data.metrics.get(policy.metric)
        start_time = (
            datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc)
            if run.info.start_time
            else None
        )

        if policy.protect_promoted and tags.get("promoted", "").lower() == "true":
            protected.add(run.info.run_id)
            continue

        candidates.append((run, start_time, metric_value))

    keep_recent = _select_recent_runs(candidates, policy.keep_last_runs)
    keep_metric = _select_metric_runs(candidates, policy)

    keep_set = keep_recent | keep_metric
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=policy.max_age_days) if policy.max_age_days is not None else None

    delete_targets: list[Run] = []
    kept: list[str] = []
    for run, start_time, _ in candidates:
        run_id = run.info.run_id
        if run_id in keep_set:
            kept.append(run_id)
            continue
        if cutoff is not None and start_time is not None and start_time >= cutoff:
            kept.append(run_id)
            continue
        delete_targets.append(run)

    kept_ids = set(kept) | keep_set

    report = HygieneReport(
        experiment=experiment,
        scanned_runs=len(runs),
        deleted_runs=[],
        protected_runs=sorted(protected),
        kept_runs=sorted(kept_ids),
        dry_run=dry_run,
        delete_artifacts=delete_artifacts,
    )

    if not delete_targets:
        LOGGER.info("No runs eligible for deletion under the retention policy.")
        return report

    LOGGER.info(
        "Identified %s runs for deletion (dry_run=%s).",
        len(delete_targets),
        dry_run,
    )

    for run in delete_targets:
        run_id = run.info.run_id
        report.deleted_runs.append(run_id)
        if dry_run:
            LOGGER.info("[dry-run] Would delete MLflow run %s.", run_id)
            continue

        LOGGER.info("Deleting MLflow run %s.", run_id)
        client.delete_run(run_id)
        if delete_artifacts:
            _delete_artifact_directory(run)

    return report


def _select_recent_runs(
    candidates: Sequence[tuple[Run, datetime | None, float | None]],
    keep_last: int,
) -> set[str]:
    if keep_last <= 0 or not candidates:
        return set()
    sorted_runs = sorted(
        candidates,
        key=lambda item: item[1] or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return {run.info.run_id for run, _, _ in sorted_runs[:keep_last]}


def _select_metric_runs(
    candidates: Sequence[tuple[Run, datetime | None, float | None]],
    policy: RetentionPolicy,
) -> set[str]:
    if policy.keep_top_runs <= 0 or not candidates:
        return set()

    def metric_sort_key(item: tuple[Run, datetime | None, float | None]) -> float:
        metric_value = item[2]
        if metric_value is None or math.isnan(metric_value):
            return math.inf if policy.metric_goal == "min" else -math.inf
        return float(metric_value)

    filtered: list[tuple[Run, datetime | None, float | None]] = []
    for candidate in candidates:
        metric_value = candidate[2]
        if policy.min_metric_value is not None:
            if policy.metric_goal == "min" and (
                metric_value is None or metric_value > policy.min_metric_value
            ):
                continue
            if policy.metric_goal == "max" and (
                metric_value is None or metric_value < policy.min_metric_value
            ):
                continue
        filtered.append(candidate)

    reverse = policy.metric_goal == "max"
    sorted_candidates = sorted(filtered, key=metric_sort_key, reverse=reverse)
    selected = sorted_candidates[: policy.keep_top_runs]
    return {run.info.run_id for run, _, _ in selected}


def _delete_artifact_directory(run: Run) -> None:
    artifact_uri = run.info.artifact_uri
    parsed = urlparse(artifact_uri)
    if parsed.scheme not in {"", "file"}:
        LOGGER.warning(
            "Skipping artifact deletion for run %s; unsupported scheme '%s'.",
            run.info.run_id,
            parsed.scheme,
        )
        return

    path = Path(parsed.path)
    if not path.exists():
        LOGGER.debug(
            "Artifact directory %s for run %s not found; skipping removal.",
            path,
            run.info.run_id,
        )
        return

    LOGGER.info("Removing artifact directory %s for run %s.", path, run.info.run_id)
    shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "StandardTagPayload",
    "build_standard_tags",
    "apply_standard_tags",
    "RetentionPolicy",
    "HygieneReport",
    "enforce_retention_policy",
]

