"""Utilities for promoting MLflow runs into the model registry."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger(__name__)


_MISSING_MODEL_CODES = {RESOURCE_DOES_NOT_EXIST, "RESOURCE_DOES_NOT_EXIST"}


@dataclass(frozen=True)
class PromotionCriteria:
    """Thresholds that a run must satisfy to be eligible for promotion."""

    max_holdout_brier: float
    max_holdout_log_loss: float
    max_cv_mean_brier: float | None = None


@dataclass(frozen=True)
class PromotionDecision:
    """Decision describing whether a run should be promoted."""

    promote: bool
    rationale: str


@dataclass(frozen=True)
class PromotionResult:
    """Result returned after evaluating and applying promotion logic."""

    run_id: str
    promoted: bool
    rationale: str
    decision_path: Path
    model_name: str | None = None
    model_version: int | None = None


_REQUIRED_METRICS = ("holdout_brier", "holdout_log_loss")
_OPTIONAL_METRICS = ("cv_mean_brier",)


def evaluate_promotion(
    metrics: Mapping[str, float],
    criteria: PromotionCriteria,
) -> PromotionDecision:
    """Evaluate run metrics against promotion criteria."""

    missing = [name for name in _REQUIRED_METRICS if name not in metrics]
    if missing:
        rationale = f"Missing required metrics: {', '.join(sorted(missing))}."
        LOGGER.info("Promotion rejected: %s", rationale)
        return PromotionDecision(promote=False, rationale=rationale)

    holdout_brier = float(metrics["holdout_brier"])
    if holdout_brier > criteria.max_holdout_brier:
        rationale = (
            f"Holdout Brier {holdout_brier:.4f} exceeds threshold {criteria.max_holdout_brier:.4f}."
        )
        LOGGER.info("Promotion rejected: %s", rationale)
        return PromotionDecision(promote=False, rationale=rationale)

    holdout_log_loss = float(metrics["holdout_log_loss"])
    if holdout_log_loss > criteria.max_holdout_log_loss:
        rationale = (
            "Holdout log-loss {:.4f} exceeds threshold {:.4f}.".format(
                holdout_log_loss,
                criteria.max_holdout_log_loss,
            )
        )
        LOGGER.info("Promotion rejected: %s", rationale)
        return PromotionDecision(promote=False, rationale=rationale)

    if criteria.max_cv_mean_brier is not None:
        if "cv_mean_brier" not in metrics:
            rationale = "Missing cv_mean_brier metric required by criteria."
            LOGGER.info("Promotion rejected: %s", rationale)
            return PromotionDecision(promote=False, rationale=rationale)

        cv_mean_brier = float(metrics["cv_mean_brier"])
        if cv_mean_brier > criteria.max_cv_mean_brier:
            rationale = (
                "Cross-validation mean Brier {:.4f} exceeds threshold {:.4f}.".format(
                    cv_mean_brier,
                    criteria.max_cv_mean_brier,
                )
            )
            LOGGER.info("Promotion rejected: %s", rationale)
            return PromotionDecision(promote=False, rationale=rationale)

    return PromotionDecision(
        promote=True,
        rationale="All promotion criteria satisfied.",
    )


def promote_run(
    run_id: str,
    *,
    tracking_uri: str | Path,
    data_dir: str | Path,
    model_name: str,
    artifact_path: str,
    criteria: PromotionCriteria,
    stage: str = "Production",
) -> PromotionResult:
    """Evaluate a run and promote it in MLflow if criteria are met."""

    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_registry_uri(str(tracking_uri))
    client = MlflowClient(tracking_uri=str(tracking_uri))

    run = client.get_run(run_id)
    metrics = run.data.metrics
    decision = evaluate_promotion(metrics, criteria)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    decisions_dir = Path(data_dir) / "models" / "promotion_decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    decision_path = decisions_dir / f"decision_{timestamp}_{run_id}.json"

    payload = {
        "run_id": run_id,
        "decision_at": timestamp,
        "promoted": decision.promote,
        "rationale": decision.rationale,
        "metrics": {name: metrics.get(name) for name in (*_REQUIRED_METRICS, *_OPTIONAL_METRICS)},
        "criteria": {
            "max_holdout_brier": criteria.max_holdout_brier,
            "max_holdout_log_loss": criteria.max_holdout_log_loss,
            "max_cv_mean_brier": criteria.max_cv_mean_brier,
        },
    }
    decision_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    tag_payload = {
        "promoted": "true" if decision.promote else "false",
        "promotion_checked_at": timestamp,
        "promotion_rationale": decision.rationale,
    }

    model_version: int | None = None

    if decision.promote:
        _ensure_registered_model(client, model_name)

        source = f"runs:/{run_id}/{artifact_path}"
        version = client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
        )
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage=stage,
            archive_existing_versions=True,
        )

        model_version = int(version.version)
        tag_payload["model_id"] = f"{model_name}:{model_version}"
        LOGGER.info(
            "Promoted run %s to model '%s' version %s at stage %s.",
            run_id,
            model_name,
            model_version,
            stage,
        )
    else:
        LOGGER.info("Promotion skipped for run %s: %s", run_id, decision.rationale)

    for key, value in tag_payload.items():
        client.set_tag(run_id, key, value)

    return PromotionResult(
        run_id=run_id,
        promoted=decision.promote,
        rationale=decision.rationale,
        decision_path=decision_path,
        model_name=model_name if decision.promote else None,
        model_version=model_version,
    )


def _ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    """Create the registered model if it does not already exist."""

    try:
        client.get_registered_model(model_name)
    except MlflowException as exc:  # pragma: no cover - defensive
        if exc.error_code in _MISSING_MODEL_CODES:
            client.create_registered_model(model_name)
        else:
            raise
