from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from nfl_pred.registry.promote import (
    PromotionCriteria,
    PromotionDecision,
    evaluate_promotion,
    promote_run,
)


@pytest.fixture()
def tracking_setup(tmp_path: Path) -> tuple[str, Path]:
    tracking_uri = tmp_path / "mlruns"
    data_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(tracking_uri), data_dir


def _create_run(
    tracking_uri: str,
    artifact_dir: Path,
    *,
    holdout_brier: float,
    holdout_log_loss: float,
    cv_mean_brier: float | None,
    artifact_contents: str,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run() as active_run:
        mlflow.log_metric("holdout_brier", holdout_brier)
        mlflow.log_metric("holdout_log_loss", holdout_log_loss)
        if cv_mean_brier is not None:
            mlflow.log_metric("cv_mean_brier", cv_mean_brier)

        model_file = artifact_dir / "model.joblib"
        model_file.write_text(artifact_contents, encoding="utf-8")
        mlflow.log_artifact(str(model_file), artifact_path="models")
        model_file.unlink()

        run_id = active_run.info.run_id

    return run_id


def test_evaluate_promotion_success() -> None:
    criteria = PromotionCriteria(max_holdout_brier=0.2, max_holdout_log_loss=0.69, max_cv_mean_brier=0.21)
    decision = evaluate_promotion(
        {"holdout_brier": 0.18, "holdout_log_loss": 0.65, "cv_mean_brier": 0.20},
        criteria,
    )

    assert decision == PromotionDecision(promote=True, rationale="All promotion criteria satisfied.")


def test_evaluate_promotion_missing_metric() -> None:
    criteria = PromotionCriteria(max_holdout_brier=0.2, max_holdout_log_loss=0.69)
    decision = evaluate_promotion({"holdout_brier": 0.18}, criteria)

    assert not decision.promote
    assert "Missing required metrics" in decision.rationale


@pytest.mark.parametrize(
    "holdout_brier, holdout_log_loss, expected", [(0.18, 0.65, True), (0.24, 0.65, False), (0.18, 0.72, False)],
)
def test_promote_run_applies_criteria(
    tracking_setup: tuple[str, Path],
    holdout_brier: float,
    holdout_log_loss: float,
    expected: bool,
) -> None:
    tracking_uri, data_dir = tracking_setup
    criteria = PromotionCriteria(max_holdout_brier=0.2, max_holdout_log_loss=0.7)

    run_id = _create_run(
        tracking_uri,
        data_dir,
        holdout_brier=holdout_brier,
        holdout_log_loss=holdout_log_loss,
        cv_mean_brier=None,
        artifact_contents="model-binary",
    )

    result = promote_run(
        run_id,
        tracking_uri=tracking_uri,
        data_dir=data_dir,
        model_name="baseline",
        artifact_path="models/model.joblib",
        criteria=criteria,
        stage="Production",
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    run_tags = client.get_run(run_id).data.tags
    decision_path = result.decision_path

    assert decision_path.exists()
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["promoted"] == expected

    if expected:
        assert result.promoted
        assert run_tags["promoted"] == "true"
        assert "model_id" in run_tags
        registered = client.get_registered_model("baseline")
        assert registered.name == "baseline"
        versions = client.get_latest_versions("baseline", stages=["Production"])
        assert versions and versions[0].current_stage == "Production"
    else:
        assert not result.promoted
        assert run_tags["promoted"] == "false"
        assert "model_id" not in run_tags
        with pytest.raises(MlflowException):
            client.get_registered_model("baseline")


