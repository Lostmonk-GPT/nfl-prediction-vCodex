# Prompt: [AI-019] Training Pipeline

Persona: Forecaster (Modeling)

Objective
- Implement a training pipeline that reads features, performs CV evaluation, calibrates final model, logs to MLflow, and persists artifacts.

Context
- Depends on: [AI-015] features; [AI-016] splits; [AI-017] baseline; [AI-018] calibration; [AI-003] config; [AI-004] logging.

Deliverables
- `src/nfl_pred/pipeline/train.py`:
  - CLI-friendly function to load features and run CV (Brier/log-loss).
  - Train final model on full training portion, apply calibration on holdout, log MLflow params/metrics/artifacts; save model under `data/models/`.

Constraints
- Reproducibility: fixed seeds, log config snapshot.
- Avoid downloading data; assume features pre-built.

Steps
- Wire components, compute metrics, produce simple reliability plot for artifacts.

Acceptance Criteria (DoD)
- Running the pipeline on a small sample produces MLflow run with metrics and a saved model artifact.

Verification Hints
- Check `mlruns/` or configured tracking URI for run; inspect saved model path.

