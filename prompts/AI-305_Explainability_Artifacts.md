# Prompt: [AI-305] Explainability Artifacts

Persona: Forecaster (Modeling)

Objective
- Persist SHAP values/plots to disk and MLflow; link to predictions and weeks.

Context
- Builds on: [AI-304] SHAP utilities; [AI-019] MLflow logging.

Deliverables
- `src/nfl_pred/explain/artifacts.py`:
  - Helpers to save arrays/plots, record metadata (season/week/model_id), and log to MLflow.

Constraints
- Storage under `data/artifacts/` with deterministic naming.

Steps
- Implement save/log functions; document paths and naming conventions.

Acceptance Criteria (DoD)
- Artifacts saved and discoverable by season/week/model_id; MLflow shows linked assets.

Verification Hints
- Inspect artifact folder structure and MLflow UI entries.

