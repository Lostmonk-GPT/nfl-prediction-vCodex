# Prompt: [AI-603] MLflow Hygiene

Persona: Scribe (Docs & ADR)

Objective
- Standardize MLflow run tags and retention policies (snapshot, season/week, lineage), and ensure consistent experiment organization.

Context
- Supports traceability and promotion in [AI-405].

Deliverables
- `src/nfl_pred/audit/mlflow_utils.py`:
  - Helpers to set tags (`snapshot_at`, `season`, `week`, `model_id`, `promoted`), and to clean/prune runs by policy.

Constraints
- Local MLflow; no destructive ops by default.

Steps
- Implement tagging utilities; document usage in training/predict pipelines.

Acceptance Criteria (DoD)
- New runs carry standardized tags; optional retention function flags old runs per policy.

Verification Hints
- Inspect runs and confirm tag presence/values.

