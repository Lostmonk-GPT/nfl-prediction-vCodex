# Prompt: [AI-405] MLflow Model Registry Hooks

Persona: Shipwright (Release & Ops)

Objective
- Promote models that meet thresholds; tag runs with lineage and persist promotion info.

Context
- Depends on: [AI-019] MLflow runs; [AI-401..AI-404] metrics/decisions.

Deliverables
- `src/nfl_pred/registry/promote.py`:
  - Functions to evaluate promotion criteria and register/tag model artifacts accordingly.

Constraints
- Local MLflow tracking (no external services assumed).

Steps
- Implement promotion rules; tag runs with `promoted=true`, `model_id`, and rationale.

Acceptance Criteria (DoD)
- When criteria met, model is tagged/promoted; otherwise, rationale recorded.

Verification Hints
- Simulate a run with metrics over thresholds and verify tags/registry entries.

