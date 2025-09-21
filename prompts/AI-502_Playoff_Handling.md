# Prompt: [AI-502] Playoff Handling

Persona: Forecaster (Modeling)

Objective
- Add playoff flags and optionally support a separate fit or toggles affecting training/evaluation.

Context
- PRD Section 7 Training Horizon and Scope.

Deliverables
- `src/nfl_pred/features/playoffs.py`:
  - Function to flag postseason rows; expose config to include/exclude or handle separately.

Constraints
- Keep default behavior consistent with MVP (regular season focus) unless configured.

Steps
- Implement playoff flagging and provide integration points in training pipeline.

Acceptance Criteria (DoD)
- Playoff rows flagged; pipelines can filter or include based on config.

Verification Hints
- Spot-check known postseason weeks are flagged.

