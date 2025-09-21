# Prompt: [AI-021] Picks and Confidence

Persona: Forecaster (Modeling)

Objective
- Implement pick selection and confidence tiers from predicted probabilities.

Context
- Depends on: [AI-020] predictions.
- Tiers: Strong ≥0.65; Lean 0.55–0.65; Pass <0.55.

Deliverables
- `src/nfl_pred/picks.py`:
  - Function to add `pick` (team with max prob) and `confidence` (Strong/Lean/Pass) to predictions.

Constraints
- Deterministic tie-handling (document policy if 0.5 exactly).

Steps
- Map probability thresholds to labels; append to predictions table.

Acceptance Criteria (DoD)
- Each prediction row has a `pick` and a `confidence` consistent with thresholds.

Verification Hints
- Synthetic checks for boundary values (0.55, 0.65).

