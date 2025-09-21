# Prompt: [AI-027] Unit Tests: Model Split and Metrics

Persona: Verifier (QA)

Objective
- Add tests verifying forward-chaining split ordering and basic metric computations.

Context
- Depends on: [AI-016] splits; [AI-022] metrics utilities.

Deliverables
- `tests/test_model_split_metrics.py`:
  - Synthetic week sequences to assert split monotonicity.
  - Simple probability/label arrays to validate Brier and log-loss calculations.

Constraints
- No network; deterministic arrays.

Steps
- Write minimal tests covering edge cases (few weeks, min train size).

Acceptance Criteria (DoD)
- Tests pass; split generator produces expected folds; metrics match hand calculations.

Verification Hints
- Compare Brier/log-loss with known formulas on tiny arrays.

