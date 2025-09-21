# Prompt: [AI-303] Calibration Selection

Persona: Forecaster (Modeling)

Objective
- Choose between isotonic and Platt calibration based on validation performance; integrate into pipeline.

Context
- Extends [AI-018] Platt; offers isotonic alternative.

Deliverables
- Extend `src/nfl_pred/model/calibration.py`:
  - Function to compare calibrators on a validation set (log-loss/Brier) and select the best.

Constraints
- Guard against overfitting with isotonic on tiny datasets; require minimum sample size.

Steps
- Implement selection routine and expose chosen calibrator.

Acceptance Criteria (DoD)
- On a sample, the selection picks the method with better validation metric and applies it.

Verification Hints
- Synthetic case where isotonic clearly outperforms Platt (nonlinear reliability) to validate selection.

