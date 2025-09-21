# Prompt: [AI-018] Calibration (Platt)

Persona: Forecaster (Modeling)

Objective
- Wrap the baseline model with sigmoid (Platt) calibration using a held-out fold.

Context
- Depends on: [AI-017] baseline model; [AI-016] splits.
- Later [AI-303] will add isotonic vs Platt selection.

Deliverables
- `src/nfl_pred/model/calibration.py`:
  - Calibrator class with `fit(base_model, X_valid, y_valid)` and `predict_proba(X)` applying sigmoid mapping.
  - Persist calibration parameters with the model.

Constraints
- Avoid leakage: calibration only on validation or dedicated holdout.

Steps
- Fit logistic regression on model logits vs labels; apply to base probs.

Acceptance Criteria (DoD)
- Calibrated probabilities shift reliability closer to diagonal on a sample.

Verification Hints
- Plot pre/post calibration reliability (simple binning); expect improvement.

