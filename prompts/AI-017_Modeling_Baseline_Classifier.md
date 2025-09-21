# Prompt: [AI-017] Modeling: Baseline Classifier

Persona: Forecaster (Modeling)

Objective
- Implement a baseline classifier (logistic regression or LightGBM) with standard preprocessing.

Context
- Depends on: [AI-015] features; [AI-016] splits.
- Calibrated later in [AI-018].

Deliverables
- `src/nfl_pred/model/baseline.py`:
  - Classifier wrapper with `fit(X,y)` and `predict_proba(X)`.
  - Handles categorical encoding if present; numeric scaling optional.

Constraints
- Keep input interface simple (X as DataFrame, y as Series/array).
- Ensure reproducibility via fixed seeds.

Steps
- Choose model (logistic for MVP or LightGBM if dependency present), implement training/predict.

Acceptance Criteria (DoD)
- Trains on a sample and outputs probabilities in [0,1].
- Reasonable runtime on small dataset.

Verification Hints
- Fit on synthetic data and verify metrics basics (accuracy/log-loss trend).

