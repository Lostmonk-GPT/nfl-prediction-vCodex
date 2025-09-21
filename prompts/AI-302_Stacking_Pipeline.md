# Prompt: [AI-302] Stacking Pipeline

Persona: Forecaster (Modeling)

Objective
- Implement stacking: generate out-of-fold (OOF) probabilities from level-0 models and train a logistic meta-learner.

Context
- Depends on: [AI-301] models; [AI-016] splits.

Deliverables
- `src/nfl_pred/model/stacking.py`:
  - Functions to produce OOF predictions and train meta-learner; apply to full data for final model.

Constraints
- Avoid leakage: OOF strictly from validation folds.

Steps
- Generate OOF matrix; fit logistic meta; provide predict interface.

Acceptance Criteria (DoD)
- OOF predictions shape matches samples x models; meta-learner improves validation loss over individual models in a small test.

Verification Hints
- Compare log-loss between best base model and stacked model on a toy dataset.

