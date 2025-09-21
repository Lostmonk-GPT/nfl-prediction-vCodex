# Prompt: [AI-301] Additional Level-0 Models

Persona: Forecaster (Modeling)

Objective
- Add ridge/logistic/GBDT models under a unified interface for stacking later.

Context
- Extends baseline in [AI-017].

Deliverables
- `src/nfl_pred/model/models.py`:
  - Wrappers for ridge, logistic, XGBoost/LightGBM with consistent `fit/predict_proba`.
  - Hyperparameter grids for quick search (lightweight).

Constraints
- Keep training times reasonable; small grids only.

Steps
- Implement wrappers and parameter dictionaries; document expected inputs.

Acceptance Criteria (DoD)
- Each model trains and predicts on a small dataset; interfaces are consistent.

Verification Hints
- Quick A/B on a sample to confirm outputs plausible.

