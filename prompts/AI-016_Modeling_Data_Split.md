# Prompt: [AI-016] Modeling: Data Split

Persona: Forecaster (Modeling)

Objective
- Implement forward-chaining cross-validation grouped by NFL week to prevent leakage.

Context
- Depends on: [AI-015] features table.
- Group by week across season(s); ensure chronological splits.

Deliverables
- `src/nfl_pred/model/splits.py`:
  - Function(s): `time_series_splits(df, group_col='week', n_splits=..., min_train_weeks=...)` yielding train/valid indices.
  - Document split construction with examples.

Constraints
- No future weeks in train relative to validation.
- Ensure teams from future weeks are excluded from train.

Steps
- Implement generator for folds; test on synthetic week sequences.

Acceptance Criteria (DoD)
- Splits cover all eligible weeks; no overlap in val across folds; strict chronology.

Verification Hints
- Print weeks in each fold to confirm monotonic progression.

