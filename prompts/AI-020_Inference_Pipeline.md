# Prompt: [AI-020] Inference Pipeline

Persona: Forecaster (Modeling)

Objective
- Load the trained model, generate `p_home_win`/`p_away_win` per game, and write to the `predictions` table.

Context
- Depends on: [AI-019] model artifact; [AI-015] features for target week; [AI-003] config; [AI-005] DuckDB.

Deliverables
- `src/nfl_pred/pipeline/predict.py`:
  - Function to load model, select feature rows for upcoming games, produce probabilities, and persist to DuckDB `predictions`.
  - Include `model_id` and `snapshot_at` in outputs.

Constraints
- Ensure probabilities sum to 1 per game.
- No writing outside DuckDB and controlled artifacts dir.

Steps
- Load model; prepare feature matrix; compute probs; write outputs.

Acceptance Criteria (DoD)
- Predictions exist for a sample week with expected shapes and keys.

Verification Hints
- Query DuckDB for the week and inspect `p_home_win`, `p_away_win` ranges.

