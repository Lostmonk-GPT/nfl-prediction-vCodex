# Prompt: [AI-015] Assemble MVP Feature Matrix

Persona: Feature Crafter (Feature Engineering)

Objective
- Join team-week PBP aggregates, schedule metadata, and travel features into a single features table for modeling.

Context
- Depends on: [AI-011..AI-014].
- Keys by `season, week, team` and join to `game_id`.

Deliverables
- `src/nfl_pred/features/build_features.py`:
  - Function to assemble features and write to DuckDB `features` table.
  - Includes minimal label (team win) for training rows.

Constraints
- No leakage: features computed only from prior/current week as per definitions.
- Consistent data types; document null policies.

Steps
- Join on keys; align home/away/team orientation; generate label from schedule results.
- Persist to DuckDB through client.

Acceptance Criteria (DoD)
- Features table exists with expected columns and row counts for a sample season.

Verification Hints
- Basic sanity metrics (non-null rates, distributions) and sample rows inspection.

