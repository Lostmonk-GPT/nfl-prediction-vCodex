# Prompt: [AI-401] Expanded Evaluation Reports

Persona: Referee (Evaluation & Monitoring)

Objective
- Produce reports by week, season-to-date, rolling 4-week, and favorite/underdog slices.

Context
- Extends [AI-022] metrics module.

Deliverables
- `src/nfl_pred/reporting/expanded.py`:
  - Functions to aggregate metrics across specified windows and slices; persist to `reports` table and plots.

Constraints
- Consistent binning and slice definitions; deterministic windows.

Steps
- Implement aggregations and plotting helpers.

Acceptance Criteria (DoD)
- Reports generated for sample data; plots saved; tables updated.

Verification Hints
- Visual sanity check of trends and favorite vs underdog splits.

