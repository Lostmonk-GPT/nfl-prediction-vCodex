# Prompt: [AI-404] Monitoring CLI/Report

Persona: Referee (Evaluation & Monitoring)

Objective
- Add CLI/reporting that surfaces PSI, rolling metrics, and retrain trigger status as dashboard-ready JSON/plots.

Context
- Depends on: [AI-401], [AI-402], [AI-403].

Deliverables
- Extend `src/nfl_pred/cli.py` and add `src/nfl_pred/reporting/monitoring_report.py`:
  - Command `monitor --season 2024 --week 8` outputting JSON plus plots to artifacts.

Constraints
- No network; rely on local DuckDB and artifacts.

Steps
- Implement report builder combining metrics/PSI/trigger decisions; return file paths.

Acceptance Criteria (DoD)
- Running the command produces a JSON summary and saves plots; values align with underlying computations.

Verification Hints
- Manually inspect JSON keys and a few PSI values.

