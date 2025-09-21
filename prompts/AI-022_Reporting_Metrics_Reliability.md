# Prompt: [AI-022] Reporting: Metrics + Reliability

Persona: Referee (Evaluation & Monitoring)

Objective
- Compute Brier score, log-loss, and a reliability curve; write results to reports and save plots as artifacts.

Context
- Depends on: [AI-019] training outputs or [AI-020] predictions.

Deliverables
- `src/nfl_pred/reporting/metrics.py`:
  - Functions to compute metrics by week and overall; reliability binning and plot generation.
  - Persist metrics into `reports` and plots to artifacts dir/MLflow.

Constraints
- Deterministic binning (fixed bins) for reliability.
- Keep plotting minimal; matplotlib OK.

Steps
- Implement metric functions; add simple plotting helper.

Acceptance Criteria (DoD)
- Metrics computed on a small sample; reliability plot saved.

Verification Hints
- Confirm calibration bin averages monotonically increase for a reasonable model.

