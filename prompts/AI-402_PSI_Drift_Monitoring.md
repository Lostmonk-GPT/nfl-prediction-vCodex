# Prompt: [AI-402] PSI Drift Monitoring

Persona: Referee (Evaluation & Monitoring)

Objective
- Implement Population Stability Index (PSI) monitoring on key features with alert threshold ≥0.2.

Context
- PRD Section 12 Monitoring; integrate with reports/alerts later.

Deliverables
- `src/nfl_pred/monitoring/psi.py`:
  - PSI computation utilities with fixed binning strategy and reference vs current datasets.

Constraints
- Deterministic bins; handle nulls consistently.

Steps
- Implement PSI formula; expose function returning PSI per feature and summary.

Acceptance Criteria (DoD)
- PSI computed on synthetic drifted data shows expected increases; thresholding works.

Verification Hints
- Create a shifted normal distribution example to validate PSI magnitude.

