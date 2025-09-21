# Prompt: [AI-403] Retrain Triggers

Persona: Referee (Evaluation & Monitoring)

Objective
- Implement retrain triggers: 4-week rolling Brier deterioration ≥10% vs baseline, PSI ≥0.2 on ≥5 features, rule-flag flip.

Context
- PRD Section 12.

Deliverables
- `src/nfl_pred/monitoring/triggers.py`:
  - Functions evaluating trigger conditions given metrics history, PSI summary, and rule flags.

Constraints
- Deterministic calculations; configurable thresholds via config.

Steps
- Implement checks and return a structured decision object (triggered flags and reasons).

Acceptance Criteria (DoD)
- On synthetic inputs, triggers flip when thresholds are crossed and remain off otherwise.

Verification Hints
- Parameterize thresholds for unit tests; assert edge cases around 10% boundary.

