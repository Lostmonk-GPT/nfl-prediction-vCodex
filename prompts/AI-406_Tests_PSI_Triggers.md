# Prompt: [AI-406] Tests: PSI/Triggers

Persona: Verifier (QA)

Objective
- Add tests covering PSI calculation and retrain trigger logic across boundary conditions.

Context
- Depends on: [AI-402], [AI-403].

Deliverables
- `tests/test_monitoring.py`:
  - PSI tests with controlled distributions; trigger tests with synthetic metric histories and rule flags.

Constraints
- No network; synthetic inputs only.

Steps
- Implement tests asserting exact PSI values for known cases and accurate trigger toggling around thresholds.

Acceptance Criteria (DoD)
- Tests pass; failures produce clear diagnostics on which threshold failed.

Verification Hints
- Include a case with PSI just below and just above 0.2 to ensure correct comparisons.

