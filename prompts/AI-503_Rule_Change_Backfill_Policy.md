# Prompt: [AI-503] Rule-Change Backfill Policy

Persona: Scribe (Docs & ADR)

Objective
- Document season weighting/inclusion around rule transitions and how historical data is treated.

Context
- PRD Section 4 and 7; affects training datasets and evaluation comparability.

Deliverables
- `docs/rule_change_policy.md`:
  - Policies for including/excluding seasons around rule flips; any weighting schemes.

Constraints
- Keep actionable; include examples.

Steps
- Draft policy options and chosen defaults; note alternatives in an ADR if needed.

Acceptance Criteria (DoD)
- Document present and clear; references PRD sections.

Verification Hints
- Ensure policy is compatible with pipeline configurations.

