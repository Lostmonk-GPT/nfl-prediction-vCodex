# Prompt: [AI-501] Rule Flags

Persona: Feature Crafter (Feature Engineering)

Objective
- Add binary flags `kickoff_2024plus` and `ot_regular_2025plus` into the feature matrix.

Context
- PRD Section 4 Rule-Change Guards.

Deliverables
- `src/nfl_pred/features/rules.py`:
  - Functions to compute rule flags by season/date and append to features.

Constraints
- Deterministic activation timing; document boundary conditions.

Steps
- Implement simple date/season-based switches and join to features.

Acceptance Criteria (DoD)
- Features include rule flags with correct values for seasons before/after cutovers.

Verification Hints
- Spot-check a season before and after to ensure flips occur as expected.

