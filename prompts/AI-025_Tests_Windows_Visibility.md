# Prompt: [AI-025] Unit Tests: Windows/Visibility

Persona: Verifier (QA)

Objective
- Add tests for rolling windows correctness and week-level visibility filtering.

Context
- Depends on: [AI-011] windows, [AI-024] visibility.

Deliverables
- `tests/test_windows_visibility.py`:
  - Synthetic datasets for teams over weeks.
  - Assertions for rolling aggregates (4/8) and exclusion of post-week rows by proxy.

Constraints
- No network; tests self-contained.

Steps
- Build small DataFrames inline; assert expected numeric results.

Acceptance Criteria (DoD)
- Tests pass locally; failures are actionable with clear messages.

Verification Hints
- Run: `pytest -k windows_visibility` once test scaffold exists in later tasks.

