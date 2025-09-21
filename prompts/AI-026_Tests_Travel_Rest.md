# Prompt: [AI-026] Unit Tests: Travel/Rest

Persona: Verifier (QA)

Objective
- Add tests validating haversine distances, rest-day rules, and neutral site handling.

Context
- Depends on: [AI-013] schedule meta; [AI-014] travel.

Deliverables
- `tests/test_travel_rest.py`:
  - Synthetic venues with known coordinates to test distances.
  - Cases for week 1 (no prior), short week, neutral site.

Constraints
- No network; fixed inputs.

Steps
- Construct minimal inputs; compute features; assert expected outputs.

Acceptance Criteria (DoD)
- Tests pass and cover key branches (neutral site, opener, short week).

Verification Hints
- Add tolerance for floating-point distance comparisons.

