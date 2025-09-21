# Prompt: [AI-107] Weather Tests with Fixtures

Persona: Verifier (QA)

Objective
- Provide fixtures for NWS and Meteostat responses and tests for normalization, indoor nulling, and station selection.

Context
- Depends on: [AI-103], [AI-104], [AI-105].

Deliverables
- `tests/fixtures/nws/*.json` sample responses.
- `tests/test_weather.py` covering:
  - Unit normalization (temp, wind, precip) from fixtures.
  - Indoor handling sets features null/zero.
  - Nearest station logic ≤10 miles.

Constraints
- No network; all tests use fixtures.

Steps
- Add minimal JSON samples; write focused tests.

Acceptance Criteria (DoD)
- Tests pass and cover the three target behaviors.

Verification Hints
- Include at least one retractable-open vs closed case.

