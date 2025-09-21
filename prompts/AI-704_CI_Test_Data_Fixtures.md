# Prompt: [AI-704] CI Test Data Fixtures

Persona: Verifier (QA)

Objective
- Provide curated small Parquet/JSON fixtures so tests run offline deterministically.

Context
- Required by all ingestion/weather-related tests.

Deliverables
- `tests/fixtures/` folder with:
  - Tiny PBP, schedules, rosters, injuries Parquet files.
  - Weather API JSONs.

Constraints
- Keep files very small; include README describing contents and provenance.

Steps
- Create minimal but representative fixtures and document schemas.

Acceptance Criteria (DoD)
- Tests across the suite can run without network using these fixtures.

Verification Hints
- Temporarily block network and run tests to ensure no calls are made.

