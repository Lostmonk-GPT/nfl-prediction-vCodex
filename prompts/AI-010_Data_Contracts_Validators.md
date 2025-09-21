# Prompt: [AI-010] Data Contracts/Validators

Persona: Data Harvester (Ingestion)

Objective
- Define input contracts and validators to assert required columns for schedules, PBP, and rosters/teams.

Context
- Depends on: [AI-007..AI-009] ingestion functions.
- Guard against upstream schema drift.

Deliverables
- `src/nfl_pred/ingest/contracts.py`:
  - Functions like `assert_schedule_contract(df)`, `assert_pbp_contract(df)`, `assert_roster_contract(df)`.
  - Clear error messages listing missing columns and sample present columns.

Constraints
- No external validation libs; simple checks.

Steps
- Enumerate required columns per source (min viable set used downstream).
- Implement assertion helpers; add docstrings.

Acceptance Criteria (DoD)
- Calling validators on typical datasets passes; removing a required column raises a helpful error.

Verification Hints
- Unit-like quick checks with synthetic DataFrames.

