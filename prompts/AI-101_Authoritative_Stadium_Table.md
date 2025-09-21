# Prompt: [AI-101] Authoritative Stadium Table

Persona: WX Ops (Stadium & Weather)

Objective
- Create an authoritative stadium reference with columns: `venue`, `team(s)`, `lat`, `lon`, `tz`, `altitude`, `surface`, `roof` (indoors/dome/open/retractable), `neutral_site`.

Context
- PRD Section 3; supersedes schedule roof/surface when conflicts arise.
- Used by travel, weather, and home-field logic.

Deliverables
- `data/ref/stadiums.csv` populated with current NFL venues (seed with known data; editable).
- `src/nfl_pred/ref/stadiums.py` helper to load and validate the table.

Constraints
- Ensure `tz` is valid IANA zone; lat/lon decimal degrees.
- Allow multiple teams per venue (e.g., shared stadiums) as a mapping.

Steps
- Draft CSV schema and populate baseline entries; add loader with validation checks.

Acceptance Criteria (DoD)
- CSV exists with all active venues; loader returns a validated DataFrame.

Verification Hints
- Spot-check a few stadiums (lat/lon, roof type) against public sources.

