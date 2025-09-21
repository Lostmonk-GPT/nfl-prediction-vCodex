# Prompt: [AI-201] Injuries Ingestion

Persona: Data Harvester (Ingestion)

Objective
- Ingest injuries and participation data with event timestamps, persisting to Parquet.

Context
- Prepares for snapshot visibility; exact publish times may be limited.

Deliverables
- `src/nfl_pred/ingest/injuries.py`:
  - Function to pull weekly injuries/participation; write Parquet with `event_time`, `pulled_at`, `source`, `source_version`.

Constraints
- Minimal cleaning; retain timestamps for visibility filtering.

Steps
- Fetch, add metadata, persist; log counts and date ranges.

Acceptance Criteria (DoD)
- Parquet written with timestamped records; function returns path(s).

Verification Hints
- Inspect a few rows for expected fields and times.

