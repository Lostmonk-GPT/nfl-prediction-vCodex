# Prompt: [AI-007] Ingestion: Schedules

Persona: Data Harvester (Ingestion)

Objective
- Pull season schedules via `nflreadpy` and persist as Parquet with ingestion metadata.

Context
- Depends on: [AI-002] deps; [AI-003] config (paths); [AI-005] DuckDB client (optional view registration).
- Columns and source per nflverse; validate via contracts in [AI-010].

Deliverables
- `src/nfl_pred/ingest/schedules.py`:
  - Function `ingest_schedules(seasons: list[int]) -> Path` writing `data/raw/schedules.parquet` (or partitioned by season).
  - Add metadata columns: `pulled_at`, `source = 'nflreadpy'`, `source_version` if available.
  - Optionally register in DuckDB as external table/view.

Constraints
- Do not perform transformations beyond light cleaning.
- No network in tests; provide fixture strategy (sample Parquet) for later.

Steps
- Call appropriate `nflreadpy` schedule function per season.
- Concatenate and write Parquet; log shape and columns.

Acceptance Criteria (DoD)
- Parquet file(s) created with expected columns and metadata fields.
- Function returns path; no exceptions on normal seasons.

Verification Hints
- Quick read back and count rows; sanity check a known season count.
