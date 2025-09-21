# Prompt: [AI-009] Ingestion: Rosters/Team Info

Persona: Data Harvester (Ingestion)

Objective
- Ingest rosters and team info via `nflreadpy` and persist Parquet with metadata.

Context
- Depends on: [AI-002], [AI-003], [AI-005].
- Used in feature joins and travel logic.

Deliverables
- `src/nfl_pred/ingest/rosters.py`:
  - Functions `ingest_rosters(seasons: list[int]) -> Path` and `ingest_teams() -> Path`.
  - Adds `pulled_at`, `source`, `source_version`.

Constraints
- Keep as close to source as possible; minimal cleaning.

Steps
- Fetch rosters per season and teams once; write Parquet.

Acceptance Criteria (DoD)
- Parquet written; metadata included; functions return paths.

Verification Hints
- Inspect schema and a few rows; spot-check team IDs.
