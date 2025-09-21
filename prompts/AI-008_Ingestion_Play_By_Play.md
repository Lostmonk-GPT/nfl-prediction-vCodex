# Prompt: [AI-008] Ingestion: Play-by-Play

Persona: Data Harvester (Ingestion)

Objective
- Ingest play-by-play (PBP) per season via `nflreadpy` and persist Parquet with ingestion metadata.

Context
- Depends on: [AI-002], [AI-003], [AI-005].
- Large tables; write partitioned Parquet `data/raw/pbp_YYYY.parquet`.

Deliverables
- `src/nfl_pred/ingest/pbp.py`:
  - Function `ingest_pbp(seasons: list[int]) -> list[Path]`.
  - Adds `pulled_at`, `source`, `source_version` columns.
  - Optionally register views in DuckDB.

Constraints
- Avoid heavy transforms; keep raw structure.
- Tests offline with small fixture files (later tasks provide fixtures).

Steps
- Loop per season, fetch PBP, write Parquet per season.
- Return list of written paths.

Acceptance Criteria (DoD)
- Parquet per season with metadata present.
- No exceptions for typical seasons.

Verification Hints
- Read one season and verify core columns (e.g., `game_id`, `posteam`, `epa`).
