# Prompt: [AI-006] DuckDB Schemas

Persona: Builder (Implementation Engineer)

Objective
- Define initial DuckDB schemas for features, predictions, reports, and runs metadata, with canonical keys.

Context
- Depends on: [AI-005] client exists; [AI-003] config with db path.
- Keys: `season`, `week`, `game_id`, `snapshot_at` (`asof_ts`).

Deliverables
- `src/nfl_pred/storage/schema.sql` including DDL for:
  - `features(...)`
  - `predictions(game_id, season, week, asof_ts, p_home_win, p_away_win, pick, confidence, model_id, snapshot_at)`
  - `reports(season, week, asof_ts, metric, value, snapshot_at)`
  - `runs_meta(run_id, model_id, created_at, params_json, metrics_json)`
- Optional indices on `(season, week, game_id, snapshot_at)`.

Constraints
- Use appropriate types (INTEGER, DOUBLE, VARCHAR, TIMESTAMP).
- Keep nullable vs non-nullable sensible for MVP.

Steps
- Write DDL with comments; provide a helper in `duckdb_client` to apply schema (optional).

Acceptance Criteria (DoD)
- Schema file exists and is valid SQL.
- Applying DDL in a fresh DB creates all tables without error.

Verification Hints
- Connect to a temp DuckDB and `INSTALL/LOAD` not required; just execute DDL.

