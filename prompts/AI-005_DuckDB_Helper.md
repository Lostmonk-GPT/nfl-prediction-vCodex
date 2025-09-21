# Prompt: [AI-005] DuckDB Helper

Persona: Builder (Implementation Engineer)

Objective
- Provide a small helper for interacting with DuckDB: open connection, run queries, write DataFrames, and register Parquet.

Context
- Depends on: [AI-001] scaffold; [AI-003] config paths.
- DuckDB file path typically `data/nfl.duckdb` from config.

Deliverables
- `src/nfl_pred/storage/duckdb_client.py`:
  - Context manager class `DuckDBClient(db_path: str)`.
  - Methods: `read_sql(sql: str, params: dict|None) -> DataFrame`, `write_df(df, table: str, mode: str)`, `register_parquet(path: str, view: str)`.
  - Optional: `execute(sql: str)` for DDL.

Constraints
- Use `duckdb` Python API; prefer `pyarrow`/`pandas` interchange.
- Keep interface small and typed.

Steps
- Implement client with context manager and minimal methods.
- Handle table write modes: create/replace/append.

Acceptance Criteria (DoD)
- Can open DB, run a `SELECT 1`, write a tiny DataFrame to a temp table, and read it back.
- Register a Parquet file and query via a view.

Verification Hints
- Use an in-memory path `:memory:` for a quick smoke test.

