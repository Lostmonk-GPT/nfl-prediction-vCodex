# Prompt: [AI-003] Base Config and Loader

Persona: Builder (Implementation Engineer)

Objective
- Provide a typed configuration loader that reads YAML and supports environment variable overrides.

Context
- Depends on: [AI-001] scaffold; [AI-002] deps declared.
- Config will hold paths (data, duckdb), MLflow URI, feature windows, etc.
- Non-goals: No business logic; no runtime env creation.

Deliverables
- `configs/default.yaml` with placeholders:
  - `paths: { data_dir: data, duckdb_path: data/nfl.duckdb }`
  - `mlflow: { tracking_uri: ./mlruns }`
  - `features: { windows: { short: 4, mid: 8 } }`
- `src/nfl_pred/config.py`:
  - Dataclasses representing the config schema.
  - Loader: `load_config(path: str | None) -> Config`
  - Env override (e.g., prefix `NFLPRED__` mapping to keys).

Constraints
- Keep schema small; extend in later tasks.
- Fail fast with clear error if YAML invalid.

Steps
- Draft dataclasses and YAML loader with env overlay.
- Provide utility to dump effective config for debugging.

Acceptance Criteria (DoD)
- Loading default YAML returns a typed `Config` instance.
- Environment variables with prefix override nested values.
- Basic unit-free validation (paths exist not required yet).

Verification Hints
- Run: `python -c "from nfl_pred.config import load_config; c=load_config(None); print(c)"`
- Set an env var (e.g., `NFLPRED__MLFLOW__TRACKING_URI=/tmp/mlruns`) and confirm override.

