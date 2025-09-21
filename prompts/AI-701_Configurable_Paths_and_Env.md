# Prompt: [AI-701] Configurable Paths and Env

Persona: Builder (Implementation Engineer)

Objective
- Ensure all paths/URIs are configurable via YAML/env; extend config and loader accordingly.

Context
- Depends on: [AI-003] config base; applies across modules.

Deliverables
- Update `configs/default.yaml` and `src/nfl_pred/config.py` to include:
  - Paths (raw data dir, artifacts dir), DuckDB path, MLflow tracking URI, cache TTLs.
  - Env overrides with `NFLPRED__` prefix across nested keys.

Constraints
- Backward compatible with existing keys; provide defaults.

Steps
- Add fields, update loader, and document new keys in docstrings.

Acceptance Criteria (DoD)
- Modules can read config values without code changes; env overrides work across nested namespaces.

Verification Hints
- Set env vars and confirm effective config reflects overrides.

