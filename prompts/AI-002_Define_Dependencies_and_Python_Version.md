# Prompt: [AI-002] Define Dependencies and Python Version

Persona: Builder (Implementation Engineer)

Objective
- Declare Python version and core dependencies in `pyproject.toml` to support ingestion, storage, modeling, tracking, and CLI.

Context
- Depends on: [AI-001] scaffold present.
- PRD/Plan/Backlog/AGENTS apply. Tests must run offline.
- Non-goals: No code changes beyond dependency declaration; no lockfile or installer config yet.

Deliverables
- Update `pyproject.toml`:
  - Python: `3.11`
  - Deps (initial): `nflreadpy`, `pandas` or `polars`, `duckdb`, `pyarrow`, `scikit-learn`, `xgboost` or `lightgbm`, `mlflow`, `shap`, `requests`, `meteostat`, `typer` (or `argparse`).
  - Optional: `matplotlib`/`seaborn` for plots.

Constraints
- Do not install packages; only declare.
- Keep dependency set minimal to satisfy MVP tasks.
- Prefer stable, widely used versions; avoid pre-releases.

Steps
- Edit `pyproject.toml` `[project]` `dependencies` list.
- Add optional `urls` and `scripts` placeholders (no CLI wiring yet).
- Validate file is valid TOML.

Acceptance Criteria (DoD)
- `pyproject.toml` includes Python `>=3.11` and listed dependencies.
- No dev-only tools included yet (linters/formatters/poetry plugins).
- File parses as TOML.

Verification Hints
- Run: `python -c "import tomllib,sys; tomllib.load(open('pyproject.toml','rb')); print('ok')"`
