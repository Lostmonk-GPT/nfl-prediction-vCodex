# Prompt: [AI-001] Create Repository Scaffold

Persona: Builder (Implementation Engineer)

Objective
- Create the repository skeleton with required folders and minimal files so the package `nfl_pred` imports and the project can evolve per plan and backlog.

Context
- PRD: NFL_Game_Outcome_Prediction_PRD_2025-09-17.md
- Dev Plan: NFL_Game_Outcome_Prediction_Dev_Plan.md
- Backlog: AI_Task_Backlog.md
- AGENTS: AGENTS.md (follow guardrails: visibility, deterministic storage, offline tests)
- Non-goals: Do not add dependencies, model code, ingestion, or tests; handled in AI-002+.

Deliverables
- Directories: `src/`, `src/nfl_pred/`, `configs/`, `tests/`, `data/`
- Files:
  - `src/nfl_pred/__init__.py` (empty)
  - `pyproject.toml` (minimal build-system stub; no dependencies yet)
  - `.gitignore` (Python-appropriate ignores; keep directories via .gitkeep)
  - `Makefile` (placeholder “help” target only; no build/test commands yet)
  - `.gitkeep` in `configs/`, `tests/`, `data/`

Constraints
- Keep changes minimal and focused on scaffolding.
- Do not install packages or hit the network.
- Do not introduce CI or tooling decisions yet (linters/formatters/poetry) — those come in AI-002.
- Ensure paths and casing match exactly (Linux FS rules).

File Contents
- pyproject.toml (minimal)
  ```toml
  [build-system]
  requires = ["setuptools>=67", "wheel"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "nfl-prediction"
  version = "0.0.0"
  description = "NFL Game Outcome Prediction — project scaffold"
  authors = [{ name = "AI Builder", email = "" }]
  readme = "README.md"
  requires-python = ">=3.11"
  ```
- .gitignore (Python-focused)
  ```gitignore
  # Python
  __pycache__/
  *.py[cod]
  *.pyo
  *.pyd
  .Python
  .venv/
  venv/
  .pytest_cache/
  .mypy_cache/
  .ruff_cache/
  .coverage
  coverage.xml
  htmlcov/

  # IDE / OS
  .DS_Store
  .vscode/
  .idea/

  # Env / secrets
  .env
  .env.*

  # Data and models (retain folders via .gitkeep)
  data/*
  !data/.gitkeep

  # Artifacts
  dist/
  build/
  *.egg-info/
  ```
- Makefile (placeholder)
  ```make
  .PHONY: help
  help:
  	@echo "Project scaffold created. Targets will be added in later tasks."
  ```
- Placeholders: `configs/.gitkeep`, `tests/.gitkeep`, `data/.gitkeep`, and empty `src/nfl_pred/__init__.py`.

Steps
- Create directories exactly as listed.
- Add files with the specified contents.
- Do not modify PRD/plan/backlog files in this task.
- Validate with a tree listing and a quick Python import of `nfl_pred`.

Acceptance Criteria (DoD)
- The repo contains the exact directories and files listed.
- Importing `nfl_pred` does not error (empty package is importable).
- No dependencies added or installed.
- `.gitignore` excludes common Python clutter and ignores `data/*` while retaining `data/.gitkeep`.
- Makefile exists with a working `help` target.

Verification Hints
- Run: `python -c "import nfl_pred; print('ok')"`
- Run: `make help` to see the placeholder message.
- Confirm structure: `src/nfl_pred/__init__.py`, `configs/.gitkeep`, `tests/.gitkeep`, `data/.gitkeep`, `pyproject.toml`, `.gitignore`, `Makefile`.

