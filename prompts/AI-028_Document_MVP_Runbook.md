# Prompt: [AI-028] Document MVP Runbook

Persona: Scribe (Docs & ADR)

Objective
- Add a concise runbook describing how to run the MVP end-to-end on a past week.

Context
- Depends on: MVP modules and CLI ([AI-023]).
- Keep it minimal, actionable, and aligned with AGENTS.md guardrails.

Deliverables
- Update `README.md` (or add if missing):
  - Pre-requisites (Python, venv creation brief).
  - Commands: `ingest`, `build-features`, `train`, `predict`, `report` with example args.
  - Expected outputs and where to find them (DuckDB tables, MLflow runs, artifacts).

Constraints
- No long prose; prioritize commands and expected results.

Steps
- Draft sections and ensure commands map to CLI.

Acceptance Criteria (DoD)
- Runbook present; a new user can follow steps to reproduce an MVP run.

Verification Hints
- Cross-check commands against CLI help text.

