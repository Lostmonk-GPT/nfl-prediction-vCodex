# Prompt: [AI-023] CLI Entrypoints

Persona: Shipwright (Release & Ops)

Objective
- Provide a CLI with commands `ingest`, `build-features`, `train`, `predict`, `report`.

Context
- Depends on: earlier ingestion, features, model, reporting modules.
- Use `typer` (or `argparse` if avoiding dependency).

Deliverables
- `src/nfl_pred/cli.py`:
  - Commands:
    - `ingest --seasons 2022 2023`
    - `build-features --seasons 2022 2023`
    - `train --config configs/default.yaml`
    - `predict --week 5 --season 2024`
    - `report --season 2024 --week 5`

Constraints
- Clear help messages; default config path.
- Console logging via `setup_logging`.

Steps
- Wire commands to call underlying functions; pass parsed args.

Acceptance Criteria (DoD)
- `python -m nfl_pred.cli --help` shows commands; each subcommand runs to completion on stub data.

Verification Hints
- Dry-run commands with minimal/no-op where feasible.

