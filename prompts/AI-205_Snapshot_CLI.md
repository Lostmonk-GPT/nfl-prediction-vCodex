# Prompt: [AI-205] Snapshot CLI

Persona: Timekeeper (Snapshot & Visibility)

Objective
- Add a `snapshot` CLI command to run the full pre-game freeze at a given timestamp.

Context
- Extends CLI from [AI-023].

Deliverables
- Extend `src/nfl_pred/cli.py`:
  - Command: `snapshot --at <iso8601>` running T-24h→T-60m or just the T-60m freeze if provided.
  - Outputs predictions with `snapshot_at` and logs stage execution.

Constraints
- Clear error on invalid timestamp; timezone awareness.

Steps
- Wire CLI to `snapshot/runner.py` and visibility helpers.

Acceptance Criteria (DoD)
- `snapshot --at` runs end-to-end and writes outputs; logs stages.

Verification Hints
- Dry-run with a historical timestamp and verify predictions rows created.

