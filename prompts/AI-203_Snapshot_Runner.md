# Prompt: [AI-203] Snapshot Runner

Persona: Timekeeper (Snapshot & Visibility)

Objective
- Implement a snapshot runner for T-24h, T-100m, T-80–75m, and T-60m; write `snapshot_at` values and freeze outputs.

Context
- PRD Appendix A; snapshot timing drives visibility enforcement and outputs.

Deliverables
- `src/nfl_pred/snapshot/runner.py`:
  - Orchestrates data refresh (injuries/rosters as applicable) per snapshot stage.
  - Invokes feature builders with `asof_ts`; at T-60m produces predictions and picks.

Constraints
- Deterministic; log stages and timestamps; no network in tests.

Steps
- Define CLI-friendly entrypoint; implement stage sequence and persistence of `snapshot_at`.

Acceptance Criteria (DoD)
- Running with a specified `asof_ts` reproduces the same outputs across runs.

Verification Hints
- Dry-run on a past game with fixed `asof_ts` and compare hashes of outputs.

