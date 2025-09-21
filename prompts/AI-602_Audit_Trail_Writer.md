# Prompt: [AI-602] Audit Trail Writer

Persona: Scribe (Docs & ADR)

Objective
- Persist an audit trail (snapshot timestamps, upstream dataset versions, model hash, code version, feature spec checksum, input row hashes) to DuckDB.

Context
- PRD Appendix A Audit Trail and Section 13 Reproducibility.

Deliverables
- `src/nfl_pred/audit/trail.py`:
  - Functions to gather and write audit records keyed by `season, week, snapshot_at` and `model_id`.

Constraints
- Deterministic hashing; avoid heavy I/O.

Steps
- Implement helpers to compute hashes and write to an `audit` table.

Acceptance Criteria (DoD)
- Running at T-60m writes a complete audit record; stable across reruns with same inputs.

Verification Hints
- Re-run with identical data and confirm identical hashes.

