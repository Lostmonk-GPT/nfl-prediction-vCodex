# Prompt: [AI-206] Historical Replay Test

Persona: Verifier (QA)

Objective
- Add test to replay a historical game at `asof_ts = T-60m` and assert no post-cutoff reads.

Context
- PRD Appendix A Enforcement and Outputs sections.

Deliverables
- `tests/test_snapshot_replay.py`:
  - Fixture or synthetic dataset with timestamps before/after cutoff.
  - Assertions that post-cutoff data is excluded and outputs are deterministic.

Constraints
- No network; use fixtures/synthetic data.

Steps
- Compose a minimal pipeline slice to exercise visibility filters and snapshot runner.

Acceptance Criteria (DoD)
- Test passes; failures indicate exactly which source leaked post-cutoff data.

Verification Hints
- Include at least one injury record after cutoff to verify exclusion.

