# Prompt: [AI-207] Injury Visibility Tests

Persona: Verifier (QA)

Objective
- Ensure only injury/participation records with `event_time ≤ asof_ts` are included in features.

Context
- Depends on: [AI-201], [AI-204].

Deliverables
- `tests/test_injury_visibility.py`:
  - Construct injury records around cutoff and assert feature rollups include/exclude correctly.

Constraints
- No network; synthetic data only.

Steps
- Build small input; run rollup; assert counts match expectation by cutoff.

Acceptance Criteria (DoD)
- Tests pass and would fail if visibility not enforced.

Verification Hints
- Include LP/FP/DNP variations to test grouping logic.

