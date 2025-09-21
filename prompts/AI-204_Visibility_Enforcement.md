# Prompt: [AI-204] Visibility Enforcement

Persona: Timekeeper (Snapshot & Visibility)

Objective
- Enforce `event_time ≤ asof_ts` filtering across all feature builders and inputs.

Context
- PRD Appendix A Enforcement section.

Deliverables
- `src/nfl_pred/snapshot/visibility.py`:
  - Common filter functions to apply to each source; wrappers/adapters for builders to accept `asof_ts`.

Constraints
- No post-cutoff reads; document any sources that lack precise event times.

Steps
- Implement filters; integrate with feature functions by optional parameter or context object.

Acceptance Criteria (DoD)
- Builders drop records after `asof_ts`; unit tests verify enforcement (see AI-206/207).

Verification Hints
- Replay a controlled dataset with timestamps straddling cutoff.

