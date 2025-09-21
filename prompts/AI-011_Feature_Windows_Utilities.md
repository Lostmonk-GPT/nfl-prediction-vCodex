# Prompt: [AI-011] Feature Windows Utilities

Persona: Feature Crafter (Feature Engineering)

Objective
- Implement rolling window utilities (4/8/season-to-date) grouped by team/week with `asof_ts` filtering capability.

Context
- Depends on: [AI-008] PBP exists; [AI-024] visibility interface later.
- Windows underpin team-week aggregates.

Deliverables
- `src/nfl_pred/features/windows.py`:
  - Helpers to compute rolling means/rates by team across specified lookbacks.
  - Accept parameters: `group_keys`, `order_key (week/date)`, `window_lengths`, and optional `asof_ts` filter.

Constraints
- Efficient operations (use pandas or polars idioms).
- Correct boundary behavior at start of season (partial windows allowed).

Steps
- Implement generic rolling utility; add docstrings and examples.

Acceptance Criteria (DoD)
- Given synthetic input, windows of 4/8/season produce expected outputs per team.
- `asof_ts` filter excludes rows after cutoff.

Verification Hints
- Small synthetic DataFrame tests; compare against hand-calculated results.

