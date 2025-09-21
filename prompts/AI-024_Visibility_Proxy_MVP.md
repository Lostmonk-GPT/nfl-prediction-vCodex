# Prompt: [AI-024] Visibility Proxy (MVP)

Persona: Timekeeper (Snapshot & Visibility)

Objective
- Implement a coarse pre-game `asof_ts` proxy at the week level using scheduled kickoff to filter data.

Context
- Applies to MVP only; stricter snapshot timeline comes in Phase 2.
- Integrate with feature builders to exclude post-week data.

Deliverables
- `src/nfl_pred/visibility.py`:
  - Functions to compute a week-level `asof_ts` and filter dataframes by `event_time <= asof_ts` when available, else by `season/week`.

Constraints
- No historical inactives/injury timing in MVP; document approximation.

Steps
- Implement filtering helpers and document usage in module docstring.

Acceptance Criteria (DoD)
- Feature assembly respects the proxy and excludes future weeks.

Verification Hints
- Construct a small example with weeks 1–3 and confirm filters drop week 3 when targeting week 2.

