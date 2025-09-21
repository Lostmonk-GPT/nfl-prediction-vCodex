# Prompt: [AI-013] Rest Days and Kickoff Bucket

Persona: Feature Crafter (Feature Engineering)

Objective
- Derive `rest_days`, `short_week`, `kickoff_bucket`, and `home_away` from schedules.

Context
- Depends on: [AI-007] schedules.
- Kickoff buckets per plan (e.g., early/late/SN/MN).

Deliverables
- `src/nfl_pred/features/schedule_meta.py`:
  - Functions computing rest vs prior game, short-week boolean, kickoff time buckets, and home/away flag.

Constraints
- Handle bye weeks and neutral site games correctly.
- Timezone handling per stadium `tz` added later (Phase 1); for MVP use schedule timestamps directly.

Steps
- Compute previous game date per team; diff in days.
- Classify kickoff time into buckets.

Acceptance Criteria (DoD)
- Returns expected columns for all games in a season; fields sensible on edges (week 1, bye weeks).

Verification Hints
- Spot-check Thursday games as short weeks; MNF in correct bucket.

