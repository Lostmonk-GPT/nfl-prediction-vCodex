# Prompt: [AI-014] Travel Features

Persona: Feature Crafter (Feature Engineering)

Objective
- Compute travel features: haversine miles from previous venue to current, `days_since_last`, and `neutral_site` handling.

Context
- Depends on: [AI-007] schedules; later Phase 1 adds authoritative stadium lat/lon.
- MVP can approximate venue coordinates via schedule/teams.

Deliverables
- `src/nfl_pred/features/travel.py`:
  - Haversine distance function.
  - Feature builder adding `travel_miles`, `days_since_last`, `neutral_site` per game/team.

Constraints
- Ensure neutral-site games not counted as home advantage in travel.
- Handle missing prior game (week 1) with null/zero policy per plan.

Steps
- Determine previous venue per team; compute distance and days diff.

Acceptance Criteria (DoD)
- Outputs sensible miles/days for back-to-back games; null/zero on season openers.

Verification Hints
- Spot-check known long trips (e.g., west-to-east coast games).

