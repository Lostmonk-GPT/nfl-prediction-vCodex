# Prompt: [AI-012] Core Team-Week Features from PBP

Persona: Feature Crafter (Feature Engineering)

Objective
- Compute core team-level weekly aggregates from PBP: EPA/play, early-down EPA, success rate, pass/run rates, play-action, shotgun, no-huddle, sack rate, explosive rates, penalties, ST EPA.

Context
- Depends on: [AI-008] PBP; [AI-011] rolling utils.
- Map to nflfastR/nflverse columns as per PRD.

Deliverables
- `src/nfl_pred/features/team_week.py`:
  - Functions to aggregate per team-week and compute rolling windows (4/8/season).
  - Output schema documented in module docstring.

Constraints
- Ensure no future leakage (use weeks up to current only).
- Handle missing weeks/bye gracefully.

Steps
- Define per-play filters (early downs, explosive definitions).
- Aggregate per team-week; then compute rolling features via windows utils.

Acceptance Criteria (DoD)
- Produces a DataFrame keyed by `season, week, team` with specified metrics.
- Rolling features align with definitions; spot-check a team-week manually.

Verification Hints
- Compare basic counts/rates against known season stats for a sample team.

