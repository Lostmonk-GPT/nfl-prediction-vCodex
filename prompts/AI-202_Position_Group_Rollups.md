# Prompt: [AI-202] Position-Group Rollups

Persona: Feature Crafter (Feature Engineering)

Objective
- Build counts of DNP/LP/FP by position group for snapshot week.

Context
- Depends on: [AI-201] injuries ingestion.
- Used as pre-game availability features.

Deliverables
- `src/nfl_pred/features/injury_rollups.py`:
  - Functions to group by team/position group and snapshot week to produce counts.

Constraints
- Define position groups mapping (QB, RB, WR, TE, OL, DL, LB, DB, ST).

Steps
- Implement grouping and counting; document null policy when missing.

Acceptance Criteria (DoD)
- Returns rollups per team/week with counts by group.

Verification Hints
- Synthetic test: inject a few injury records and verify group counts.

