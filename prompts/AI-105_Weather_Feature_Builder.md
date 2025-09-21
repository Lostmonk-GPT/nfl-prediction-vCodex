# Prompt: [AI-105] Weather Feature Builder

Persona: WX Ops (Stadium & Weather)

Objective
- Build weather features from NWS (forecast) and Meteostat (historical) for outdoor/open/retractable-open games; null/zero for indoor/closed.

Context
- Depends on: [AI-101] stadium authority; [AI-103] NWS; [AI-104] Meteostat; [AI-102] joined venue metadata.

Deliverables
- `src/nfl_pred/features/weather.py`:
  - Create `wx_temp`, `wx_wind`, `precip` (and any buckets/transform per plan).
  - Apply indoor handling policy and persist normalized features.

Constraints
- Respect visibility timing; forecast usage for pre-game only.

Steps
- Merge venue metadata, call clients (or load fixtures), compute features; add to feature matrix.

Acceptance Criteria (DoD)
- Outdoor games show non-null weather features; indoor games are null/zero per policy.

Verification Hints
- Unit tests with fixtures (see [AI-107]) cover indoor/outdoor cases.

