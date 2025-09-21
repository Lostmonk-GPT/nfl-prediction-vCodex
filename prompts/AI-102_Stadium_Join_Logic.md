# Prompt: [AI-102] Stadium Join Logic

Persona: WX Ops (Stadium & Weather)

Objective
- Join schedules to the authoritative stadium table and resolve conflicts (use authority when differing from schedule roof/surface).

Context
- Depends on: [AI-101] stadium reference; [AI-007] schedules.

Deliverables
- `src/nfl_pred/features/stadium_join.py`:
  - Join function producing per-game venue metadata: `roof`, `surface`, `tz`, `lat`, `lon`, `neutral_site`.
  - Conflict resolution policy documented.

Constraints
- Deterministic override rules; log any unresolved mismatches.

Steps
- Implement join on venue/team and date; apply authoritative fields; emit warnings on mismatches.

Acceptance Criteria (DoD)
- Output includes authoritative fields; mismatches counted and logged.

Verification Hints
- Create a synthetic mismatch case to confirm override behavior.

