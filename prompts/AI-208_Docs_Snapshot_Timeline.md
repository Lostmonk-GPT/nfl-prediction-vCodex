# Prompt: [AI-208] Docs: Snapshot Timeline

Persona: Scribe (Docs & ADR)

Objective
- Document the snapshot sequence (T-24h, T-100m, T-80–75m, T-60m), visibility rules, and outputs at freeze.

Context
- PRD Appendix A.

Deliverables
- `docs/snapshot_timeline.md`:
  - Sections: Objectives, Cutoffs, Allowed data per snapshot, Weather/Stadium visibility, Enforcement, Outputs, Audit trail.

Constraints
- Concise and operational; link to relevant modules (runner/visibility).

Steps
- Draft doc with bullet lists matching the PRD language.

Acceptance Criteria (DoD)
- File exists and clearly explains what runs at each snapshot and what data is allowed.

Verification Hints
- Cross-check definitions against PRD Appendix A.

