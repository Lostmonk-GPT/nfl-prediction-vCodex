# Prompt: [AI-108] Docs Update for Weather

Persona: Scribe (Docs & ADR)

Objective
- Update README/docs describing NWS/Meteostat usage, caching, and indoor policy.

Context
- Depends on: [AI-103..AI-106].

Deliverables
- Update `README.md` with a Weather section covering:
  - Endpoints used, unit normalization, caching location, and TTL.
  - Indoor/closed roof handling (null/zero policy).

Constraints
- Keep concise; include links to relevant modules.

Steps
- Draft section and ensure terminology matches code.

Acceptance Criteria (DoD)
- Documentation present; a reader understands how weather is incorporated and cached.

Verification Hints
- Ensure alignment with PRD Section 3 and Appendix A.

