# Prompt: [AI-106] Weather Artifacts and Metadata

Persona: WX Ops (Stadium & Weather)

Objective
- Persist raw API responses and call metadata for NWS/Meteostat with caching TTL and provenance.

Context
- Supports reproducibility and debugging per PRD.
- Used by [AI-103] and [AI-104].

Deliverables
- `src/nfl_pred/weather/storage.py`:
  - Helpers to save/load raw JSON with keys derived from endpoint and params.
  - Attach metadata: `called_at`, `source`, `version`, `ttl_expires_at`.

Constraints
- File-based storage under `data/weather/` by default.

Steps
- Implement simple file store with hashed keys and manifest index.

Acceptance Criteria (DoD)
- Raw payloads and metadata persist and can be reloaded deterministically.

Verification Hints
- Simulate save/load roundtrip with small JSON objects.

