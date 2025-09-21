# Prompt: [AI-702] Caching Layer

Persona: Builder (Implementation Engineer)

Objective
- Provide a TTL-based caching utility for HTTP calls with persistent storage and key hashing.

Context
- Supports [AI-103] NWS and [AI-104] Meteostat; no network in tests.

Deliverables
- `src/nfl_pred/utils/cache.py`:
  - API: `get_or_fetch(key: str, fetch_fn, ttl_seconds: int) -> Any` backed by file storage under `data/cache/`.

Constraints
- Deterministic key hashing; include versioning in keys to avoid collisions.

Steps
- Implement file-based cache with metadata and TTL enforcement.

Acceptance Criteria (DoD)
- Subsequent calls within TTL return cached results; after TTL, refresh occurs.

Verification Hints
- Unit-like checks with a dummy `fetch_fn` and short TTL.

