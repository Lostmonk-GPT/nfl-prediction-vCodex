# Prompt: [AI-103] NWS Client

Persona: WX Ops (Stadium & Weather)

Objective
- Implement an HTTP client for NWS forecasts: `/points/{lat,lon}` → `/gridpoints/{wfo}/{x},{y}/forecast` (or hourly), with caching and unit normalization.

Context
- PRD Section 3 and Appendix A visibility.
- Use nearest station within 10 miles (computed from gridpoint metadata when applicable).

Deliverables
- `src/nfl_pred/weather/nws_client.py`:
  - Functions: `point_metadata(lat, lon)`, `gridpoint_forecast(wfo, x, y, hourly=False)`.
  - Caching/backoff; persist raw JSON via storage helper (see [AI-106]).
  - Unit normalization (e.g., temps F→C if desired, wind units consistent).

Constraints
- No network in tests; design for injected transport or fixture loading.
- Respect API etiquette (User-Agent); handle rate limiting gracefully.

Steps
- Implement client with `requests`; add simple TTL cache layer.

Acceptance Criteria (DoD)
- Given fixtures, client parses and returns normalized fields; caches repeated calls.

Verification Hints
- Test with recorded sample responses (fixtures) and confirm consistent units.

