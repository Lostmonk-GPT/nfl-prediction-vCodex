# Prompt: [AI-104] Meteostat Client

Persona: WX Ops (Stadium & Weather)

Objective
- Integrate Meteostat for historical weather backfill near venues (≤10 miles) and normalize outputs.

Context
- PRD Section 3 and Appendix A; used for history/calibration and missing forecasts.

Deliverables
- `src/nfl_pred/weather/meteostat_client.py`:
  - Station selection via `Stations.nearby` or `Point` with distance calculation.
  - Fetch daily/hourly data; normalize units; return structured records.
  - Persist raw payloads via storage helper (see [AI-106]).

Constraints
- No network in tests; allow fixture injection.

Steps
- Implement wrappers using `meteostat` library; include unit conversions.

Acceptance Criteria (DoD)
- Given fixtures, selects nearest station within 10 miles and returns normalized fields.

Verification Hints
- Unit tests with synthetic lat/lon verifying station selection logic.

