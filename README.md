# nfl-prediction-vCodex

## Quick Start Runbook (AI-028)

### 1. Environment
- Python 3.11+
- Optional virtualenv (recommended):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
  ```
- Default configuration lives at `configs/default.yaml` (data under `data/`, MLflow at `./mlruns`).

### 2. MVP Flow
1. **Ingest raw nflverse datasets**
   ```bash
   nfl-pred ingest --seasons 2022 2023
   ```
   - Outputs: `data/raw/schedules.parquet`, `data/raw/pbp_<season>.parquet`, `data/raw/rosters.parquet`, `data/raw/teams.parquet`, DuckDB view `schedules_raw`.
2. **Assemble modeling features**
   ```bash
   nfl-pred build-features --seasons 2022 2023 --config configs/default.yaml
   ```
   - Outputs: Feature payload stored in `data/nfl.duckdb` table `features` (default `replace` mode) with metadata columns.
3. **Train baseline model + calibration**
   ```bash
   nfl-pred train --config configs/default.yaml --feature-set mvp_v1
   ```
   - Outputs: Joblib artifact in `data/models/`, reliability plot + config snapshot alongside it, MLflow run recorded in `./mlruns` with metrics.
4. **Generate weekly predictions**
   ```bash
   nfl-pred predict --season 2023 --week 18 --config configs/default.yaml
   ```
   - Outputs: Predictions appended to `predictions` table inside `data/nfl.duckdb` (defaults to latest model artifact) with `p_home_win`/`p_away_win` probabilities.
5. **Publish evaluation reports**
   ```bash
   nfl-pred report --season 2023 --week 18 --config configs/default.yaml
   ```
   - Outputs: `data/reports/metrics_s2023_w18.csv` and `data/reliability_s2023_w18.csv`, plus summary rows in DuckDB table `reports`.

Re-run steps 1–5 with different seasons/weeks as needed; predictions/reports require completed games in DuckDB schedules.

## Weather Data Overview (AI-108)

The modeling features incorporate both forecast and historical weather context:

- **National Weather Service (NWS)** — [`NWSClient`](src/nfl_pred/weather/nws_client.py) first resolves stadium coordinates through `/points/{lat},{lon}` and then fetches `/gridpoints/{wfo}/{x},{y}/forecast` (or `/forecast/hourly`). Responses are normalized to Celsius, meters-per-second, and probability units inside the client, and downstream feature assembly converts them to Fahrenheit, miles-per-hour, and fractional precipitation. The client memoizes point metadata for 6 hours and forecasts for 15 minutes, and it persists raw JSON payloads under `data/weather/nws/` via the [`WeatherArtifactStore`](src/nfl_pred/weather/storage.py) with matching TTL metadata recorded in `manifest.json`.
- **Meteostat fallback** — [`MeteostatClient`](src/nfl_pred/weather/meteostat_client.py) selects the nearest station within 10 miles and can supply hourly or daily backfill when a forecast is unavailable. Raw payloads are cached to `data/weather/meteostat/` through the same artifact store (no TTL is enforced by default), allowing offline replays of historical windows.
- **Indoor / closed-roof policy** — [`compute_weather_features`](src/nfl_pred/features/weather.py) only applies weather readings when the authoritative stadium roof is `outdoor`, `open`, or `retractable`. Games marked as indoor/closed keep `wx_temp` as `NaN` and default `wx_wind`/`precip` to `0`, reflecting the absence of weather impact.

