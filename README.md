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
