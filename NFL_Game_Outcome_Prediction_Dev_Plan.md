# NFL Game Outcome Prediction — Development Plan (MVP → Full PRD)

Version: 1.0  
Date: 2025-09-19  
Source PRD: NFL_Game_Outcome_Prediction_PRD_2025-09-17.md

---

## Overview & Scope
- Goal: Pre-game win probabilities and picks with confidence tiers, frozen at T-60m before kickoff.
- Inputs: `nflreadpy` (nflverse PBP, schedules, rosters, injuries, participation, officials, team info), authoritative stadium table, NWS forecasts, Meteostat historical weather.
- Outputs: `p_home_win`/`p_away_win`, pick, confidence tier; weekly reports (Brier, log-loss, calibration); SHAP summaries; monitoring and retraining signals. Persist to DuckDB and track with MLflow.
- Constraints: Strict visibility and deterministic cutoffs (T-24h → T-100m → T-80–75m → T-60m). All builders accept `asof_ts` and filter sources to `event_time ≤ asof_ts`.

---

## Assumptions & Tech Stack
- Python 3.11+, with: `nflreadpy`, `pandas`/`polars`, `duckdb`, `pyarrow`, `scikit-learn`, `xgboost` or `lightgbm`, `mlflow`, `shap`, `requests`, `meteostat`.
- Storage: Parquet caches + DuckDB warehouse. Keys: `season`, `week`, `game_id`, `snapshot_at` (aka `asof_ts`).
- Experiment tracking: MLflow local tracking URI with params, metrics, artifacts (plots, SHAP, config).
- Orchestration: Simple CLI (e.g., `typer`/`argparse`) + cron/CI scheduler for snapshot cadence.
- Reproducibility: `pyproject.toml` + lockfile; pin model/config versions in DuckDB and MLflow.

---

## MVP Definition
- Features (in-scope):
  - Team-week aggregates from PBP (season-to-date and rolling 4/8/season where available): EPA/play, early-down EPA, success rate, pass/run rate, play-action rate, shotgun rate, no-huddle rate, sack rate, explosive pass/run, penalties/play, special teams EPA.
  - Schedule/meta: home/away, rest days, kickoff bucket, surface/roof (from schedule as proxy), short-week flag.
  - Travel: approximate previous venue → current venue distance with haversine; days since last game.
- Modeling: Single model (logistic regression or GBDT) + Platt calibration. Forward-chaining CV grouped by NFL week.
- Outputs: Per-game probabilities, pick rule (≥0.5), confidence tiers (Strong ≥0.65; Lean 0.55–0.65; Pass <0.55), basic weekly report (Brier/log-loss + reliability plot). Persist to DuckDB; log to MLflow.
- Explicitly out-of-scope for MVP: NWS/Meteostat weather, SHAP, model stacking, PSI drift monitoring, detailed inactives-based injury visibility, strict T-80 sweep.

---

## MVP Architecture
- `data/`: Ingestion from `nflreadpy`; Parquet cache with source metadata (`pulled_at`, `source_version`).
- `features/`: Team-week aggregations (4/8/season windows), rest days, travel, schedule/meta joins. Enforce coarse `asof_ts` using scheduled kickoff as proxy.
- `storage/`: DuckDB helpers; schemas for raw external tables and curated feature/prediction/report tables.
- `model/`: Train/eval/predict with forward CV, calibration, and MLflow logging.
- `cli/`: Commands: `ingest`, `build-features`, `train`, `predict`, `report`.
- `configs/`: Feature windows, train/eval settings, paths, and snapshot settings.

---

## MVP Implementation Steps
1) Project scaffolding  
- Create project structure (`src/`, `configs/`, `tests/`, `Makefile`) and `pyproject.toml`.  
- Configure MLflow tracking URI, DuckDB path, and data cache dirs.

2) Ingestion & storage  
- Pull schedules, PBP, rosters/team info via `nflreadpy`.  
- Persist to Parquet with `pulled_at`, `source`, and version metadata.  
- Expose DuckDB views for raw Parquet and create curated schemas.

3) Feature engineering  
- Build team-week aggregates and rolling windows; compute rest days and travel distance.  
- Join schedule/meta; derive home/away and kickoff bucket.  
- Enforce `asof_ts` (MVP proxy: pre-game week-level freeze).

4) Modeling  
- Implement forward-chaining CV grouped by week.  
- Fit single model (logistic or GBDT) with Platt calibration.  
- Log params, metrics (Brier/log-loss), and artifacts to MLflow.

5) Inference & reporting  
- Produce per-game probabilities/picks/confidence tiers for target week.  
- Store predictions to DuckDB; generate basic report (Brier/log-loss, reliability plot).  
- Export artifacts to MLflow.

6) Testing & validation (MVP)  
- Unit tests: feature window correctness, travel calc, coarse `asof_ts` gating, reproducible splits.  
- Sanity replay for a past week; verify no obvious post-game leakage.

---

## Full PRD Delivery — Phased Plan

### Phase 1: Stadium Authority & Weather
- Authoritative stadium table with `venue`, `team(s)`, `lat`, `lon`, `tz`, `altitude`, `surface`, `roof ∈ {indoors,dome,open,retractable}`, `neutral_site`.
- NWS integration: `/points/{lat,lon}` → `/gridpoints/{wfo}/{x},{y}/forecast` (or hourly). Normalize units; indoor handling sets weather to null/zero if `roof ∈ {indoors,dome,closed}`.
- Meteostat historical weather: nearest station ≤10 miles using `Stations.nearby` or `Point` lookup. Persist raw payloads and normalized features.
- Caching and retry policy; persist API call metadata and backfill coverage.

### Phase 2: Injuries/Participation & Snapshot Timeline
- Ingest weekly injuries and participation; create position-group rollups (counts of DNP/LP/FP).  
- Snapshot runner implementing T-24h, T-100m, T-80–75m, and T-60m.  
- Strict visibility: all feature builders accept `asof_ts` and filter `event_time ≤ asof_ts`.
- Unit test: replay historical games with `asof_ts = T-60m` and assert no post-cutoff reads.

### Phase 3: Modeling Enhancements
- Level-0 models: logistic regression, ridge, GBDT (XGBoost/LightGBM).  
- Stacking: combine out-of-fold probabilities via a logistic meta-learner.  
- Final calibration: isotonic or Platt selected by validation.  
- Explainability: SHAP TreeExplainer on 10–20% sampled rows per week; fallback to approximate/GPU if needed.

### Phase 4: Evaluation, Monitoring, Retraining
- Expanded evaluation: weekly, season-to-date, rolling 4-week; stratify by favorite/underdog.  
- Monitoring: PSI on key features (alert at PSI ≥0.2); weekly calibration charts.  
- Retrain triggers: any of (1) 4-week Brier worsens ≥10% vs baseline, (2) PSI ≥0.2 on ≥5 key features, (3) rule-flag flip.

### Phase 5: Rule-Change Guards & Postseason Handling
- Add binary flags: `kickoff_2024plus`, `ot_regular_2025plus` to feature matrix.  
- Playoffs: separate fit or explicit flags; distinct evaluation slices.

### Phase 6: Reproducibility & Documentation
- Feature spec table with: name, definition, source columns, window, snapshot timing, null policy, rule notes.  
- Audit trail in DuckDB: snapshot timestamps, upstream dataset versions, model hash, code version, feature spec checksum, input row hashes.  
- MLflow experiment hygiene: run tags for snapshot, season/week, and model lineage.

---

## Data Model & Keys (DuckDB)
- Raw (external Parquet) with metadata: `pulled_at`, `source`, `source_version`.
- Curated features: `features(team_id, game_id, season, week, asof_ts, …)`.
- Predictions: `predictions(game_id, season, week, asof_ts, p_home_win, p_away_win, pick, confidence, model_id)`.
- Reports: `metrics(season, week, asof_ts, metric, value)`, `calibration_bins(...)`, `drift(...)`.
- Keys: `season`, `week`, `game_id`, `snapshot_at`/`asof_ts` canonical for joins and audit.

---

## Validation & Monitoring
- CV methodology: forward-chaining grouped by NFL week; no leakage; stable seeds.  
- Metrics: Brier score, log loss, calibration error; bucketed by favorite/underdog; rolling 4-week aggregation.  
- Monitoring: PSI on selected features; weekly calibration charts; alerting for retrain triggers.

---

## Risks & Mitigations
- Historical injury/inactives visibility times may be incomplete.  
  Mitigation: approximate with week-level snapshots initially; document limitations; add strict T-80 replays where feasible.
- Weather API reliability/rate limits.  
  Mitigation: caching with TTL and retries; persist raw payloads and normalized outputs.
- nflverse schema drift.  
  Mitigation: input contracts and smoke tests; pin versions; add schema validation at ingestion.
- SHAP performance at scale.  
  Mitigation: sample 10–20% weekly; approximate/GPU modes.

---

## Milestones & Timeline
- Week 1: Scaffold project; ingestion + DuckDB; core features; base model; basic report.  
- Week 2: MVP complete (CLI, MLflow, predictions, basic validation and docs).  
- Week 3: Stadium authority + NWS/Meteostat integration; weather features + caching; tests.  
- Week 4: Injuries/participation rollups; snapshot runner T-24h→T-60m; visibility unit tests.  
- Week 5: Stacking + calibration selection; SHAP summaries; expanded evaluation.  
- Week 6: Monitoring (PSI, rolling metrics, alerts) + retrain pipeline; rule flags; playoffs handling; finalize docs/audit.

---

## Deliverables
- Code structure: `src/` modules, `configs/`, `tests/`, `Makefile`, `pyproject.toml` (scaffolded later per request).  
- Data: Parquet caches and DuckDB database tables (raw external references and curated).  
- Artifacts: MLflow runs, calibration plots, PSI drift plots, SHAP summaries.  
- Documentation: Feature spec table, runbook for snapshot timeline, model cards, monitoring SOP, and this development plan.

---

## Alignment to PRD Sections
- Data Sources & Refresh: ingestion via `nflreadpy`, nightly refresh after 12:00 AM ET.  
- Snapshot Timeline: T-24h, T-100m, T-80–75m, T-60m implemented via snapshot runner.  
- Stadiums & Weather: authoritative stadium table, NWS, Meteostat, indoor nulling.  
- Rule Guards: binary flags `kickoff_2024plus`, `ot_regular_2025plus`.  
- Features: mapped from PBP/schedule/roster/injuries/weather/travel; rolling windows.  
- Label Policy: binary win label; ties count as 0.5 in calibration.  
- Training Horizon: rolling 3–5 recent seasons; exclude preseason; playoff handling.  
- Modeling: level-0 models, stacking, final calibration, SHAP.  
- Evaluation: Brier, log-loss, calibration plots; weekly and rolling.  
- Validation: forward-chaining grouped by week; no leakage.  
- Picks & Confidence: thresholds 0.5/0.55/0.65.  
- Monitoring & Retraining: PSI, rolling metrics, triggers.  
- Reproducibility & Storage: DuckDB over Parquet; MLflow; env pinning.  
- Documentation: feature spec table and audit trail.

---

## Appendix A — Visibility Enforcement Notes
- All feature builders accept `asof_ts` and filter `event_time ≤ asof_ts`.  
- Unit test to replay a historical game at `asof_ts = T-60m` and assert no post-cutoff reads.  
- Weather visibility: outdoors/open/retractable-open only; indoor sets weather features null/zero.  
- Travel computed at T-24h and updated if venue changes.  
- Freeze outputs at T-60m: `p_home_win`, `p_away_win`, pick, confidence tier, weekly SHAP on ≤20% sampled rows.

---

End of Development Plan.
