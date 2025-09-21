# AI Implementation Task Backlog

Version: 1.0
Date: 2025-09-19
Source Plan: NFL_Game_Outcome_Prediction_Dev_Plan.md

---

## MVP (Weeks 1–2)

- [AI-001] Create repository scaffold
  - Files: pyproject.toml, src/, configs/, tests/, data/, Makefile, .gitignore; src/nfl_pred/__init__.py
  - DoD: Repo tree exists; package import `nfl_pred` works.

- [AI-002] Define dependencies and Python version
  - Files: pyproject.toml
  - DoD: Python ≥3.11 pinned; deps listed (nflreadpy, duckdb, pandas/polars, pyarrow, scikit-learn, xgboost or lightgbm, mlflow, shap, requests, meteostat, typer or argparse).

- [AI-003] Base config and loader
  - Files: configs/default.yaml, src/nfl_pred/config.py
  - DoD: Dataclass loader reads YAML + env overrides; returns typed config.

- [AI-004] Logging setup
  - Files: src/nfl_pred/logging_setup.py
  - DoD: `setup_logging(level)` configures console logs with module and time.

- [AI-005] DuckDB helper
  - Files: src/nfl_pred/storage/duckdb_client.py
  - DoD: Context-managed client opens data/nfl.duckdb; read_sql/write_df/register_parquet methods.

- [AI-006] DuckDB schemas
  - Files: src/nfl_pred/storage/schema.sql
  - DoD: Declares features, predictions, reports, runs_meta; keys (season, week, game_id, snapshot_at).

- [AI-007] Ingestion: schedules
  - Files: src/nfl_pred/ingest/schedules.py
  - DoD: Pulls season schedules via nflreadpy; writes data/raw/schedules.parquet; adds pulled_at, source_version.

- [AI-008] Ingestion: play-by-play
  - Files: src/nfl_pred/ingest/pbp.py
  - DoD: Pulls PBP per season; writes data/raw/pbp_YYYY.parquet; includes metadata columns.

- [AI-009] Ingestion: rosters/team info
  - Files: src/nfl_pred/ingest/rosters.py
  - DoD: Pulls rosters and teams; writes Parquet; metadata included.

- [AI-010] Data contracts/validators
  - Files: src/nfl_pred/ingest/contracts.py
  - DoD: Functions assert required columns for schedules/PBP/rosters with clear errors.

- [AI-011] Feature windows utilities
  - Files: src/nfl_pred/features/windows.py
  - DoD: Rolling windows (4/8/season-to-date) grouped by team/week with asof_ts filter.

- [AI-012] Core team-week features from PBP
  - Files: src/nfl_pred/features/team_week.py
  - DoD: Computes EPA/play, early-down EPA, success rate, pass/run, play-action, shotgun, no-huddle, sacks, explosive rates, penalties, ST EPA.

- [AI-013] Rest days and kickoff bucket
  - Files: src/nfl_pred/features/schedule_meta.py
  - DoD: Derives rest_days, short_week, kickoff_bucket, home_away.

- [AI-014] Travel features
  - Files: src/nfl_pred/features/travel.py
  - DoD: Haversine prev venue→current venue miles, days_since_last, neutral_site; caches venue lat/lon.

- [AI-015] Assemble MVP feature matrix
  - Files: src/nfl_pred/features/build_features.py
  - DoD: Joins PBP aggregates + schedule meta + travel into DataFrame; writes to DuckDB features table.

- [AI-016] Modeling: data split
  - Files: src/nfl_pred/model/splits.py
  - DoD: Forward-chaining CV grouped by week; no future leakage.

- [AI-017] Modeling: baseline classifier
  - Files: src/nfl_pred/model/baseline.py
  - DoD: Logistic or LightGBM; standard preprocessing; exposes fit and predict_proba.

- [AI-018] Calibration (Platt)
  - Files: src/nfl_pred/model/calibration.py
  - DoD: Wraps base model with sigmoid scaling on held-out fold; calibrated predict_proba.

- [AI-019] Training pipeline
  - Files: src/nfl_pred/pipeline/train.py
  - DoD: Reads features; runs CV (Brier/log-loss); calibrates; logs to MLflow; saves model to data/models/.

- [AI-020] Inference pipeline
  - Files: src/nfl_pred/pipeline/predict.py
  - DoD: Loads model; outputs p_home_win, p_away_win; writes predictions table.

- [AI-021] Picks and confidence
  - Files: src/nfl_pred/picks.py
  - DoD: Implements pick rule (≥0.5) and tiers: Strong ≥0.65; Lean 0.55–0.65; Pass <0.55.

- [AI-022] Reporting: metrics + reliability
  - Files: src/nfl_pred/reporting/metrics.py
  - DoD: Computes Brier/log-loss and reliability curve; writes reports and saves plot artifacts.

- [AI-023] CLI entrypoints
  - Files: src/nfl_pred/cli.py
  - DoD: Commands ingest, build-features, train, predict, report; args for seasons/week/config paths.

- [AI-024] Visibility proxy (MVP)
  - Files: src/nfl_pred/visibility.py
  - DoD: Coarse week-level asof_ts filter using scheduled kickoff as proxy.

- [AI-025] Unit tests: windows/visibility
  - Files: tests/test_windows_visibility.py
  - DoD: Tests rolling windows correctness and excludes post-week data for asof_ts.

- [AI-026] Unit tests: travel/rest
  - Files: tests/test_travel_rest.py
  - DoD: Validates haversine, rest-day rules, neutral site handling.

- [AI-027] Unit tests: splits and metrics
  - Files: tests/test_model_split_metrics.py
  - DoD: Verifies forward split ordering and metric calculations.

- [AI-028] Document MVP runbook
  - Files: README.md (update)
  - DoD: Includes commands and expected outputs for a past week replay.

---

## Phase 1 — Stadium Authority & Weather (Week 3)

- [AI-101] Authoritative stadium table
  - Files: data/ref/stadiums.csv, src/nfl_pred/ref/stadiums.py
  - DoD: Schema includes venue, teams, lat, lon, tz, altitude, surface, roof, neutral_site.

- [AI-102] Stadium join logic
  - Files: src/nfl_pred/features/stadium_join.py
  - DoD: Schedules joined to stadiums; schedule roof conflicts resolved by authority rules.

- [AI-103] NWS client
  - Files: src/nfl_pred/weather/nws_client.py
  - DoD: point_forecast(lat, lon) using /points → /gridpoints/.../forecast; caching/backoff; unit normalization.

- [AI-104] Meteostat client
  - Files: src/nfl_pred/weather/meteostat_client.py
  - DoD: Nearest-station ≤10mi, hourly/daily fetch, normalized outputs, persistence of raw payloads.

- [AI-105] Weather feature builder
  - Files: src/nfl_pred/features/weather.py
  - DoD: Derives wx_temp, wx_wind, precip for outdoor/open/retractable-open; indoor/closed set null/zero.

- [AI-106] Weather artifacts and metadata
  - Files: src/nfl_pred/weather/storage.py
  - DoD: Persists raw API responses (JSON) with call metadata and TTL.

- [AI-107] Weather tests with fixtures
  - Files: tests/fixtures/nws/*.json, tests/test_weather.py
  - DoD: Tests normalization, indoor nulling, station selection.

- [AI-108] Docs update for weather
  - Files: README.md (update)
  - DoD: Notes API usage, caching, and indoor policy.

---

## Phase 2 — Injuries/Participation & Snapshot Timeline (Week 4)

- [AI-201] Injuries ingestion
  - Files: src/nfl_pred/ingest/injuries.py
  - DoD: Pulls injuries/participation; writes Parquet with event timestamps.

- [AI-202] Position-group rollups
  - Files: src/nfl_pred/features/injury_rollups.py
  - DoD: DNP/LP/FP counts per position group at snapshot week.

- [AI-203] Snapshot runner
  - Files: src/nfl_pred/snapshot/runner.py
  - DoD: Implements T-24h, T-100m, T-80–75m, T-60m; writes snapshot_at values to outputs.

- [AI-204] Visibility enforcement
  - Files: src/nfl_pred/snapshot/visibility.py
  - DoD: Common filter to enforce event_time ≤ asof_ts across sources; integrated into all builders.

- [AI-205] Snapshot CLI
  - Files: src/nfl_pred/cli.py (extend)
  - DoD: Command `snapshot --at <iso8601>` runs full pre-game freeze and outputs predictions.

- [AI-206] Historical replay test
  - Files: tests/test_snapshot_replay.py
  - DoD: Replays a past game with asof_ts=T-60m and asserts no post-cutoff reads.

- [AI-207] Injury visibility tests
  - Files: tests/test_injury_visibility.py
  - DoD: Ensures only records with event_time ≤ asof_ts included.

- [AI-208] Docs: snapshot timeline
  - Files: docs/snapshot_timeline.md
  - DoD: Documents snapshot sequence and visibility rules.

---

## Phase 3 — Modeling Enhancements & Explainability (Week 5)

- [AI-301] Additional level-0 models ✅ COMPLETED 2025-09-20
  - Files: src/nfl_pred/model/models.py
  - DoD: Ridge/logistic/GBDT with unified interface.
  - Notes: Includes sklearn compatibility fixes, comprehensive docstrings, and ADR-20250920-level-0-model-interface.md

- [AI-302] Stacking pipeline ✅ COMPLETED
  - Files: src/nfl_pred/model/stacking.py
  - DoD: OOF generation and logistic meta-learner; consistent CV folds.
  - Notes: Includes StackingEnsemble class with generate_oof_predictions() for leakage prevention

- [AI-303] Calibration selection ✅ COMPLETED
  - Files: src/nfl_pred/model/calibration.py (extend)
  - DoD: Choose isotonic vs Platt based on validation log-loss.
  - Notes: Includes IsotonicCalibrator, CalibrationSelector, compare_calibrators with minimum sample size guards

- [AI-304] SHAP explainability ✅
  - Files: src/nfl_pred/explain/shap_utils.py, configs/default.yaml, examples/shap_explainability_example.py
  - DoD: TreeExplainer on 10–20% sample; approximate/GPU fallback; weekly summary plots.
  - Notes: Complete implementation with ShapConfig, compute_shap_values, visualization plots, MLflow integration, version compatibility fixes

- [AI-305] Explainability artifacts ✅
  - Files: src/nfl_pred/explain/artifacts.py, configs/default.yaml, examples/artifacts_workflow_demo.py
  - DoD: Save SHAP values/plots to MLflow and disk; link to predictions.
  - Notes: Complete implementation with ArtifactMetadata, deterministic storage, MLflow integration, discovery utilities, and cleanup functionality

- ✅ [AI-306] Tests for stacking/SHAP
  - Files: tests/test_stacking_shap.py
  - DoD: Validates OOF shape, meta-learner training, and SHAP sampling routine.
  - Status: COMPLETED - Comprehensive test suite with 19 tests covering stacking OOF generation, ensemble functionality, SHAP sampling, integration, edge cases, and error reporting

---

## Phase 4 — Evaluation, Monitoring, Retraining (Week 6)

- ✅ [AI-401] Expanded evaluation reports
  - Files: src/nfl_pred/reporting/expanded.py
  - DoD: Weekly, season-to-date, rolling 4-week; favorite/underdog slices.
  - Status: COMPLETED - Comprehensive reporting system with 3 time windows (weekly, season-to-date, rolling 4-week), confidence slice analysis (favorite/tossup/underdog), trend plots, slice comparison charts, and database persistence

- [AI-402] PSI drift monitoring
  - Files: src/nfl_pred/monitoring/psi.py
  - DoD: PSI computation for key features with alert threshold ≥0.2.

- [AI-403] Retrain triggers
  - Files: src/nfl_pred/monitoring/triggers.py
  - DoD: Implements 10% 4-week Brier deterioration, PSI breaches, rule-flag flips.

- [AI-404] Monitoring CLI/report
  - Files: src/nfl_pred/cli.py (extend), src/nfl_pred/reporting/monitoring_report.py
  - DoD: Outputs dashboard-ready JSON/plots.

- [AI-405] MLflow model registry hooks
  - Files: src/nfl_pred/registry/promote.py
  - DoD: Promotes models meeting thresholds; tags runs with lineage.

- [AI-406] Tests for PSI/triggers
  - Files: tests/test_monitoring.py
  - DoD: Unit tests for PSI calc and trigger logic.

---

## Phase 5 — Rule-Change Guards & Postseason

- [AI-501] Rule flags
  - Files: src/nfl_pred/features/rules.py
  - DoD: Adds kickoff_2024plus, ot_regular_2025plus to feature matrix.

- [AI-502] Playoff handling
  - Files: src/nfl_pred/features/playoffs.py
  - DoD: Flags postseason rows and allows separate fit toggle.

- [AI-503] Rule-change backfill policy
  - Files: docs/rule_change_policy.md
  - DoD: Documents season weighting and inclusion around transitions.

- [AI-504] Tests for flags/playoffs
  - Files: tests/test_rule_flags.py
  - DoD: Verifies activation by date/season and playoff splits.

---

## Phase 6 — Reproducibility & Documentation

- [AI-601] Feature spec generator
  - Files: src/nfl_pred/docs/feature_spec.py
  - DoD: Generates table (name, definition, sources, window, snapshot timing, null policy, rule notes).

- [AI-602] Audit trail writer
  - Files: src/nfl_pred/audit/trail.py
  - DoD: Writes snapshot timestamps, dataset versions, model hash, code version, feature spec checksum, input row hashes to DuckDB.

- [AI-603] MLflow hygiene
  - Files: src/nfl_pred/audit/mlflow_utils.py
  - DoD: Standardizes tags (snapshot, season/week, lineage), retention policies.

- [AI-604] Operational runbook
  - Files: docs/runbook.md
  - DoD: End-to-end instructions for weekly cadence and snapshot management.

- [AI-605] PRD conformance checklist
  - Files: docs/prd_checklist.md
  - DoD: Checkboxes mapped to PRD sections, all met.

---

## Cross-Cutting

- [AI-701] Configurable paths and env
  - Files: configs/default.yaml (extend), src/nfl_pred/config.py (extend)
  - DoD: All paths/URIs configurable via YAML/env.

- [AI-702] Caching layer
  - Files: src/nfl_pred/utils/cache.py
  - DoD: TTL cache for HTTP calls with persistent store and key hashing.

- [AI-703] Secrets handling
  - Files: src/nfl_pred/utils/secrets.py
  - DoD: Reads API keys/tokens from env or .env; no secrets hardcoded.

- [AI-704] CI test data fixtures
  - Files: tests/fixtures/ curated small Parquet/JSON
  - DoD: Tests run offline with deterministic inputs.

---

## Notes for the AI Implementer

- Enforce visibility: all feature builders accept asof_ts and must filter sources to event_time ≤ asof_ts.
- Persist deterministically: DuckDB tables keyed by season, week, game_id, snapshot_at.
- Keep interfaces typed and documented; avoid breaking public function signatures across tasks.
- Tests must not require network; use fixtures for NWS/Meteostat and nflverse samples.
