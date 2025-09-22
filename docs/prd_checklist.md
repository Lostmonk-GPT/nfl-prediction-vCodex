# PRD Conformance Checklist

Use this checklist to confirm that each PRD requirement is implemented, documented, and verified. Link items directly to the modules, docs, or tests that satisfy the requirement.

## 1. Data Sources and Refresh
- [ ] Ingestion flows pull nflverse schedules, play-by-play, and roster data via `nflreadpy` (`src/nfl_pred/ingest/schedules.py`, `src/nfl_pred/ingest/pbp.py`, `src/nfl_pred/ingest/rosters.py`).
- [ ] Injury and participation snapshots are captured with deterministic metadata (`src/nfl_pred/ingest/injuries.py`, `tests/test_ingest_injuries.py`).
- [ ] Contracts enforce required columns and refresh cadence metadata (`src/nfl_pred/ingest/contracts.py`, `tests/test_contracts.py`).

## 2. Game Snapshot Timeline
- [ ] Snapshot runner orchestrates T-24h, T-100m, T-80–75m, and T-60m checkpoints (`src/nfl_pred/snapshot/runner.py`, `tests/test_snapshot_runner.py`).
- [ ] Timeline and operational steps are documented (`docs/snapshot_timeline.md`, `docs/runbook.md`).

## 3. Stadiums and Weather
- [ ] Authoritative stadium reference table maintained and validated (`src/nfl_pred/ref/stadiums.py`, `data/ref/stadiums.csv`, `tests/test_stadiums.py`).
- [ ] Weather clients normalize NWS forecasts and Meteostat history with ≤10 mi station selection (`src/nfl_pred/weather/nws_client.py`, `src/nfl_pred/weather/meteostat_client.py`, `tests/test_weather_nws_client.py`, `tests/test_weather_meteostat_client.py`).
- [ ] Indoor/roof handling nulls weather features appropriately (`src/nfl_pred/features/weather.py`, `tests/test_weather.py`).

## 4. Rule-Change Guards
- [ ] Feature flags for kickoff/OT changes and playoffs are enforced in feature builders (`src/nfl_pred/features/rules.py`, `src/nfl_pred/features/playoffs.py`, `tests/test_rule_flags.py`, `tests/test_playoff_handling.py`).
- [ ] Policy documented for backfills and guard rails (`docs/rule_change_policy.md`).

## 5. Features and Field Mapping
- [ ] Team-week aggregations and rolling windows align with nflfastR columns (`src/nfl_pred/features/team_week.py`, `src/nfl_pred/features/windows.py`, `tests/test_windows.py`).
- [ ] Schedule metadata and travel features cover rest days, kickoff buckets, neutral sites, and haversine miles (`src/nfl_pred/features/schedule_meta.py`, `src/nfl_pred/features/travel.py`, `tests/test_schedule_meta.py`, `tests/test_travel_rest.py`).
- [ ] Injury rollups and weather enrichments feed the assembled matrix (`src/nfl_pred/features/injury_rollups.py`, `src/nfl_pred/features/weather.py`, `src/nfl_pred/features/build_features.py`, `tests/test_injury_rollups.py`).
- [ ] Feature specification is generated and versioned (`src/nfl_pred/docs/feature_spec.py`, `docs/feature_spec.md`).

## 6. Label Policy
- [ ] Label computation treats ties as 0.5 for calibration outputs (`src/nfl_pred/features/build_features.py`, `tests/test_pipeline_train.py`).

## 7. Training Horizon and Scope
- [ ] Configured training spans rolling recent seasons and excludes preseason (`configs/default.yaml`, `src/nfl_pred/pipeline/train.py`).
- [ ] Playoff handling is separated or flagged (`src/nfl_pred/features/playoffs.py`, `tests/test_playoff_handling.py`).

## 8. Modeling Recipe
- [ ] Level-0 models implemented (logistic, gradient-boosted, ridge) with consistent interfaces (`src/nfl_pred/model/baseline.py`, `src/nfl_pred/model/models.py`, `tests/test_model_baseline.py`, `tests/test_model_models.py`).
- [ ] Stacking meta-learner consumes out-of-fold predictions (`src/nfl_pred/model/stacking.py`, `tests/test_model_stacking.py`).
- [ ] Calibration selection compares Platt vs isotonic and persists final calibrator (`src/nfl_pred/model/calibration.py`, `tests/test_calibration_selection.py`).
- [ ] Explainability artifacts capture SHAP outputs with MLflow logging (`src/nfl_pred/explain/shap_utils.py`, `src/nfl_pred/explain/artifacts.py`, `tests/test_explain_shap.py`, `tests/test_explain_artifacts.py`).

## 9. Evaluation and Reporting
- [ ] Metrics pipeline computes Brier, log loss, and reliability views (`src/nfl_pred/reporting/metrics.py`, `tests/test_reporting_metrics.py`).
- [ ] Weekly and rolling reports generated for stakeholders (`src/nfl_pred/reporting/expanded.py`, `tests/test_reporting_expanded.py`).
- [ ] Monitoring-ready report CLI exists (`src/nfl_pred/reporting/monitoring_report.py`, `tests/test_reporting_monitoring.py`).

## 10. Validation Plan
- [ ] Forward-chaining time-series splits prevent leakage (`src/nfl_pred/model/splits.py`, `tests/test_model_splits.py`, `tests/test_windows_visibility.py`).
- [ ] Training pipeline regression test covers CV and holdout calibration (`src/nfl_pred/pipeline/train.py`, `tests/test_pipeline_train.py`).

## 11. Picks and Confidence
- [ ] Picks and confidence tiers follow PRD thresholds (`src/nfl_pred/picks.py`, `tests/test_picks.py`).

## 12. Monitoring and Retraining
- [ ] PSI and monitoring utilities enforce alert thresholds (`src/nfl_pred/monitoring/psi.py`, `tests/test_monitoring_psi.py`).
- [ ] Retrain triggers fire on PSI, Brier deterioration, and rule flips (`src/nfl_pred/monitoring/triggers.py`, `tests/test_monitoring_triggers.py`, `tests/test_psi_trigger_boundaries.py`).

## 13. Reproducibility and Storage
- [ ] DuckDB client manages deterministic storage with canonical keys (`src/nfl_pred/storage/duckdb_client.py`, `tests/test_snapshot_replay.py`).
- [ ] MLflow tracking and hygiene scripts maintain experiment artifacts (`src/nfl_pred/pipeline/train.py`, `src/nfl_pred/registry/hygiene.py`, `tests/test_mlflow_hygiene.py`).
- [ ] Configurable paths documented and managed via CLI (`configs/default.yaml`, `src/nfl_pred/cli.py`, `tests/test_registry_promote.py`).

## 14. Documentation Artifact
- [ ] Feature spec generator and runbook remain current (`src/nfl_pred/docs/feature_spec.py`, `docs/runbook.md`).
- [ ] Audit trail captures change history (`src/nfl_pred/docs/audit_trail.py`, `tests/test_audit_trail.py`).

## 15. Appendix A — Visibility Rules
- [ ] Visibility proxy enforces `asof_ts` filtering and prevents leakage (`src/nfl_pred/visibility.py`, `src/nfl_pred/snapshot/visibility.py`, `tests/test_visibility.py`, `tests/test_snapshot_visibility.py`).
- [ ] Snapshot replay verifies T-60m freeze compliance (`tests/test_snapshot_replay.py`).


---

## Verification Commands

| Checklist ID | Verification command(s) |
| --- | --- |
| 1.1 | `poetry run pytest tests/test_snapshot_runner.py::test_snapshot_runner_executes_stages_in_order` |
| 1.2 | `poetry run pytest tests/test_ingest_injuries.py` |
| 1.3 | `poetry run pytest tests/test_contracts.py` |
| 2.1 | `poetry run pytest tests/test_snapshot_runner.py` |
| 2.2 | `cat docs/snapshot_timeline.md`<br>`cat docs/runbook.md` |
| 3.1 | `poetry run pytest tests/test_stadiums.py` |
| 3.2 | `poetry run pytest tests/test_weather_nws_client.py tests/test_weather_meteostat_client.py` |
| 3.3 | `poetry run pytest tests/test_weather.py` |
| 4.1 | `poetry run pytest tests/test_rule_flags.py tests/test_playoff_handling.py` |
| 4.2 | `cat docs/rule_change_policy.md` |
| 5.1 | `poetry run pytest tests/test_windows.py` |
| 5.2 | `poetry run pytest tests/test_schedule_meta.py tests/test_travel_rest.py` |
| 5.3 | `poetry run pytest tests/test_injury_rollups.py tests/test_weather.py` |
| 5.4 | `poetry run pytest tests/docs/test_feature_spec.py` |
| 6.1 | `poetry run pytest tests/test_pipeline_train.py` |
| 7.1 | `poetry run pytest tests/test_pipeline_train.py` |
| 7.2 | `poetry run pytest tests/test_playoff_handling.py` |
| 8.1 | `poetry run pytest tests/test_model_baseline.py tests/test_model_models.py` |
| 8.2 | `poetry run pytest tests/test_model_stacking.py tests/test_stacking_shap.py` |
| 8.3 | `poetry run pytest tests/test_calibration_selection.py tests/test_model_calibration.py` |
| 8.4 | `poetry run pytest tests/test_explain_shap.py tests/test_explain_artifacts.py` |
| 9.1 | `poetry run pytest tests/test_reporting_metrics.py` |
| 9.2 | `poetry run pytest tests/test_reporting_expanded.py` |
| 9.3 | `poetry run pytest tests/test_reporting_monitoring.py` |
| 10.1 | `poetry run pytest tests/test_model_splits.py tests/test_windows_visibility.py` |
| 10.2 | `poetry run pytest tests/test_pipeline_train.py` |
| 11.1 | `poetry run pytest tests/test_picks.py` |
| 12.1 | `poetry run pytest tests/test_monitoring_psi.py tests/test_psi_trigger_boundaries.py` |
| 12.2 | `poetry run pytest tests/test_monitoring_triggers.py` |
| 13.1 | `poetry run pytest tests/test_duckdb_client.py tests/test_snapshot_replay.py` |
| 13.2 | `poetry run pytest tests/test_mlflow_hygiene.py` |
| 13.3 | `poetry run pytest tests/test_cli.py tests/test_registry_promote.py` |
| 14.1 | `poetry run pytest tests/docs/test_feature_spec.py`<br>`cat docs/runbook.md` |
| 14.2 | `poetry run pytest tests/test_audit_trail.py` |
| 15.1 | `poetry run pytest tests/test_visibility.py tests/test_snapshot_visibility.py` |
| 15.2 | `poetry run pytest tests/test_snapshot_replay.py` |
