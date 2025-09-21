# Snapshot Timeline

## Objectives
- Align all pre-game processing with the freeze rules from Appendix A of the PRD to prevent post-cutoff leakage.
- Provide operators a concise checklist describing what to run at each stage and what inputs are permitted.
- Anchor the automation surface area to the [`run_snapshot_workflow`](../src/nfl_pred/snapshot/runner.py) orchestration and [`VisibilityContext`](../src/nfl_pred/snapshot/visibility.py) enforcement helpers.

## Cutoffs
- **T-24h:** Initial visibility window using data available one day prior to kickoff.
- **T-100m:** Refresh window after late-week injury reports and roster adjustments.
- **T-80–75m:** Post-official inactives confirmation.
- **T-60m:** Final freeze; lock features and produce picks.

## Allowed Data per Snapshot
- **T-24h:**
  - Season-to-date box score, rolling windows, travel metrics, and venue metadata persisted from ingestion.
  - Historical weather backfill and any injury participation reports captured before the cutoff.
  - No same-day roster transactions unless timestamped before the cutoff.
- **T-100m:**
  - Re-run injury and roster ingestions to capture late practice reports (`refresh_injuries=True`, `refresh_rosters=False` by default) before re-building features via the snapshot runner.
  - Injury designations, depth chart changes, and participation statuses effective at or before the cutoff.
  - Weather updates from public APIs are limited to data published prior to the cutoff; no inactives yet.
- **T-80–75m:**
  - Official inactives files, starting QB confirmations, and final roster scrapes.
  - Injury refresh remains enabled in [`DEFAULT_SNAPSHOT_STAGES`](../src/nfl_pred/snapshot/runner.py) to pull the inactives payloads.
  - Weather refresh allowed only if API timestamps precede the cutoff; indoor games retain null weather features.
- **T-60m:**
  - Snapshot feature tables generated with `feature_write_mode="append"` so historical snapshots remain immutable.
  - No additional roster or injury pulls; use cached outputs from prior stages to avoid leakage.
  - Features, predictions, and explanations are persisted using the same `snapshot_at` timestamp.

## Weather and Stadium Visibility
- Stadium metadata is authoritative via [`ref/stadiums`](../src/nfl_pred/ref/stadiums.py) and downstream joins; it drives roof type, surface, timezone, altitude, and neutral-site flags.
- Weather ingestion obeys Appendix A:
  - Outdoor / retractable-open venues query NOAA `/points` → `/gridpoints/.../forecast` (hourly when available) and cache raw + normalized series.
  - Meteostat backfill is limited to stations within 10 miles when NOAA gaps exist.
  - Indoor or closed-roof games set weather features to null/zero regardless of external feeds.
- The snapshot runner only refreshes weather-derived features indirectly when the feature builder re-runs; all API pulls must be timestamped before the stage cutoff.

## Enforcement
- All feature builders accept `asof_ts` and `snapshot_at` and must route through [`VisibilityContext`](../src/nfl_pred/snapshot/visibility.py) filters (e.g., `filter_play_by_play`, `filter_schedule`, `filter_weekly_frame`).
- Legacy week-level helpers in [`visibility.py`](../src/nfl_pred/visibility.py) remain available for coarse replays but should degrade to the stricter snapshot filters when `asof_ts` is provided.
- Historical replays (e.g., tests under `tests/test_windows_visibility.py`) must assert no data with `event_time > asof_ts` is consumed at T-60m.
- The CLI entry point `nfl_pred.cli:snapshot` wires timestamps into `run_snapshot_workflow`, guaranteeing consistent enforcement regardless of operator.

## Outputs
- Each stage records a [`StageExecution`](../src/nfl_pred/snapshot/runner.py) with the stage name, timestamp, feature build result, and optional prediction result.
- T-60m stages call [`run_inference_pipeline`](../src/nfl_pred/pipeline/predict.py) to persist:
  - Calibrated probabilities (`p_home_win`, `p_away_win`) and derived picks/confidence tiers.
  - Feature matrices registered in DuckDB with deterministic `snapshot_at` partitions.
  - SHAP summaries (≤20% sampling) and config snapshots logged to MLflow when enabled.
- Earlier stages update feature stores in-place but do not emit predictions or explanations.

## Audit Trail
- Persist the following alongside each `snapshot_at`:
  - Snapshot timestamp, season, week, and game identifiers as composite keys in DuckDB (`storage/schema.sql`).
  - Upstream dataset versions and ingestion hashes recorded by ingestion utilities.
  - Model artifact hash/ID, calibration choice, and feature specification checksum from the training pipeline.
  - Input row hashes or fingerprints stored with feature outputs to support reproducibility checks.
- `run_snapshot_workflow` logs executions; MLflow runs should include config artifacts and output manifests for cross-stage diffing.
