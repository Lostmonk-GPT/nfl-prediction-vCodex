# Operational Runbook

## Purpose and Scope
This runbook provides the end-to-end operational procedures for the NFL Game Outcome
Prediction platform. It aligns weekly duties with the Tuesday→Monday cadence outlined in the
PRD, integrates the CLI workflows under `nfl-pred`, and documents controls for visibility, model
hygiene, monitoring, and emergency response. Every task includes actionable commands and
validation steps so on-call operators can execute reliably without institutional knowledge.

## Weekly Operations Timeline (Tuesday → Monday)
| Day | Primary Focus | Required Commands | Validation & Artifacts |
| --- | -------------- | ----------------- | ---------------------- |
| **Tuesday** | Post-week ingestion catch-up and data QA. | `poetry run nfl-pred ingest --seasons 2024 2025` (adjust seasons to current + upcoming). | Confirm new Parquet files under `data/raw/`; run `duckdb data/nfl.duckdb "SELECT COUNT(*) FROM schedules WHERE season=2024"` to verify row counts match NFL schedule totals. |
| **Wednesday** | Feature regeneration dry-run (no predictions). | `poetry run nfl-pred build-features --seasons 2024 --write-mode replace --asof-ts "2024-10-09T17:00:00Z" --snapshot-at "2024-10-09T17:00:00Z"` | Inspect DuckDB `features` table (`SELECT DISTINCT snapshot_at FROM features ORDER BY snapshot_at DESC LIMIT 1;`) to ensure new snapshot timestamp registered. |
| **Thursday** | Monitoring refresh + trigger evaluation. | `poetry run nfl-pred monitor --season 2024 --week 6` | Review generated CSV/PNG artifacts in `data/reports/monitoring/season_2024/week_06/`; cross-check PSI summary for breaches ≥0.2 and confirm automated alert if ≥5 features triggered. |
| **Friday** | Candidate retrain window assessment. | `poetry run nfl-pred report --season 2024 --week 6` to confirm metrics; if triggers fire, schedule training with `poetry run nfl-pred train --feature-set mvp_v1 --calibration-weeks 2`. | Verify MLflow run ID in console output; confirm new artifact via `ls -lt data/models/*.joblib | head -n 5` and capture metrics from `data/models/reliability_*.png`. |
| **Saturday** | Staging snapshot rehearsal using next kickoff. | `poetry run nfl-pred snapshot --season 2024 --week 7 --at "2024-10-13T17:00:00Z" --final-only` | Ensure final-stage execution logged and predictions appended to DuckDB `predictions` table with rehearsal timestamp; rollback rehearsal predictions via `duckdb data/nfl.duckdb "DELETE FROM predictions WHERE snapshot_at = '<rehearsal_ts>'"` if necessary. |
| **Sunday (Game Day)** | Full production snapshot ladder (T-24h through T-60m), live monitoring. | See [Snapshot Execution Playbook](#snapshot-execution-playbook) for per-stage commands. | Check MLflow for logged SHAP/config artifacts and confirm Slack `#nfl-ops` receives status pings; ensure final predictions exported to downstream channels. |
| **Monday** | Post-week reconciliation, anomaly triage, handoff report. | `poetry run nfl-pred report --season 2024 --week 7` followed by `poetry run nfl-pred monitor --season 2024 --week 7`; run `poetry run pytest` if changes deployed. | Update `docs/runbook.md` handoff notes if procedures changed; publish summary in Confluence and attach DuckDB extracts plus monitoring CSVs from `data/reports/monitoring/season_2024/week_07/`. |

> **Timekeeping Reminder:** Always operate in UTC in commands. Convert local kickoff times using
`python - <<'PY'` scripts or trusted scheduling tools before invoking CLI timestamps.

## Snapshot Execution Playbook
The snapshot workflow enforces visibility via `run_snapshot_workflow` with predefined
`DEFAULT_SNAPSHOT_STAGES`. Execute each stage relative to kickoff with the commands below. The
`--model-path` flag is optional if the latest artifact exists under `data/models/`.

### T-24 Hours
- **Command:**
  ```bash
  poetry run nfl-pred snapshot --season 2024 --week 7 --at "2024-10-13T17:00:00Z" \
    --feature-set mvp_v1 --model-id prod_v1 --full-timeline
  ```
  The CLI automatically schedules all stages; running at T-24h seeds downstream stages.
- **Inputs Allowed:** Historical ingestion outputs, weather backfill <= cutoff, injury reports
  filed before timestamp.
- **Validation:**
  - `duckdb data/nfl.duckdb "SELECT COUNT(*) FROM features WHERE snapshot_at='2024-10-12T17:00:00Z'"`
  - Review CLI output for "Executed 4 snapshot stage runs" confirmation.

### T-100 Minutes
- **Command:** Rerun the same snapshot invocation; the scheduler advances to the next stage when
  the wall clock reaches T-100m. To force only this stage, use:
  ```bash
  poetry run nfl-pred snapshot --season 2024 --week 7 --at "2024-10-13T17:00:00Z" \
    --feature-set mvp_v1 --model-id prod_v1 --full-timeline
  ```
  (no changes—stage execution is idempotent; rerun if previous attempt failed).
- **Inputs Allowed:** Late-week injury/practice participation, roster updates timestamped before
  cutoff. No inactives yet.
- **Validation:**
  - Inspect CLI stdout for stage completion messages.
  - `duckdb data/nfl.duckdb "SELECT COUNT(*) FROM features WHERE snapshot_at='2024-10-13T15:20:00Z'"`

### T-80 to T-75 Minutes
- **Command:**
  ```bash
  poetry run nfl-pred snapshot --season 2024 --week 7 --at "2024-10-13T17:00:00Z"
  ```
  (the CLI schedules both T-80m and T-75m runs).
- **Inputs Allowed:** Official inactives, depth charts, QB confirmations.
- **Validation:**
  - Re-run `poetry run nfl-pred ingest --seasons 2024` and confirm stdout indicates inactives refresh.
  - `duckdb data/nfl.duckdb "SELECT COUNT(*) FROM features WHERE snapshot_at BETWEEN '2024-10-13T15:40:00Z' AND '2024-10-13T15:45:00Z'"`

### T-60 Minutes (Final Freeze)
- **Command:**
  ```bash
  poetry run nfl-pred snapshot --season 2024 --week 7 --at "2024-10-13T17:00:00Z" --final-only
  ```
- **Outputs:** Predictions appended to DuckDB `predictions`, SHAP summaries under
  `data/models/shap/`, config + manifest logged in MLflow.
- **Validation:**
  - `duckdb data/nfl.duckdb "SELECT COUNT(*) FROM predictions WHERE snapshot_at='2024-10-13T16:00:00Z'"`
  - Export predictions for distribution: `duckdb data/nfl.duckdb "COPY (SELECT * FROM predictions WHERE snapshot_at='2024-10-13T16:00:00Z') TO 'exports/predictions_s2024_w7_T-60.csv' (HEADER, DELIMITER ',');"`
  - Post status in Slack `#nfl-ops` (use shared `/ops-broadcast` slash command).

## CLI Command Reference
| Workflow | Command | Notes |
| -------- | ------- | ----- |
| Ingestion | `poetry run nfl-pred ingest --seasons <year...>` | Pulls schedules, play-by-play, rosters, teams into `data/raw/`. |
| Feature Build | `poetry run nfl-pred build-features --seasons <year...> [--asof-ts ISO] [--snapshot-at ISO]` | Stores feature matrix in DuckDB with deterministic snapshot key. |
| Training | `poetry run nfl-pred train [--config PATH] [--feature-set ID]` | Logs MLflow run, writes `*.joblib` artifact under `data/models/`. |
| Inference | `poetry run nfl-pred predict --season N --week N [--model-path PATH]` | Generates predictions independent of snapshot ladder (use for ad-hoc scoring). |
| Snapshot Orchestration | `poetry run nfl-pred snapshot --season N --week N --at ISO [--final-only]` | Automates staged snapshots and predictions. |
| Reporting | `poetry run nfl-pred report --season N --week N` | Saves metrics/reliability CSV + PNG to `data/reports/`. |
| Monitoring | `poetry run nfl-pred monitor --season N --week N [--feature-set ID]` | Computes PSI, rolling metrics, populates `data/reports/monitoring/season_<season>/week_<week>/`. |
| MLflow Hygiene | `poetry run nfl-pred mlflow-hygiene [--dry-run/--no-dry-run]` | Applies retention policy per `configs/default.yaml`. |

All commands assume project root as CWD and environment managed via Poetry. Export `POETRY_VIRTUALENVS_IN_PROJECT=true` for local reproducibility.

## Monitoring and Alerting Protocols
1. **Scheduled Monitoring:** Run `poetry run nfl-pred monitor` every Thursday and Monday.
2. **Automated Thresholds:**
   - PSI breach when ≥5 features exceed PSI ≥0.20.
   - Rolling four-week Brier deterioration triggers alert at ≥10% worse than baseline.
   - Rule flag flips (see `data/ref/rule_flags.yml`) automatically raise warning.
3. **Alert Channels:**
   - Slack `#nfl-ops`: Primary notification channel via automation webhook.
   - PagerDuty service **NFL Prediction Ops**: Trigger manual incidents for severe degradations.
4. **Manual Validation:** Review `data/reports/monitoring/season_<season>/week_<week>/psi_summary_s<season>_w<week>.csv` and
   reliability plots. Confirm time stamps align with latest snapshot.
5. **Escalation:** If triggers fire, follow [Retrain Trigger Workflow](#retrain-trigger-workflow).

## Retrain Trigger Workflow
1. **Collect Inputs:** Use outputs from monitoring command plus baseline Brier from
   `data/reports/metrics_s<season>_w<week>.csv`.
2. **Evaluate:**
   ```bash
   python - <<'PY'
   from nfl_pred.monitoring.triggers import evaluate_retrain_triggers, RetrainTriggerConfig
   import pandas as pd
   recent = pd.read_csv('data/reports/monitoring/season_2024/week_07/expanded_metrics_s2024_w7.csv')
   recent_scores = recent.loc[recent['metric_window'] == 'weekly', 'brier_score'][-4:]
   baseline = float(pd.read_csv('data/reports/metrics_s2024_w3.csv')['brier_score'].iloc[0])
   from nfl_pred.reporting.monitoring_report import compute_monitoring_psi_from_features, load_feature_payloads
   from nfl_pred.config import load_config
   cfg = load_config('configs/default.yaml')
   features_df = load_feature_payloads(cfg.paths.duckdb_path, feature_set='mvp_v1')
   trigger = RetrainTriggerConfig()
   psi = compute_monitoring_psi_from_features(features_df, season=2024, week=7, psi_threshold=trigger.psi_threshold)
   decision = evaluate_retrain_triggers(
       recent_brier_scores=recent_scores,
       baseline_brier=baseline,
       psi_summary=psi,
       previous_rule_flags={'kickoff_touchback': True},
       current_rule_flags={'kickoff_touchback': False},
       config=trigger,
   )
   print(decision)
   PY
   ```
3. **If Triggered:**
   - Schedule training window (`poetry run nfl-pred train ...`).
   - After training, execute smoke predictions on past completed week.
   - Submit change request in release tracker and notify stakeholders.
4. **If Not Triggered:** Document decision in weekly ops log; no action required beyond routine
   monitoring.

## Model Promotion and Release Management
1. **Candidate Selection:** Identify MLflow runs meeting performance targets (Brier <= baseline,
   reliability slope within ±0.05). Tag run with `stage=candidate`.
2. **Validation Checklist:**
   - Reproduce inference on historical week: `poetry run nfl-pred predict --season 2023 --week 18 --model-path data/models/<artifact>.joblib`.
   - Verify predictions distribution vs. production: `duckdb data/nfl.duckdb "SELECT AVG(p_home_win) FROM predictions WHERE model_id='candidate_v2'"`.
   - Ensure SHAP artifacts exist under `data/models/shap/` for candidate model.
3. **Promotion:**
   - Update symlink `data/models/current.joblib -> <artifact>` (atomic `ln -sf`).
   - Record promotion in MLflow (set `stage=production`) and update run description.
   - Communicate promotion in Slack with validation metrics.
4. **Rollback Plan:** Maintain previous artifact symlink `data/models/previous.joblib`. Revert by
   switching symlink and re-running `poetry run nfl-pred snapshot --final-only` if needed.

## Release Management Cadence
- **Change Window:** Deploy model or pipeline changes Tuesday–Thursday before 18:00 UTC.
- **Pre-release Tests:** `poetry run pytest`; `poetry run nfl-pred build-features --seasons <recent> --write-mode replace` in staging DuckDB copy.
- **Documentation:** Update README, runbook, and ADRs; log release notes with model ID, data
  window, and calibration method.
- **Approvals:** Require sign-off from Modeling Lead and Ops Lead before promoting to production.

## Emergency Procedures
| Scenario | Immediate Actions | Follow-Up |
| -------- | ---------------- | --------- |
| **Snapshot Failure** (command exits non-zero) | Re-run with `--final-only` if failure at T-60m; capture stderr with `poetry run nfl-pred snapshot --final-only ... 2>&1 | tee logs/snapshot_failure.log`; if DuckDB locked, run `fuser data/nfl.duckdb` to identify processes and terminate safely. | Document incident, attach failure log, ensure rerun success recorded by saving CLI summary in the incident ticket. |
| **Data Corruption** (missing schedule rows) | Restore latest Parquet backup: `aws s3 cp s3://nfl-backups/schedules.parquet data/raw/ --recursive`; re-run ingestion for affected seasons. | Add post-mortem entry; verify counts via `duckdb data/nfl.duckdb "SELECT season, COUNT(*) FROM schedules GROUP BY season"`. |
| **Model Artifact Missing** | Resolve via symlink: `ln -sf data/models/previous.joblib data/models/current.joblib`; rerun final snapshot. | Investigate MLflow artifact retention settings; adjust hygiene policy if overly aggressive. |
| **PSI Spike Live Alert** | Pause promotion; run monitoring command manually to confirm; if confirmed, initiate retrain workflow and notify PagerDuty. | After mitigation, update monitoring thresholds or feature spec as needed. |

## Troubleshooting Guide
- **DuckDB Connectivity Error:** Ensure no other process holds lock; try `duckdb data/nfl.duckdb` from CLI and run `PRAGMA database_list;`.
- **MLflow Tracking URI Unreachable:** Verify `mlflow server` container; restart via `systemctl restart mlflow` on ops host; fall back to local file store by exporting `MLFLOW_TRACKING_URI=file:./mlruns`.
- **CLI Missing:** Activate environment `poetry shell` or use full path `poetry run nfl-pred ...`.
- **Visibility Violations:** Check `asof_ts` column in features for values > snapshot cutoff; rerun build with explicit `--asof-ts`.
- **Slow Snapshot Runs:** Inspect DuckDB indexes; vacuum via `duckdb data/nfl.duckdb "PRAGMA optimize"` during maintenance window.

## Contacts and Escalation
| Role | Primary Contact | Channel | Notes |
| ---- | --------------- | ------- | ----- |
| Ops Lead | Taylor Morgan | Slack `@taylor.ops` / PagerDuty | First responder for ingestion, snapshot issues. |
| Modeling Lead | Priya Shah | Slack `@priya.modeling` / Email `priya@nflpred.dev` | Approves model promotions and retrain plans. |
| Data Engineering | Sam Lee | Slack `@sam.data` | Handles DuckDB maintenance and data restores. |
| Product Owner | Alex Kim | Email `alex.kim@nflpred.dev` | Communicates release outcomes to stakeholders. |

**When escalating:**
1. Post summary in Slack `#nfl-ops` with incident ID and latest command output (copy from `poetry run nfl-pred <command> 2>&1 | tee logs/<incident>.log`).
2. Attach relevant log snippet using `tail -n 200 logs/<incident>.log` for quick context.
3. Create PagerDuty incident via web console or `curl -X POST https://events.pagerduty.com/v2/enqueue -H 'Content-Type: application/json' -d @payloads/pagerduty_high.json` (maintain payload templates under `payloads/`).
4. Update this runbook if new mitigation steps were required.

## Validation Checklist for Operators
- [ ] Latest DuckDB backup stored in `backups/` within last 24h (`ls -lt backups/`).
- [ ] `poetry run pytest` passes after any code/config change.
- [ ] `poetry run nfl-pred snapshot --final-only` executed within 60 minutes of kickoff.
- [ ] Monitoring artifacts (`data/reports/monitoring/`) refreshed within last 48h.
- [ ] MLflow hygiene task run within last 7 days (`poetry run nfl-pred mlflow-hygiene --dry-run`).

Keep this checklist alongside PagerDuty runbook for rapid confirmation during audits or shifts.
