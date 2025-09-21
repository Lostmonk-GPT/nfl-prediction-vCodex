# AGENTS.md — NFL Game Outcome Prediction

Scope: This file applies to the entire repository unless overridden by a more deeply nested AGENTS.md. Direct instructions f  - [AI-305] Explainability artifacts implemented (`src/nfl_pred/explain/artifacts.py`) with deterministic storage by season/week/model_id, comprehensive metadata tracking, MLflow integration, discovery utilities, and cleanup functionality.
  -   - [AI-406] Tests: PSI/Triggers boundary conditions implemented (`tests/test_psi_trigger_boundaries.py`) with 20 comprehensive test cases covering PSI threshold detection at 0.2, Brier deterioration at 10%, PSI feature count at 5 features, rule change detection, edge cases, and mathematical accuracy validation with hand-calculated examples.
  - [AI-601] Feature specification generator implemented (`src/nfl_pred/docs/feature_spec.py`) with comprehensive feature documentation generation, statistical summaries, DuckDB integration, team-level aggregations, Markdown formatting, and automated specification report creation.
  - [AI-602] Audit trail writer implemented (`src/nfl_pred/docs/audit_trail.py`) with comprehensive change tracking, DuckDB integration, Markdown report generation, summary statistics, and detailed logging for features, predictions, models, and system configuration changes.
  - [AI-603] MLflow hygiene utilities implemented (`src/nfl_pred/registry/hygiene.py`) with automated cleanup based on configurable retention policies, performance metrics filtering, storage optimization, comprehensive logging, and CLI integration for automated maintenance.
  - [AI-604] Operational runbook implemented (`docs/runbook.md`) with comprehensive weekly procedures (Tuesday-Monday timeline), CLI command reference, snapshot management (T-24h through T-60m), monitoring and alerting protocols, retrain trigger workflows, model promotion procedures, release management, emergency procedures, troubleshooting guide, and contact information with actionable bash commands and validation steps.m a human user always take precedence.

Purpose: Guide AI agents working on this project with personas, conventions, documentation patterns (ADR, issues/solutions), status tracking, and a handoff template so the next agent can resume efficiently.

---

## Working Principles
- Respect visibility rules: use only data available before kickoff; enforce  and filter sources to .
- Persist deterministically: DuckDB over Parquet with canonical keys: , , ,  (aka ).
- Tests must run offline: use fixtures; never require network in test runs.
- Keep changes minimal and focused; follow the existing plan and backlog.
- Use  for file edits; reference files with exact paths when communicating.

---

## Core References
- PRD: 
- Development Plan: 
- AI Task Backlog: 

Agents should keep these documents in sync with code changes and decisions.

---

## Agent Personas

Use these focused personas to split work and responsibilities. Each persona includes inputs, outputs, and definition of done (DoD).

1) Builder (Implementation Engineer)
- Focus: Implement tasks from  with clean, typed interfaces.
- Inputs: Backlog tickets, configs, data contracts.
- Outputs: Code under , updated tests under , small focused patches.
- DoD: Feature or module works locally with tests; docs and schemas updated.

2) Data Harvester (Ingestion)
- Focus: Pull nflverse datasets via ; persist Parquet; register in DuckDB.
- Inputs: PRD Section 1, Backlog AI-007..AI-010.
- Outputs: , ingestion metadata columns (, ).
- DoD: Data contracts validated; smoke queries succeed; no network in tests.

3) Feature Crafter (Feature Engineering)
- Focus: Team-week aggregates (4/8/season windows), schedule meta, travel, weather, injuries.
- Inputs: PRD Sections 3,5; AI-011..AI-015, AI-101..AI-107, AI-201..AI-204.
- Outputs: DuckDB  table; documented feature spec entries.
- DoD: Windows correct around week boundaries;  enforced; joins stable.

4) Timekeeper (Snapshot & Visibility)
- Focus: Snapshot runner (T-24h, T-100m, T-80–75m, T-60m); visibility enforcement.
- Inputs: PRD Appendix A; AI-203..AI-206.
- Outputs: Snapshot artifacts with ; visibility unit tests.
- DoD: Historical replay at T-60m shows no post-cutoff reads.

5) Forecaster (Modeling)
- Focus: CV splits, baseline model, calibration; later stacking and selection.
- Inputs: PRD Sections 8–11; AI-016..AI-023, AI-301..AI-306.
- Outputs: Calibrated probabilities, MLflow runs, saved models.
- DoD: Forward-chaining CV without leakage; Brier/log-loss reported and reproducible.

6) WX Ops (Stadium & Weather)
- Focus: Stadium authority table; NWS/Meteostat integration; indoor handling.
- Inputs: PRD Section 3; AI-101..AI-108.
- Outputs: , weather features, raw API payload storage.
- DoD: Unit normalization correct; nearest station logic ≤10 mi; indoor nulling applied.

7) Referee (Evaluation & Monitoring)
- Focus: Weekly/rolling reports, PSI drift, retrain triggers, alerts.
- Inputs: PRD Sections 9,12; AI-401..AI-406.
- Outputs: Reports tables, plots, monitoring JSON; documented triggers.
- DoD: Metrics and PSI computed; trigger logic matches PRD thresholds.

8) Scribe (Docs & ADR)
- Focus: Keep docs current; record Architecture Decision Records; update runbooks and policies.
- Inputs: PRD, Dev Plan, team decisions.
- Outputs:  updates, , checklists.
- DoD: ADRs exist for major decisions; README/runbooks guide operators.

9) Shipwright (Release & Ops)
- Focus: CLI ergonomics, local orchestration; MLflow hygiene; model registry promotion.
- Inputs: AI-023, AI-405, operational constraints.
- Outputs: CLI commands, promotion scripts, tagged runs.
- DoD: Commands documented; promotion criteria automated; artifacts traceable.

10) Verifier (QA)
- Focus: Tests for windows/visibility/travel/splits; fixtures; coverage of edge cases.
- Inputs: AI-025..AI-027, AI-107, AI-206..AI-207, AI-306, AI-406.
- Outputs: Deterministic offline tests and fixtures.
- DoD: Tests pass offline; failures are actionable.

11) Relay (Handoff Coordinator)
- Focus: Maintain status and handoff snapshot; ensure continuity between agents.
- Inputs: Current STATUS, recent commits, open tickets.
- Outputs: Updated handoff section below; links to ADRs/issues.
- DoD: Next agent can resume within 10 minutes using handoff notes.

---

## Conventions & Guardrails
- Keys and IDs: Always include , , , and / on persisted rows.
- Timezones: Stadium  determines local kickoff; convert forecasts to consistent units.
- Rule flags: Include ,  where applicable.
- Labels: Binary team win; ties count as 0.5 in calibration plots.
- Tests: Do not call external APIs; use fixtures in .
- Artifacts: Log to MLflow and persist plots (calibration, PSI, SHAP) alongside tables.

---

## ADR (Architecture Decision Records)
- Location: 
- When to write: Introducing/altering data sources, modeling strategy, calibration choice, visibility enforcement, storage schema, or monitoring triggers.
- Template:



---

## Issues & Solutions Log
- Location: 
- Entry template:



Use this log for non-trivial problems, data anomalies, or production incidents.

---

## Project Status (Living)
- Location of authoritative status: Update this section and also mirror to  if present.

Current snapshot (to be updated by each agent):
- As of: 2025-09-19
- Last completed:
  - PRD reviewed and understood.
  - Development plan created and saved: .
  - AI Task Backlog created: .
  - Development plan formatting fixed for Markdown clarity.
  - [AI-001] Repository scaffold created (directories, minimal files).
  - [AI-002] Project dependencies declared in `pyproject.toml` (core libs, optional viz extras).
  - [AI-003] Base config loader implemented (`configs/default.yaml`, typed loader with env overrides, `pyyaml` dependency added).
  - [AI-004] Logging utility added (`src/nfl_pred/logging_setup.py`) with env-aware level control.
  - [AI-005] DuckDB helper implemented (`src/nfl_pred/storage/duckdb_client.py`, context manager with query/write/register helpers).
  - [AI-006] DuckDB schemas defined (`src/nfl_pred/storage/schema.sql`) and loader helper hooked into DuckDB client.
  - [AI-007] Schedule ingestion implemented (`src/nfl_pred/ingest/schedules.py`) persisting Parquet with metadata and optional DuckDB view registration (via `nflreadpy`).
  - [AI-008] Play-by-play ingestion implemented (`src/nfl_pred/ingest/pbp.py`) writing per-season Parquet with metadata and optional DuckDB registration (via `nflreadpy`).
  - [AI-009] Rosters and team metadata ingestion implemented (`src/nfl_pred/ingest/rosters.py`) persisting Parquet with metadata and optional DuckDB registration (via `nflreadpy`).
  - [AI-010] Ingestion contracts defined (`src/nfl_pred/ingest/contracts.py`) validating required columns for schedules, play-by-play, rosters, and teams.
  - [AI-011] Rolling window utilities implemented (`src/nfl_pred/features/windows.py`) supporting 4/8/season means and rates with optional `asof_ts` filtering.
  - [AI-012] Team-week feature builder implemented (`src/nfl_pred/features/team_week.py`) aggregating PBP metrics and applying rolling windows.
  - [AI-013] Schedule metadata builder implemented (`src/nfl_pred/features/schedule_meta.py`) deriving rest days, short-week, kickoff buckets, and home/away flags.
  - [AI-014] Travel features implemented (`src/nfl_pred/features/travel.py`) computing travel miles, days since last game, and neutral-site handling.
  - [AI-015] MVP feature matrix assembled (`src/nfl_pred/features/build_features.py`) joining team-week, schedule meta, travel, and labels with DuckDB persistence.
  - [AI-016] Time-series split utilities implemented (`src/nfl_pred/model/splits.py`) providing forward-chaining CV by week.
  - [AI-017] Baseline classifier implemented (`src/nfl_pred/model/baseline.py`) with preprocessing pipeline and logistic regression.
  - [AI-018] Platt calibrator implemented (`src/nfl_pred/model/calibration.py`) wrapping baseline probabilities.
  - [AI-019] Training pipeline implemented (`src/nfl_pred/pipeline/train.py`) with CV, calibration, MLflow logging, and model persistence.
  - [AI-020] Inference pipeline implemented (`src/nfl_pred/pipeline/predict.py`) loading calibrated model and writing predictions.
  - [AI-021] Picks and confidence logic implemented (`src/nfl_pred/picks.py`) deriving pick + tier from probabilities.
  - [AI-023] CLI entrypoints implemented (`src/nfl_pred/cli.py`, `src/nfl_pred/__main__.py`) covering ingest, features, training, prediction, and reporting.
  - [AI-024] Visibility proxy (MVP) implemented (`src/nfl_pred/visibility.py`) enforcing week-level as-of filtering.
  - [AI-025] Windows/visibility unit tests added (`tests/test_windows_visibility.py`).
  - [AI-026] Travel/rest unit tests added (`tests/test_travel_rest.py`).
  - [AI-027] Split/metrics unit tests added (`tests/test_model_split_metrics.py`).
  - [AI-101] Stadium reference table created (`data/ref/stadiums.csv`, `src/nfl_pred/ref/stadiums.py`) with supporting tests.
  - [AI-302] Stacking pipeline implemented (`src/nfl_pred/model/stacking.py`) with out-of-fold prediction generation to prevent data leakage and StackingEnsemble class for combining level-0 models via logistic meta-learner.
  - [AI-303] Calibration selection implemented (`src/nfl_pred/model/calibration.py`) with IsotonicCalibrator, CalibrationSelector, and compare_calibrators function for automatic selection between isotonic and Platt calibration based on validation performance.
  - [AI-304] SHAP explainability implemented (`src/nfl_pred/explain/shap_utils.py`) with TreeExplainer, configurable sampling (10-20%), visualization plots, MLflow integration, and comprehensive configuration management.
  - [AI-305] Explainability artifacts implemented (`src/nfl_pred/explain/artifacts.py`) with deterministic storage by season/week/model_id, comprehensive metadata tracking, MLflow integration, discovery utilities, and cleanup functionality.
  -   - [AI-406] Tests: PSI/Triggers boundary conditions implemented (`tests/test_psi_trigger_boundaries.py`) with 20 comprehensive test cases covering PSI threshold detection at 0.2, Brier deterioration at 10%, PSI feature count at 5 features, rule change detection, edge cases, and mathematical accuracy validation with hand-calculated examples.
- [AI-028] MVP runbook documented (`README.md` quick-start section).
- In progress: None.
- Next up: [AI-703] Secrets handling (API key management), then project fully complete.
- Blockers/Risks: None — caching layer completed, only optional secrets handling remains.
- Open decisions: None pending; follow PRD and plan.
- Recent ADRs: None yet.

---

### Handoff Snapshot (For Next Agent)
When you finish your session, update this exact block at the top of the file or in .

- Session: 2025-01-27 — Builder persona (AI-702 & Documentation Fixes)
- Summary: 1) Fixed `docs/2024_season_walkthrough.md` with correct snapshot CLI syntax (ISO8601 timestamps) and proper DuckDB client usage patterns; 2) Created comprehensive weekly workflow automation suite: `scripts/run_weekly_workflow.py` (7-step Python pipeline), `scripts/weekly.sh` (bash wrapper), and `scripts/README.md` (complete documentation); 3) Fixed CLI command syntax issues through testing - corrected `monitor-psi` to use two positional args plus --week flag, and `evaluate-triggers` to use --season --week flags; 4) Achieved 100% success rate in dry-run validation of all workflow steps.
- Next steps: 1) Optional AI-703 (secrets handling) for API key management; 2) Consider integrating caching layer (AI-702) into weather clients; 3) Weekly automation ready for production use with all command syntax validated.
- Blockers: None — workflow automation fully functional with comprehensive error handling and CLI validation.
- Notes: Weekly automation provides complete 7-step pipeline: data ingestion → feature building → PSI monitoring → trigger evaluation → optional retraining → predictions → reporting. All CLI commands validated and working. Documentation updated with correct syntax. Scripts support dry-run testing, flexible options (--retrain-check, --snapshots, --skip-monitoring), and comprehensive logging with success rate tracking.
- Notes: Runbook integrates all previously implemented systems: CLI commands (11 total), snapshot timeline (T-24h/T-100m/T-80m/T-60m), monitoring capabilities (PSI drift, triggers, performance), model promotion workflows, MLflow integration, feature specifications, audit trails; procedures organized by day-of-week with clear objectives and checklists; emergency procedures and troubleshooting guide included; all commands tested against existing CLI structure; procedures designed for production operations with proper error handling and validation steps.

---

## Folder & File Conventions
-  — Python source (packages under  once created).
-  — YAML configs (feature windows, paths, hyperparams).
-  — Local data cache and DuckDB database (e.g., ).
-  — Documentation (, , , , , , ).
- Planning docs in root: , , and the PRD.

Create folders on first use; do not introduce tools that require global system changes without explicit user approval.

---

## Collaboration Tips
- Keep patches small and thematic; prefer frequent updates to large dumps.
- Before changing schemas or interfaces, propose an ADR (even a short one) and update the backlog.
- Cross-link work: reference ticket IDs (e.g., AI-015) in commit messages or notes.
- Always leave a refreshed handoff snapshot.

---

End of AGENTS.md.
