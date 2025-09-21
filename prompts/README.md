# AI Task Prompts Index

Use these prompts to guide an AI coding agent. Each prompt is self-contained and follows a consistent structure (Persona, Objective, Context, Deliverables, Constraints, Steps, DoD, Verification).

Related docs: `AGENTS.md`, `AI_Task_Backlog.md`, `NFL_Game_Outcome_Prediction_Dev_Plan.md`.

---

## MVP (Weeks 1–2)
- [AI-001 Create Repository Scaffold](AI-001_Create_Repository_Scaffold.md)
- [AI-002 Define Dependencies and Python Version](AI-002_Define_Dependencies_and_Python_Version.md)
- [AI-003 Base Config and Loader](AI-003_Base_Config_and_Loader.md)
- [AI-004 Logging Setup](AI-004_Logging_Setup.md)
- [AI-005 DuckDB Helper](AI-005_DuckDB_Helper.md)
- [AI-006 DuckDB Schemas](AI-006_DuckDB_Schemas.md)
- [AI-007 Ingestion: Schedules](AI-007_Ingestion_Schedules.md)
- [AI-008 Ingestion: Play-by-Play](AI-008_Ingestion_Play_By_Play.md)
- [AI-009 Ingestion: Rosters/Team Info](AI-009_Ingestion_Rosters_Team_Info.md)
- [AI-010 Data Contracts/Validators](AI-010_Data_Contracts_Validators.md)
- [AI-011 Feature Windows Utilities](AI-011_Feature_Windows_Utilities.md)
- [AI-012 Core Team-Week Features](AI-012_Core_Team_Week_Features.md)
- [AI-013 Rest Days and Kickoff Bucket](AI-013_Rest_Days_and_Kickoff_Bucket.md)
- [AI-014 Travel Features](AI-014_Travel_Features.md)
- [AI-015 Assemble MVP Feature Matrix](AI-015_Assemble_MVP_Feature_Matrix.md)
- [AI-016 Modeling: Data Split](AI-016_Modeling_Data_Split.md)
- [AI-017 Modeling: Baseline Classifier](AI-017_Modeling_Baseline_Classifier.md)
- [AI-018 Calibration (Platt)](AI-018_Calibration_Platt.md)
- [AI-019 Training Pipeline](AI-019_Training_Pipeline.md)
- [AI-020 Inference Pipeline](AI-020_Inference_Pipeline.md)
- [AI-021 Picks and Confidence](AI-021_Picks_and_Confidence.md)
- [AI-022 Reporting: Metrics + Reliability](AI-022_Reporting_Metrics_Reliability.md)
- [AI-023 CLI Entrypoints](AI-023_CLI_Entrypoints.md)
- [AI-024 Visibility Proxy (MVP)](AI-024_Visibility_Proxy_MVP.md)
- [AI-025 Tests: Windows/Visibility](AI-025_Tests_Windows_Visibility.md)
- [AI-026 Tests: Travel/Rest](AI-026_Tests_Travel_Rest.md)
- [AI-027 Tests: Model Split and Metrics](AI-027_Tests_Model_Split_Metrics.md)
- [AI-028 Document MVP Runbook](AI-028_Document_MVP_Runbook.md)

---

## Phase 1 — Stadium Authority & Weather
- [AI-101 Authoritative Stadium Table](AI-101_Authoritative_Stadium_Table.md)
- [AI-102 Stadium Join Logic](AI-102_Stadium_Join_Logic.md)
- [AI-103 NWS Client](AI-103_NWS_Client.md)
- [AI-104 Meteostat Client](AI-104_Meteostat_Client.md)
- [AI-105 Weather Feature Builder](AI-105_Weather_Feature_Builder.md)
- [AI-106 Weather Artifacts and Metadata](AI-106_Weather_Artifacts_and_Metadata.md)
- [AI-107 Weather Tests with Fixtures](AI-107_Weather_Tests_with_Fixtures.md)
- [AI-108 Docs Update for Weather](AI-108_Docs_Update_for_Weather.md)

---

## Phase 2 — Injuries/Participation & Snapshot Timeline
- [AI-201 Injuries Ingestion](AI-201_Injuries_Ingestion.md)
- [AI-202 Position-Group Rollups](AI-202_Position_Group_Rollups.md)
- [AI-203 Snapshot Runner](AI-203_Snapshot_Runner.md)
- [AI-204 Visibility Enforcement](AI-204_Visibility_Enforcement.md)
- [AI-205 Snapshot CLI](AI-205_Snapshot_CLI.md)
- [AI-206 Historical Replay Test](AI-206_Historical_Replay_Test.md)
- [AI-207 Injury Visibility Tests](AI-207_Injury_Visibility_Tests.md)
- [AI-208 Docs: Snapshot Timeline](AI-208_Docs_Snapshot_Timeline.md)

---

## Phase 3 — Modeling Enhancements & Explainability
- [AI-301 Additional Level-0 Models](AI-301_Additional_Level_0_Models.md)
- [AI-302 Stacking Pipeline](AI-302_Stacking_Pipeline.md)
- [AI-303 Calibration Selection](AI-303_Calibration_Selection.md)
- [AI-304 SHAP Explainability](AI-304_SHAP_Explainability.md)
- [AI-305 Explainability Artifacts](AI-305_Explainability_Artifacts.md)
- [AI-306 Tests: Stacking/SHAP](AI-306_Tests_Stacking_SHAP.md)

---

## Phase 4 — Evaluation, Monitoring, Retraining
- [AI-401 Expanded Evaluation Reports](AI-401_Expanded_Evaluation_Reports.md)
- [AI-402 PSI Drift Monitoring](AI-402_PSI_Drift_Monitoring.md)
- [AI-403 Retrain Triggers](AI-403_Retrain_Triggers.md)
- [AI-404 Monitoring CLI/Report](AI-404_Monitoring_CLI_Report.md)
- [AI-405 MLflow Model Registry Hooks](AI-405_MLflow_Model_Registry_Hooks.md)
- [AI-406 Tests: PSI/Triggers](AI-406_Tests_PSI_Triggers.md)

---

## Phase 5 — Rule-Change Guards & Postseason
- [AI-501 Rule Flags](AI-501_Rule_Flags.md)
- [AI-502 Playoff Handling](AI-502_Playoff_Handling.md)
- [AI-503 Rule-Change Backfill Policy](AI-503_Rule_Change_Backfill_Policy.md)
- [AI-504 Tests: Rule Flags/Playoffs](AI-504_Tests_Rule_Flags_Playoffs.md)

---

## Phase 6 — Reproducibility & Documentation
- [AI-601 Feature Spec Generator](AI-601_Feature_Spec_Generator.md)
- [AI-602 Audit Trail Writer](AI-602_Audit_Trail_Writer.md)
- [AI-603 MLflow Hygiene](AI-603_MLflow_Hygiene.md)
- [AI-604 Operational Runbook](AI-604_Operational_Runbook.md)
- [AI-605 PRD Conformance Checklist](AI-605_PRD_Conformance_Checklist.md)

---

## Cross-Cutting
- [AI-701 Configurable Paths and Env](AI-701_Configurable_Paths_and_Env.md)
- [AI-702 Caching Layer](AI-702_Caching_Layer.md)
- [AI-703 Secrets Handling](AI-703_Secrets_Handling.md)
- [AI-704 CI Test Data Fixtures](AI-704_CI_Test_Data_Fixtures.md)

---

## How To Use
- Open a prompt file and provide its full contents to your AI coding agent as the current task.
- Keep AGENTS.md open for persona, guardrails, and handoff guidance.
- Progress through prompts sequentially or by phase, updating status in AGENTS.md.

