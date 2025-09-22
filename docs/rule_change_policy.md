# Rule-Change Backfill Policy

## Purpose and Scope
This document defines how the project handles historical training data when the NFL introduces
material rule changes, aligning with PRD Section 4 (League Rule Tracking) and Section 7 (Modeling
Data Windows). The policy governs data selection, weighting, and documentation practices for
regular-season and postseason games used by the feature pipeline, training job, and monitoring
reports.

## Definitions
- **Rule Change Category** — Each change recorded in the rule registry is labeled as `minor`,
  `material`, or `transformational` per PRD guidance.
- **Transition Window** — The set of seasons immediately before and after the effective date in
  which historical data may need re-weighting or exclusion.
- **Evaluation Horizon** — The seasons included in model training, validation, and monitoring
  comparisons.

## Governance Workflow
1. **Catalog the change**: When a rule change enters PRD Appendix B, update the rule registry
   (`src/nfl_pred/features/rules.py`) with the effective season, category, and impacted play types.
2. **Classify severity**: Assign the category using the officiating rubric from PRD Section 4. The
   category drives the inclusion rules below.
3. **Decide policy**: Apply the corresponding inclusion/weighting table and log the decision in the
   audit trail (see `src/nfl_pred/docs/audit_trail.py`).
4. **Configure pipeline**: Update the training configuration (`configs/*.yaml`) with the desired
   season range and weights. The policy is enforced by the training pipeline (PRD Section 7) via the
   dataset builder.
5. **Document for operators**: Append a summary entry to the operational runbook and include the
   rationale in the weekly monitoring report for transparency.

## Inclusion and Weighting Rules
| Category | Included Seasons | Weighting Approach | Notes |
|----------|------------------|--------------------|-------|
| Minor (e.g., clock management tweak) | Last 6 seasons, no exclusions. | Uniform weights (1.0). | Monitor PSI for early drift but no retrain required. |
| Material (e.g., kickoff positioning, overtime format) | Last 6 seasons with the two seasons prior to the change down-weighted. | Seasons `T-2` and `T-1` receive weights 0.5 and 0.75, seasons `T` onward weight 1.0. | Trigger mid-season review if PSI or rule flag breach occurs. |
| Transformational (e.g., extra point distance, two-point rule overhaul) | Drop all pre-change seasons except `T-1` for calibration. | Only `T-1` retained at weight 0.25; seasons `T` and `T+1` weight 1.0; defer use of `T+2+` until available. | Run accelerated retrain once four weeks of `T` data exists. |

`T` denotes the season the rule change takes effect for scheduling purposes. When a rule is
announced in offseason `YYYY` and applies to games in `YYYY+1`, the policy treats `T` as `YYYY+1`
consistent with the rule flag implementation.

### Handling Multiple Concurrent Changes
- Evaluate each change independently; the most restrictive category wins for any overlapping
  seasons.
- If two material changes overlap, add the weight adjustments (e.g., cap at 1.25) only for
  post-change seasons to emphasize rapid adaptation.
- If a transformational change overlaps with any other category, follow the transformational
  policy exclusively.

## Backfill Procedure
1. Generate a season-level summary of rule flags and impacted features using the explainability
   artifacts module (PRD Section 7).
2. For excluded seasons, retain raw ingestion outputs but mark them with `visibility = historical`
   so the inference pipeline ignores them without deleting source data.
3. When weights change, persist the weighting vector alongside the training dataset metadata so
   MLflow runs can reproduce the exact composition.
4. Update monitoring baselines to match the evaluation horizon; PSI comparisons should use the same
   weighted distribution to avoid false alarms.

## Examples
- **2023 kickoff touchback rule (Material)**: Include seasons 2018–2024. Down-weight 2021 and 2022 as
  0.5 and 0.75 respectively, treat 2023 onward at 1.0. Schedule a follow-up calibration check after
  Week 6 of 2023 because special-teams rates shift quickly.
- **2015 extra point distance change (Transformational)**: Retain 2014 at 0.25 weight for red-zone
  baselines, exclude seasons prior to 2014. Retrain the kicker-related submodels once Weeks 1–4 of
  2015 are complete to avoid mixing pre-change dynamics.
- **2021 taunting enforcement emphasis (Minor)**: No season exclusions; rely on monitoring triggers
  to detect if penalty trends materially affect win probabilities.

## Compatibility Notes
- The training pipeline reads seasonal filters and weights from configuration. Store weights in
  `training.rule_change_weights` so automated runs remain deterministic.
- Feature builders already append rule flags; ensure backfill jobs re-run `append_rule_flags` after
  any schedule updates so the monitoring layer can surface rule-triggered retrain recommendations.
- Monitoring triggers combine PSI, Brier deterioration, and `rule_change` flags. Updating the policy
  requires verifying that `tests/test_psi_trigger_boundaries.py` still passes without modification.

## Change Management
- Log policy adjustments in the audit trail (PRD Section 7) and update the AI Task Backlog with a
  short note referencing the relevant rule change ID.
- When policy updates alter training scope, create an ADR summarizing the decision and link it from
  the runbook for operator awareness.
- Review this policy annually during the offseason checklist to ensure new rule categories are
  accounted for.
