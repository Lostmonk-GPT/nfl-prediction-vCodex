# NFL Game Outcome Prediction — PRD

**Version:** 1.0  
**Date:** 2025-09-17  
**Scope:** Pre-game win probability and pick generation for NFL games using public data.

---

## Table of Contents
1. Data Sources and Refresh
2. Game Snapshot Timeline
3. Stadiums and Weather
4. Rule-Change Guards
5. Features and Field Mapping
6. Label Policy
7. Training Horizon and Scope
8. Modeling Recipe
9. Evaluation and Reporting
10. Validation Plan
11. Picks and Confidence
12. Monitoring and Retraining
13. Reproducibility and Storage
14. Documentation Artifact
15. Appendix A — Visibility Rules

---

## 1) Data Sources and Refresh
- **Primary library:** `nflreadpy` for nflverse datasets: play-by-play, schedules, rosters, weekly rosters, participation, injuries, officials, team info. Nightly refresh any time after 12:00 AM ET.
- **Upstream cadence references:** nflreadr dictionaries and schedule loader; schedules update frequently in season.

---

## 2) Game Snapshot Timeline (pre-game only)
- **T-24h:** baseline roster/participation, travel, priors.  
- **T-100m:** first injury/actives sweep.  
- **T-80–75m:** “post-inactives” capture.  
- **T-60m:** freeze inputs; generate probabilities and picks.  
Supporting rule: clubs deliver inactive list in the officiating meeting ~90 minutes before kickoff.

---

## 3) Stadiums and Weather
- **Canonical stadium table:** `venue`, `team(s)`, `lat`, `lon`, `tz`, `altitude`, `surface`, `roof ∈ {indoors,dome,open,retractable}`, `neutral_site`. Schedules include a cleaned `roof` field, but this table is authoritative.
- **Forecasts:** NWS API flow `/points/{lat,lon}` → `/gridpoints/{wfo}/{x},{y}/forecast` (or hourly). Use nearest station ≤10 mi. Unit normalization required.
- **Historical:** Meteostat Python, choose nearest station ≤10 mi via `Stations.nearby` or `Point`.
- **Indoor handling:** if `roof ∈ {indoors,dome,closed}`, set weather features null/zero.

---

## 4) Rule-Change Guards
- **Binary flags only (now):** `kickoff_2024plus`, `ot_regular_2025plus`.

---

## 5) Features and Field Mapping
- **From PBP (team-week aggregates, windows 4/8/season):** EPA/play, early-down EPA, success rate, pass/run rate, play-action rate (`play_action`), shotgun rate (`shotgun`), no-huddle rate (`no_huddle`), sack rate, QB hits per dropback, explosive pass/run rates (e.g., `air_yards≥15`, rush≥10), special-teams EPA, penalties/play, home/away, rest days. Map to nflfastR columns.
- **From schedules/rosters:** kickoff time bucket, surface, roof, bye week, short-week flag.
- **From injuries/participation:** counts of DNP/LP/FP by position group at snapshot week.
- **From stadium/weather:** forecast temp/wind/precip for outdoor/open/retractable-open; null if indoor.
- **Travel:** great-circle miles previous venue → current venue (haversine) and `neutral_site` plus `days_since_last`.
- **Home-field advantage:** derived venue-level rolling features `hfa_rolling_3y`, `hfa_yoy_delta`.

---

## 6) Label Policy
- Binary team-win label. Ties count as 0.5 for calibration plots and are included in metrics.

---

## 7) Training Horizon and Scope
- Primary fit on rolling 3–5 recent seasons. Run ablations with older seasons and season weighting.  
- Exclude preseason. Playoffs: separate fit or special flags.

---

## 8) Modeling Recipe
- **Level-0 models:** logistic regression, gradient-boosted trees (XGBoost/LightGBM), ridge.  
- **Stacking:** out-of-fold probabilities → logistic meta-learner.  
- **Final calibration:** isotonic or Platt on held-out set.  
- **Explainability:** SHAP TreeExplainer on tree models; sample 10–20% rows per week; fall back to approximate/GPU mode if slow.

---

## 9) Evaluation and Reporting
- **Metrics:** Brier score, log loss; reliability diagrams and calibration error; report by week and by favorite/underdog buckets.
- **Cadence:** weekly, season-to-date, rolling 4-week.

---

## 10) Validation Plan
- Forward-chaining CV by NFL week using grouped weekly folds (`TimeSeriesSplit`-style). No future leakage.

---

## 11) Picks and Confidence
- **Pick rule:** choose team with `p(win) ≥ 0.5`.  
- **Confidence tiers:** Strong ≥0.65; Lean 0.55–0.65; Pass <0.55.  
- **Optional pool mode:** maximize expected points when applicable.

---

## 12) Monitoring and Retraining
- **Weekly checks:** calibration charts; Brier/log-loss by favorite/underdog; PSI on key features (alert at PSI ≥0.2).  
- **Retrain triggers:** any of  
  1) 4-week rolling Brier worsens ≥10% vs season baseline,  
  2) PSI ≥0.2 on ≥5 key features,  
  3) rule-flag flip (kickoff/OT).  

---

## 13) Reproducibility and Storage
- **Storage:** DuckDB over Parquet; keys = `season`, `week`, `game_id`, `snapshot_at`.  
- **Experiment tracking:** MLflow for params, metrics, artifacts; pin env via `pyproject` + lockfile.

---

## 14) Documentation Artifact
- Maintain a living “feature spec” table with: name, definition, source columns, window, snapshot timing, null policy, rule notes.  
  Examples (source → mapping):  
  - `play_action_rate` → PBP `play_action` boolean.  
  - `shotgun_rate`, `no_huddle_rate` → PBP `shotgun`, `no_huddle`.  
  - `explosive_pass_rate` → PBP `air_yards`.  
  - `roof`/`surface` → schedules.  
  - `travel_miles` → haversine.  
  - `wx_temp`, `wx_wind` → NWS forecast for outdoor/open.

---

# Appendix A — Visibility Rules (Pre-Game Freeze)

## Objective
Use only information available before kickoff with a deterministic cutoff.

## Cutoffs
- Official inactives ~T-90m. Take a post-inactives snapshot at T-80–75m. Freeze all inputs at **T-60m** and produce picks.

## Allowed data by snapshot
- **T-24h:** season-to-date and rolling stats, travel, venue metadata, historical weather backfill, preliminary injuries/participation.  
- **T-100m:** refresh injuries/participation and roster status.  
- **T-80–75m:** incorporate official inactives; finalize starting QB if confirmed.  
- **T-60m:** lock features; output probabilities, picks, explanations.

## Weather visibility
- Outdoors/open/retractable-open only. Query NWS `/points` then `/gridpoints/.../forecast` (or hourly). Persist raw and normalized features; nearest station within 10 miles.  
- Historical backfill via Meteostat nearest station ≤10 mi. If indoor/closed, set weather features null/zero.

## Stadium visibility
- Stadium table is source of truth for roof/surface/timezone/altitude/neutral site; schedules’ `roof` is cleaned upstream.

## Rule-aware visibility
- Include `kickoff_2024plus` and `ot_regular_2025plus` in the feature matrix; avoid in-game kickoff structure features in pre-game models.

## Travel visibility
- Compute `travel_miles` with haversine between previous and current venues at T-24h; update only if venue changes. Set `neutral_site` from schedule/stadium table.

## Enforcement
- All feature builders accept `asof_ts` and must filter sources to `event_time ≤ asof_ts`.  
- Unit test: replay a historical game with `asof_ts = T-60m` and assert no post-cutoff reads.

## Outputs at freeze
- `p_home_win`, `p_away_win` (sum to 1), pick, confidence tier, weekly SHAP summary on ≤20% sampled rows.

## Audit trail
- Persist: snapshot timestamps, upstream dataset versions, model hash, code version, feature spec checksum, and input row hashes in DuckDB; log run in MLflow.

--- 

**End of PRD.**
