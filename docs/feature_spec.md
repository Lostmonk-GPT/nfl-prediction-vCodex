# Feature Specification

Generated on 2025-09-22 01:27:43 UTC from DuckDB database `data/nfl.duckdb` using feature set `mvp_v1`.
No persisted feature rows were available; documentation defaults to static metadata.

## Feature Documentation

### Keys & Context

| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `asof_ts` | Visibility cutoff timestamp applied to source data. | Snapshot metadata stored with each payload row. | N/A | Defines the visibility boundary for included inputs. | Not nullable; timezone-aware UTC timestamp. | — |
| `game_id` | nflverse game identifier for the matchup. | Schedule ingestion via `compute_schedule_meta`. | N/A | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Not nullable; string identifier. | — |
| `home_away` | Game site classification for the team (home/away/neutral). | Schedule ingestion via `compute_schedule_meta`. | N/A | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Not nullable; neutral overrides home/away flags when detected. | — |
| `opponent` | Opponent abbreviation for the matchup. | Schedule ingestion via `compute_schedule_meta`. | N/A | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Not nullable; mirrors schedule pairings. | — |
| `season` | Season identifier for the game row. | Schedule ingestion via `compute_schedule_meta`. | N/A | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Not nullable; coerced to integer. | — |
| `snapshot_at` | Timestamp when the feature snapshot executed. | Snapshot metadata stored with each payload row. | N/A | Recorded post-run to support reproducibility. | Not nullable; timezone-aware UTC timestamp. | — |
| `start_time` | UTC kickoff timestamp for the game. | Schedule ingestion via `compute_schedule_meta`. | N/A | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | May be null when kickoff timestamp unavailable in schedule. | — |
| `team` | Team associated with the row using nflfastR abbreviations. | Schedule ingestion via `compute_schedule_meta`. | N/A | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Not nullable; derived from schedule home/away assignments. | — |
| `team_side` | Home/away indicator persisted alongside the payload row. | Derived during persistence in `build_and_store_features`. | N/A | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Not nullable; mirrors `home_away` values. | — |
| `week` | NFL week number for the contest (regular and postseason). | Schedule ingestion via `compute_schedule_meta`. | N/A | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Not nullable; coerced to integer. | — |

### Schedule Metadata

| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `kickoff_bucket` | Kickoff window classification (early/late/SNF/MNF). | Schedule ingestion via `compute_schedule_meta`. | Single-game | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Null when kickoff timestamp missing. | — |
| `rest_days` | Days of rest since the team's previous game in the season. | Schedule ingestion via `compute_schedule_meta`. | Single-game delta | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | NaN for first game or when previous kickoff missing. | — |
| `short_week` | Boolean flag for rest shorter than seven days. | Schedule ingestion via `compute_schedule_meta`. | Single-game | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | False when rest not computed; otherwise boolean. | — |

### Travel

| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `days_since_last` | Calendar days between consecutive games. | Schedule travel context via `compute_travel_features`. | Single-game | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | NaN for first game or missing kickoff data. | — |
| `neutral_site` | Boolean indicator for neutral-site games. | Schedule travel context via `compute_travel_features`. | Single-game | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | False when venue resolved; respects schedule neutral flags. | — |
| `travel_miles` | Great-circle miles traveled since previous game. | Schedule travel context via `compute_travel_features`. | Single-game | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | NaN when coordinates unavailable or neutral-site fallback required. | — |
| `venue_latitude` | Latitude for resolved venue coordinates in decimal degrees. | Schedule travel context via `compute_travel_features`. | Single-game | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | NaN when venue coordinates unavailable or neutral site. | — |
| `venue_longitude` | Longitude for resolved venue coordinates in decimal degrees. | Schedule travel context via `compute_travel_features`. | Single-game | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | NaN when venue coordinates unavailable or neutral site. | — |

### Play-by-Play Totals

| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `dropbacks` | Quarterback dropbacks including sacks and scrambles. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `early_down_epa` | EPA accumulated on early-down offensive plays. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `early_down_plays` | Offensive plays occurring on downs one and two. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `explosive_pass_plays` | Pass plays gaining at least 15 yards. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `explosive_rush_plays` | Rush plays gaining at least 10 yards. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `no_huddle_plays` | Offensive plays run without a huddle. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `pass_plays` | Pass attempts logged for the offense. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `penalties` | Offensive penalties assessed to the team. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `play_action_plays` | Pass plays flagged as play action. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `plays_offense` | Total offensive plays (pass + rush). | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `rush_plays` | Rush attempts logged for the offense. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `sacks` | Sacks taken by the offense. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `shotgun_plays` | Offensive plays executed from shotgun. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `st_epa` | EPA accumulated on special teams plays. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `st_plays` | Special teams plays (punts, kicks, field goals). | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `success_plays` | Sum of play success indicators used for success rate. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |
| `total_epa` | Offensive expected points added across all plays. | Play-by-play aggregation via `compute_team_week_features`. | Single-week total | Play-by-play rows filtered to events visible at `asof_ts`. | Zero when no qualifying plays; numeric aggregation. | — |

### Play-by-Play Rates

| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `early_down_epa_per_play` | EPA per early-down offensive play. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `epa_per_play` | EPA per offensive play. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `explosive_pass_rate` | Explosive pass plays divided by pass attempts. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `explosive_rush_rate` | Explosive rushes divided by rush attempts. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `no_huddle_rate` | Share of offensive plays run without a huddle. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `pass_rate` | Share of offensive plays that are passes. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `penalty_rate` | Offensive penalties divided by offensive plays. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `play_action_rate` | Share of pass plays using play action. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `rush_rate` | Share of offensive plays that are rushes. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `sack_rate` | Sacks divided by dropbacks. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `shotgun_rate` | Share of offensive plays from shotgun. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `st_epa_per_play` | Special teams EPA per special teams play. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |
| `success_rate` | Share of offensive plays marked successful. | Play-by-play aggregation via `compute_team_week_features`. | Single-week rate | Play-by-play rows filtered to events visible at `asof_ts`. | NaN when denominator is zero or missing. | — |

### Rolling Windows

| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `early_down_epa_per_play_season` | EPA per early-down offensive play. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `early_down_epa_per_play_w4` | EPA per early-down offensive play. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `early_down_epa_per_play_w8` | EPA per early-down offensive play. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `epa_per_play_season` | EPA per offensive play. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `epa_per_play_w4` | EPA per offensive play. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `epa_per_play_w8` | EPA per offensive play. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `explosive_pass_rate_season` | Explosive pass plays divided by pass attempts. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `explosive_pass_rate_w4` | Explosive pass plays divided by pass attempts. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `explosive_pass_rate_w8` | Explosive pass plays divided by pass attempts. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `explosive_rush_rate_season` | Explosive rushes divided by rush attempts. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `explosive_rush_rate_w4` | Explosive rushes divided by rush attempts. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `explosive_rush_rate_w8` | Explosive rushes divided by rush attempts. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `no_huddle_rate_season` | Share of offensive plays run without a huddle. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `no_huddle_rate_w4` | Share of offensive plays run without a huddle. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `no_huddle_rate_w8` | Share of offensive plays run without a huddle. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `pass_rate_season` | Share of offensive plays that are passes. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `pass_rate_w4` | Share of offensive plays that are passes. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `pass_rate_w8` | Share of offensive plays that are passes. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `penalty_rate_season` | Offensive penalties divided by offensive plays. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `penalty_rate_w4` | Offensive penalties divided by offensive plays. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `penalty_rate_w8` | Offensive penalties divided by offensive plays. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `play_action_rate_season` | Share of pass plays using play action. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `play_action_rate_w4` | Share of pass plays using play action. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `play_action_rate_w8` | Share of pass plays using play action. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `rush_rate_season` | Share of offensive plays that are rushes. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `rush_rate_w4` | Share of offensive plays that are rushes. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `rush_rate_w8` | Share of offensive plays that are rushes. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `sack_rate_season` | Sacks divided by dropbacks. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `sack_rate_w4` | Sacks divided by dropbacks. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `sack_rate_w8` | Sacks divided by dropbacks. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `shotgun_rate_season` | Share of offensive plays from shotgun. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `shotgun_rate_w4` | Share of offensive plays from shotgun. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `shotgun_rate_w8` | Share of offensive plays from shotgun. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `st_epa_per_play_season` | Special teams EPA per special teams play. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `st_epa_per_play_w4` | Special teams EPA per special teams play. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `st_epa_per_play_w8` | Special teams EPA per special teams play. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `success_rate_season` | Share of offensive plays marked successful. computed over season-to-date expanding window. | Play-by-play aggregation via `compute_team_week_features`. | Season-to-date expanding window | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `success_rate_w4` | Share of offensive plays marked successful. computed over trailing 4 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 4 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |
| `success_rate_w8` | Share of offensive plays marked successful. computed over trailing 8 games. | Play-by-play aggregation via `compute_team_week_features`. | Trailing 8 games | Play-by-play rows filtered to events visible at `asof_ts`. | NaN until sufficient history or denominator available. | — |

### Weather

| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `precip` | Probability of precipitation expressed as 0-1 fraction. | Stadium-enriched forecasts via `compute_weather_features`. | Single-game | Forecasts/backfill limited to updates <= `asof_ts`; indoor roofs yield defaults. | 0 for indoor venues; NaN when outdoor forecast unavailable. | — |
| `wx_temp` | Forecast or backfilled temperature in degrees Fahrenheit. | Stadium-enriched forecasts via `compute_weather_features`. | Single-game | Forecasts/backfill limited to updates <= `asof_ts`; indoor roofs yield defaults. | NaN when outdoor forecast unavailable; indoor venues left NaN. | — |
| `wx_wind` | Sustained wind speed in miles per hour. | Stadium-enriched forecasts via `compute_weather_features`. | Single-game | Forecasts/backfill limited to updates <= `asof_ts`; indoor roofs yield defaults. | 0 for indoor venues; NaN when outdoor forecast unavailable. | — |

### Rule & Phase Flags

| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `is_postseason` | Boolean flag identifying postseason contests. | Postseason flags appended by `append_playoff_flags`. | Single-game | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | False when postseason designation unavailable. | Uses schedule `game_type` when present; falls back to week threshold. |
| `kickoff_2024plus` | Rule-change flag for seasons impacted by the 2024 kickoff overhaul. | Rule flags appended by `append_rule_flags`. | Season scope | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Boolean; flips to True for seasons >= 2024. | Applies to all games assigned to seasons 2024 and beyond. |
| `ot_regular_2025plus` | Flag for updated regular-season overtime procedure starting 2025. | Rule flags appended by `append_rule_flags`. | Season scope | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Boolean; True for seasons >= 2025 during regular season weeks. | Weeks above regular-season threshold remain False pending playoff flags. |
| `season_phase` | Text label describing season phase (`regular` or `postseason`). | Postseason flags appended by `append_playoff_flags`. | Single-game | Schedule rows filtered to kickoff <= `asof_ts` via visibility rules. | Defaults to `regular` when postseason flag unavailable. | — |

### Scores & Labels

| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `label_team_win` | Training label: 1.0 win, 0.5 tie, 0.0 loss. | Scores merged from schedule within `build_and_store_features`. | Single-game | Available once both team and opponent scores are known. | NaN until official scores available. | — |
| `opponent_score` | Final score for the opponent. | Scores merged from schedule within `build_and_store_features`. | Single-game | Scores only appear after game completion (post-asof). | NaN until final scores recorded in schedule feed. | — |
| `team_score` | Final score for the documented team. | Scores merged from schedule within `build_and_store_features`. | Single-game | Scores only appear after game completion (post-asof). | NaN until final scores recorded in schedule feed. | — |
