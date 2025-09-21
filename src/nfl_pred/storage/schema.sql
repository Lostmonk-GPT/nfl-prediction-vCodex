-- DuckDB schema definitions for core prediction artifacts.
-- The canonical keys across tables include season, week, game_id, and snapshot timestamps.

CREATE TABLE IF NOT EXISTS features (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    game_id VARCHAR NOT NULL,
    team_side VARCHAR NOT NULL,
    asof_ts TIMESTAMP NOT NULL,
    snapshot_at TIMESTAMP NOT NULL,
    feature_set VARCHAR NOT NULL,
    payload_json VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (season, week, game_id, team_side, asof_ts)
);

CREATE INDEX IF NOT EXISTS idx_features_lookup
    ON features (season, week, game_id, snapshot_at);

CREATE TABLE IF NOT EXISTS predictions (
    game_id VARCHAR NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    asof_ts TIMESTAMP NOT NULL,
    p_home_win DOUBLE,
    p_away_win DOUBLE,
    pick VARCHAR,
    confidence VARCHAR,
    model_id VARCHAR NOT NULL,
    snapshot_at TIMESTAMP NOT NULL,
    PRIMARY KEY (game_id, season, week, asof_ts, model_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_lookup
    ON predictions (season, week, game_id, snapshot_at);

CREATE TABLE IF NOT EXISTS reports (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    asof_ts TIMESTAMP NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE,
    snapshot_at TIMESTAMP NOT NULL,
    PRIMARY KEY (season, week, asof_ts, metric)
);

CREATE INDEX IF NOT EXISTS idx_reports_lookup
    ON reports (season, week, snapshot_at);

CREATE TABLE IF NOT EXISTS runs_meta (
    run_id VARCHAR NOT NULL,
    model_id VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    params_json VARCHAR,
    metrics_json VARCHAR,
    PRIMARY KEY (run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_meta_model
    ON runs_meta (model_id, created_at);
