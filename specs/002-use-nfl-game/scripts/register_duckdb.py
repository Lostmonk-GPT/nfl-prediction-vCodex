#!/usr/bin/env python3
"""Create a DuckDB file and register partitioned Parquet feature files for analytics.

This script expects features partitioned by season under a features directory, e.g. features/season=2018/features.parquet
It will create `db/features.duckdb` and create a view `features` reading from the parquet files.
"""
import argparse
from pathlib import Path

import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-dir', required=True)
    parser.add_argument('--out-db', default='specs/002-use-nfl-game/db/features.duckdb')
    args = parser.parse_args()
    fdir = Path(args.features_dir)
    if not fdir.exists():
        raise SystemExit('Features dir not found: ' + str(fdir))
    # find parquet files recursively
    parquet_files = [str(p) for p in fdir.rglob('*.parquet')]
    if not parquet_files:
        raise SystemExit('No parquet files found under features dir')
    out_db_path = Path(args.out_db)
    out_db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out_db_path))
    # create a view that unions all parquet files
    # duckdb supports reading multiple files via glob
    glob_path = str(fdir / '**' / '*.parquet')
    con.execute("CREATE OR REPLACE VIEW features AS SELECT * FROM read_parquet('%s')" % glob_path)
    # ensure weekly_metrics table exists and create a trend view with 4-week rolling means
    con.execute('''
        CREATE TABLE IF NOT EXISTS weekly_metrics (
            season INTEGER,
            week INTEGER,
            model_hash VARCHAR,
            dataset VARCHAR,
            accuracy DOUBLE,
            brier DOUBLE,
            logloss DOUBLE,
            ece DOUBLE,
            created_at TIMESTAMP
        )
    ''')
    # create a trends view which includes 4-week rolling means
    con.execute('''
        CREATE OR REPLACE VIEW weekly_metric_trends AS
        SELECT
            season,
            week,
            accuracy,
            brier,
            logloss,
            ece,
            avg(accuracy) OVER (ORDER BY season, week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS accuracy_4wk_mean,
            avg(brier) OVER (ORDER BY season, week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS brier_4wk_mean,
            avg(logloss) OVER (ORDER BY season, week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS logloss_4wk_mean,
            avg(ece) OVER (ORDER BY season, week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS ece_4wk_mean
        FROM weekly_metrics
        ORDER BY season, week
    ''')
    con.close()
    print('Created DuckDB at', args.out_db)


if __name__ == '__main__':
    main()
