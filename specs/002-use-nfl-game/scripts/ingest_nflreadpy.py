#!/usr/bin/env python3
"""Ingest script for MVP: prefer `nflreadpy` when --live is set, otherwise use a cached snapshot.

Writes Parquet files under `data/raw/YYYYMMDD/` and a manifest containing file checksums and the
chosen source. Computes `data_manifest_id` as the sha256 of the concatenated file checksums.

This script intentionally avoids network access unless `--live` is provided. Use `--cache-snapshot YYYYMMDD`
to copy an existing snapshot from `data/raw_cache/YYYYMMDD/` into the output dir (CI should use cached snapshots).
"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def _file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_manifest(snapshot_dir: str, files: dict, source: str):
    # files: {relpath: checksum}
    checks_concat = ''.join(files[p] for p in sorted(files))
    data_manifest_id = hashlib.sha256(checks_concat.encode('utf-8')).hexdigest()
    m = {
        'source': source,
        'snapshot_path': str(snapshot_dir),
        'files': files,
        'data_manifest_id': data_manifest_id,
        'created_at': datetime.utcnow().isoformat() + 'Z'
    }
    man_dir = Path(snapshot_dir) / 'manifests'
    man_dir.mkdir(parents=True, exist_ok=True)
    man_path = man_dir / f"manifest_{data_manifest_id}.json"
    with open(man_path, 'w') as f:
        json.dump(m, f, indent=2)
    return str(man_path), data_manifest_id


def _copy_cached_snapshot(cache_date: str, out_dir: str):
    cache_root = Path('specs/002-use-nfl-game/data/raw_cache')
    src = cache_root / cache_date
    if not src.exists():
        raise FileNotFoundError(f'Cached snapshot not found: {src}')
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    files = {}
    for p in src.glob('*.parquet'):
        dst_p = dst / p.name
        shutil.copyfile(p, dst_p)
        files[p.name] = _file_checksum(str(dst_p))
    man_path, data_manifest_id = _write_manifest(dst, files, source='cached_snapshot')
    print('Copied cached snapshot and wrote manifest', man_path)
    return data_manifest_id


def _fetch_with_nflreadpy(start_season: int, end_season: int, out_dir: str):
    # Minimal adapter that returns team-week aggregates and games tables as pandas DataFrames
    try:
        import nflreadpy as nr
    except Exception as e:
        raise RuntimeError('nflreadpy is required for --live ingestion') from e

    # Example usage: nflreadpy.play_by_play or schedule endpoints
    # For MVP we'll attempt to load play-by-play (pbp) and schedules, and compute true team-week EPA and success_rate
    aggs = []
    games = []
    dl = nr.downloader.get_downloader()
    last_exc = None
    for season in range(start_season, end_season + 1):
        try:
            # Attempt to load play-by-play for season; pbp contains per-play epa and success indicators
            try:
                pbp = nr.load_pbp(season)
            except Exception:
                pbp = None

            # load schedules to get game-level rows
            sched = nr.load_schedules(season)

            try:
                sched_pd = sched.to_pandas()
            except Exception:
                sched_pd = pd.DataFrame(sched)

            # normalize schedule columns to ensure expected keys exist
            if 'season' not in sched_pd.columns:
                sched_pd['season'] = season
            if 'week' not in sched_pd.columns:
                # try common alternatives
                if 'game_week' in sched_pd.columns:
                    sched_pd['week'] = sched_pd['game_week']
                else:
                    # default to 0 if unknown
                    sched_pd['week'] = 0
            # normalize team column names
            if 'home_team' not in sched_pd.columns and 'home' in sched_pd.columns:
                sched_pd['home_team'] = sched_pd['home']
            if 'away_team' not in sched_pd.columns and 'away' in sched_pd.columns:
                sched_pd['away_team'] = sched_pd['away']

            # If pbp is available, compute per-team-week aggregates from play-by-play
            if pbp is not None:
                try:
                    pbp_pd = pbp.to_pandas()
                except Exception:
                    pbp_pd = pd.DataFrame(pbp)

                if 'epa' in pbp_pd.columns and 'game_id' in pbp_pd.columns:
                    # merge pbp with schedule to obtain season/week mapping per game
                    if 'game_id' in sched_pd.columns:
                        right = sched_pd[['game_id', 'season', 'week', 'home_team', 'away_team']].drop_duplicates()
                        merged = pbp_pd.merge(right, how='left', on='game_id')
                    else:
                        merged = pbp_pd

                    # determine team for each play: prefer posteam then try other fields
                    if 'posteam' in merged.columns:
                        merged['team'] = merged['posteam']
                    elif 'pos_team' in merged.columns:
                        merged['team'] = merged['pos_team']
                    else:
                        # as a best-effort fallback, skip plays without team
                        merged['team'] = merged.get('posteam')

                    # play_success may be named differently in different nflreadpy versions
                    success_col = None
                    for cand in ['play_success', 'success', 'is_success']:
                        if cand in merged.columns:
                            success_col = cand
                            break

                    if success_col is None:
                        # cannot compute success_rate without a success indicator; fill with NaNs
                        merged['play_success'] = pd.NA
                        success_col = 'play_success'

                    # aggregate to team-week level
                    tw = merged.groupby(['season', 'week', 'team']).agg({'epa': 'mean', success_col: 'mean'}).reset_index()
                    tw = tw.rename(columns={success_col: 'success_rate'})
                    aggs.append(tw)

            # build simple game list from schedule
            for _, r in sched_pd.iterrows():
                # series-like access with fallback to known defaults
                s = r
                season_val = int(s['season']) if 'season' in s and pd.notna(s['season']) else int(season)
                week_val = int(s['week']) if 'week' in s and pd.notna(s['week']) else 0
                home_t = s['home_team'] if 'home_team' in s and pd.notna(s['home_team']) else (s.get('home') if hasattr(s, 'get') else None)
                away_t = s['away_team'] if 'away_team' in s and pd.notna(s['away_team']) else (s.get('away') if hasattr(s, 'get') else None)
                g = {
                    'season': season_val,
                    'week': week_val,
                    'home_team': home_t,
                    'away_team': away_t,
                }
                if 'home_score' in r and 'away_score' in r:
                    try:
                        g['home_score'] = float(r['home_score'])
                        g['away_score'] = float(r['away_score'])
                        g['winner'] = 'home' if g['home_score'] > g['away_score'] else ('away' if g['away_score'] > g['home_score'] else None)
                    except Exception:
                        g['winner'] = None
                games.append(g)

            print(f'info: processed season {season}, schedule rows={len(sched_pd)}')
        except Exception as e:
            last_exc = e
            print(f'warning: failed to fetch/process season {season}: {e}')
            continue

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    if aggs:
        team_week_df = pd.concat(aggs, ignore_index=True)
        team_path = out / 'team_week_raw.parquet'
        team_week_df.to_parquet(team_path, index=False)
        files[team_path.name] = _file_checksum(str(team_path))
    if games:
        games_df = pd.DataFrame(games)
        games_path = out / 'games_raw.parquet'
        games_df.to_parquet(games_path, index=False)
        files[games_path.name] = _file_checksum(str(games_path))
    # attempt to compute source URL(s) via downloader helper
    urls = []
    try:
        # build likely URL for the schedules parquet
        schedules_path = f"schedules/{start_season}.parquet"
        try:
            # some versions expect (bucket, path, format_type)
            schedules_url = dl._build_url('nflverse-data', schedules_path, 'parquet')
        except TypeError:
            schedules_url = dl._build_url('nflverse-data', schedules_path)
        urls.append(schedules_url)
    except Exception:
        pass
    man_path, data_manifest_id = _write_manifest(out, files, source='nflreadpy')
    # append discovered urls to manifest file
    try:
        import json as _json
        m = _json.loads(open(man_path).read())
        m['source_urls'] = urls
        open(man_path, 'w').write(_json.dumps(m, indent=2))
    except Exception:
        pass
    print('Wrote live snapshot and manifest', man_path)
    return data_manifest_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--start-season', type=int, default=2018)
    parser.add_argument('--end-season', type=int, default=2023)
    parser.add_argument('--live', action='store_true', help='Allow live network fetch using nflreadpy')
    parser.add_argument('--cache-snapshot', help='Copy an existing cached snapshot (YYYYMMDD) from data/raw_cache')
    args = parser.parse_args()

    out_root = Path(args.out)
    # snapshot dir name
    snapshot_name = datetime.utcnow().strftime('%Y%m%d')
    snapshot_dir = out_root / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_snapshot:
        # copy from cached location
        data_manifest_id = _copy_cached_snapshot(args.cache_snapshot, str(snapshot_dir))
    elif args.live:
        data_manifest_id = _fetch_with_nflreadpy(args.start_season, args.end_season, str(snapshot_dir))
    else:
        # default CI/local behavior: prefer latest cached snapshot if present
        # fall back to error to avoid accidental network access
        cache_root = Path('specs/002-use-nfl-game/data/raw_cache')
        # pick newest date dir inside cache_root
        dates = [p for p in cache_root.iterdir() if p.is_dir()]
        if not dates:
            raise SystemExit('No cached snapshots available; rerun with --live to fetch data')
        latest = sorted(dates)[-1].name
        data_manifest_id = _copy_cached_snapshot(latest, str(snapshot_dir))

    print('data_manifest_id:', data_manifest_id)


if __name__ == '__main__':
    main()
