"""Feature specification generator for documentation output.

This module produces a Markdown specification describing every feature in the
team-week modeling table. The generator collates structured metadata describing
how each feature is computed, joins DuckDB to collect summary statistics, and
writes a deterministic Markdown report that can be versioned alongside the
codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import json
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

from nfl_pred.config import load_config
from nfl_pred.storage.duckdb_client import DuckDBClient


@dataclass(frozen=True, slots=True)
class FeatureSpecEntry:
    """Structured metadata describing a single feature column."""

    name: str
    category: str
    definition: str
    source: str
    window: str
    snapshot_timing: str
    null_policy: str
    rule_notes: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureStats:
    """Computed statistics for a feature derived from persisted data."""

    dtype: str
    non_null_ratio: float | None
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    team_avg_min: float | None
    team_avg_median: float | None
    team_avg_max: float | None
    is_numeric: bool


_DEFAULT_OUTPUT = Path("docs/feature_spec.md")
_SCHEDULE_SNAPSHOT = "Schedule rows filtered to kickoff <= `asof_ts` via visibility rules."
_PLAY_BY_PLAY_SNAPSHOT = "Play-by-play rows filtered to events visible at `asof_ts`."
_WEATHER_SNAPSHOT = "Forecasts/backfill limited to updates <= `asof_ts`; indoor roofs yield defaults."

_CATEGORY_ORDER: Sequence[str] = (
    "Keys & Context",
    "Schedule Metadata",
    "Travel",
    "Play-by-Play Totals",
    "Play-by-Play Rates",
    "Rolling Windows",
    "Weather",
    "Rule & Phase Flags",
    "Scores & Labels",
)


def generate_feature_spec(
    *,
    duckdb_path: str | Path | None = None,
    feature_set: str = "mvp_v1",
    output_path: str | Path | None = None,
) -> Path:
    """Generate the feature specification Markdown document.

    Args:
        duckdb_path: Optional path to the DuckDB database. Defaults to the path
            configured via :func:`nfl_pred.config.load_config` when omitted.
        feature_set: Feature set identifier to extract from DuckDB.
        output_path: Optional override for the Markdown output path. Defaults to
            ``docs/feature_spec.md`` within the repository.

    Returns:
        Path to the rendered Markdown file.
    """

    if duckdb_path is None:
        config = load_config()
        duckdb_path = config.paths.duckdb_path

    feature_frame = _load_feature_payloads(duckdb_path, feature_set=feature_set)

    entries = list(_build_feature_entries())
    stats = _compute_feature_statistics(feature_frame, feature_names=[entry.name for entry in entries])

    rendered = _render_markdown(
        entries,
        stats,
        duckdb_path=Path(duckdb_path),
        feature_set=feature_set,
        row_count=len(feature_frame) if feature_frame is not None else 0,
    )

    target_path = Path(output_path) if output_path is not None else _DEFAULT_OUTPUT
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(rendered, encoding="utf-8")

    return target_path


def _load_feature_payloads(duckdb_path: str | Path, *, feature_set: str) -> pd.DataFrame | None:
    """Return the most recent payload rows for ``feature_set`` from DuckDB.

    The helper mirrors the training pipeline loader by expanding ``payload_json``
    into native columns and ensuring ``home_away`` is present for downstream
    grouping.
    """

    query = """
        SELECT
            season,
            week,
            game_id,
            team_side,
            asof_ts,
            snapshot_at,
            payload_json
        FROM features
        WHERE feature_set = ?
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY season, week, game_id, team_side
            ORDER BY snapshot_at DESC
        ) = 1
        ORDER BY season, week, game_id, team_side
    """

    db_path = str(duckdb_path)

    try:
        with DuckDBClient(db_path, read_only=True) as client:
            try:
                client.apply_schema()
            except duckdb.Error as schema_error:
                if "read-only" not in str(schema_error).lower():
                    raise
            frame = client.read_sql(query, (feature_set,))
    except FileNotFoundError:
        return None
    except duckdb.Error as error:
        message = str(error).lower()
        if any(token in message for token in ("cannot open file", "no such file", "does not exist", "cannot open database")):
            return None
        raise RuntimeError(f"Failed to read features from DuckDB at '{duckdb_path}': {error}") from error
    except Exception as error:  # pragma: no cover - defensive logging path
        raise RuntimeError(f"Failed to read features from DuckDB at '{duckdb_path}': {error}") from error

    if frame.empty:
        return None

    payload_df = pd.DataFrame.from_records(frame["payload_json"].map(json.loads))
    payload_df = payload_df.fillna(value=np.nan)

    combined = pd.concat(
        [frame.drop(columns=["payload_json"]).reset_index(drop=True), payload_df],
        axis=1,
    )

    combined["team_side"] = combined["team_side"].astype(str)
    combined["season"] = combined["season"].astype(int)
    combined["week"] = combined["week"].astype(int)

    if "home_away" not in combined.columns:
        combined["home_away"] = combined["team_side"].astype(str)

    return combined


def _compute_feature_statistics(
    frame: pd.DataFrame | None,
    *,
    feature_names: Iterable[str],
    team_column: str = "team",
) -> Mapping[str, FeatureStats]:
    """Compute summary statistics for the provided feature frame."""

    if frame is None or frame.empty:
        return {}

    stats: dict[str, FeatureStats] = {}
    total_rows = len(frame)
    numeric_team_cache: dict[str, pd.Series] = {}

    team_available = team_column in frame.columns
    if team_available:
        team_values = frame[team_column].astype(str)
    else:
        team_values = None  # type: ignore[assignment]

    for name in feature_names:
        if name not in frame.columns:
            continue

        series = frame[name]
        dtype = str(series.dtype)
        non_null = series.notna().sum()
        ratio = non_null / total_rows if total_rows else None

        is_numeric = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)

        mean = std = minimum = maximum = None
        team_min = team_median = team_max = None

        if is_numeric:
            numeric_series = pd.to_numeric(series, errors="coerce")
            if numeric_series.notna().any():
                mean = float(numeric_series.mean())
                std = float(numeric_series.std(ddof=0))
                minimum = float(numeric_series.min())
                maximum = float(numeric_series.max())

                if team_available:
                    cache_key = name
                    if cache_key not in numeric_team_cache:
                        grouped = pd.concat([team_values, numeric_series], axis=1)
                        grouped.columns = [team_column, name]
                        team_means = (
                            grouped.dropna(subset=[name])
                            .groupby(team_column)[name]
                            .mean()
                        )
                        numeric_team_cache[cache_key] = team_means
                    team_means = numeric_team_cache[cache_key]
                    if not team_means.empty:
                        team_min = float(team_means.min())
                        team_max = float(team_means.max())
                        team_median = float(team_means.median())

        stats[name] = FeatureStats(
            dtype=dtype,
            non_null_ratio=ratio,
            mean=mean,
            std=std,
            minimum=minimum,
            maximum=maximum,
            team_avg_min=team_min,
            team_avg_median=team_median,
            team_avg_max=team_max,
            is_numeric=is_numeric,
        )

    return stats


def _render_markdown(
    entries: Sequence[FeatureSpecEntry],
    stats: Mapping[str, FeatureStats],
    *,
    duckdb_path: Path,
    feature_set: str,
    row_count: int,
) -> str:
    """Return Markdown string for the provided entries and statistics."""

    generated_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Feature Specification")
    lines.append("")
    lines.append(
        f"Generated on {generated_ts} from DuckDB database `{duckdb_path}` using feature set `{feature_set}`."
    )
    if row_count:
        lines.append(f"Summary statistics computed from {row_count:,} team-week rows.")
    else:
        lines.append("No persisted feature rows were available; documentation defaults to static metadata.")
    lines.append("")

    by_category: dict[str, list[FeatureSpecEntry]] = {category: [] for category in _CATEGORY_ORDER}
    for entry in entries:
        by_category.setdefault(entry.category, []).append(entry)

    lines.append("## Feature Documentation")
    lines.append("")

    for category in _CATEGORY_ORDER:
        category_entries = by_category.get(category, [])
        if not category_entries:
            continue
        lines.append(f"### {category}")
        lines.append("")
        lines.append(
            "| Feature | Definition | Source | Window | Snapshot Timing | Null Policy | Rule Notes |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for entry in sorted(category_entries, key=lambda item: item.name):
            notes = entry.rule_notes or "—"
            lines.append(
                "| `{name}` | {definition} | {source} | {window} | {snapshot} | {null_policy} | {notes} |".format(
                    name=entry.name,
                    definition=entry.definition,
                    source=entry.source,
                    window=entry.window,
                    snapshot=entry.snapshot_timing,
                    null_policy=entry.null_policy,
                    notes=notes,
                )
            )
        lines.append("")

    numeric_entries = [entry for entry in entries if stats.get(entry.name, None) and stats[entry.name].is_numeric]
    if numeric_entries:
        lines.append("## Numeric Feature Summary")
        lines.append("")
        lines.append(
            "| Feature | Type | Non-Null % | Mean | Std | Min | Max | Team Avg Min | Team Avg Median | Team Avg Max |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for entry in sorted(numeric_entries, key=lambda item: item.name):
            stat = stats[entry.name]
            lines.append(
                "| `{name}` | {dtype} | {non_null} | {mean} | {std} | {min_val} | {max_val} | {team_min} | {team_median} | {team_max} |".format(
                    name=entry.name,
                    dtype=stat.dtype,
                    non_null=_format_percent(stat.non_null_ratio),
                    mean=_format_number(stat.mean),
                    std=_format_number(stat.std),
                    min_val=_format_number(stat.minimum),
                    max_val=_format_number(stat.maximum),
                    team_min=_format_number(stat.team_avg_min),
                    team_median=_format_number(stat.team_avg_median),
                    team_max=_format_number(stat.team_avg_max),
                )
            )
        lines.append("")

    if stats:
        missing_stats = [entry.name for entry in entries if entry.name not in stats]
        if missing_stats:
            missing_list = ", ".join(f"`{name}`" for name in sorted(missing_stats))
            lines.append(
                f"_The following features are documented but not present in the current feature snapshot: {missing_list}._"
            )
            lines.append("")

    return "\n".join(lines)


def _format_number(value: float | None) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "—"
    if abs(value) >= 1000 or abs(value) < 0.01 and value != 0:
        return f"{value:.3e}"
    return f"{value:.3f}"


def _format_percent(ratio: float | None) -> str:
    if ratio is None:
        return "—"
    return f"{ratio * 100:.1f}%"


def _build_feature_entries() -> Iterable[FeatureSpecEntry]:
    """Yield :class:`FeatureSpecEntry` instances covering the MVP feature set."""

    schedule_source = "Schedule ingestion via `compute_schedule_meta`."
    travel_source = "Schedule travel context via `compute_travel_features`."
    pbp_source = "Play-by-play aggregation via `compute_team_week_features`."
    weather_source = "Stadium-enriched forecasts via `compute_weather_features`."
    rule_source = "Rule flags appended by `append_rule_flags`."
    playoff_source = "Postseason flags appended by `append_playoff_flags`."
    score_source = "Scores merged from schedule within `build_and_store_features`."

    # Keys & context
    yield FeatureSpecEntry(
        name="season",
        category="Keys & Context",
        definition="Season identifier for the game row.",
        source=schedule_source,
        window="N/A",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Not nullable; coerced to integer.",
    )
    yield FeatureSpecEntry(
        name="week",
        category="Keys & Context",
        definition="NFL week number for the contest (regular and postseason).",
        source=schedule_source,
        window="N/A",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Not nullable; coerced to integer.",
    )
    yield FeatureSpecEntry(
        name="game_id",
        category="Keys & Context",
        definition="nflverse game identifier for the matchup.",
        source=schedule_source,
        window="N/A",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Not nullable; string identifier.",
    )
    yield FeatureSpecEntry(
        name="team",
        category="Keys & Context",
        definition="Team associated with the row using nflfastR abbreviations.",
        source=schedule_source,
        window="N/A",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Not nullable; derived from schedule home/away assignments.",
    )
    yield FeatureSpecEntry(
        name="opponent",
        category="Keys & Context",
        definition="Opponent abbreviation for the matchup.",
        source=schedule_source,
        window="N/A",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Not nullable; mirrors schedule pairings.",
    )
    yield FeatureSpecEntry(
        name="home_away",
        category="Keys & Context",
        definition="Game site classification for the team (home/away/neutral).",
        source=schedule_source,
        window="N/A",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Not nullable; neutral overrides home/away flags when detected.",
    )
    yield FeatureSpecEntry(
        name="team_side",
        category="Keys & Context",
        definition="Home/away indicator persisted alongside the payload row.",
        source="Derived during persistence in `build_and_store_features`.",
        window="N/A",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Not nullable; mirrors `home_away` values.",
    )
    yield FeatureSpecEntry(
        name="asof_ts",
        category="Keys & Context",
        definition="Visibility cutoff timestamp applied to source data.",
        source="Snapshot metadata stored with each payload row.",
        window="N/A",
        snapshot_timing="Defines the visibility boundary for included inputs.",
        null_policy="Not nullable; timezone-aware UTC timestamp.",
    )
    yield FeatureSpecEntry(
        name="snapshot_at",
        category="Keys & Context",
        definition="Timestamp when the feature snapshot executed.",
        source="Snapshot metadata stored with each payload row.",
        window="N/A",
        snapshot_timing="Recorded post-run to support reproducibility.",
        null_policy="Not nullable; timezone-aware UTC timestamp.",
    )
    yield FeatureSpecEntry(
        name="start_time",
        category="Keys & Context",
        definition="UTC kickoff timestamp for the game.",
        source=schedule_source,
        window="N/A",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="May be null when kickoff timestamp unavailable in schedule.",
    )

    # Schedule metadata
    yield FeatureSpecEntry(
        name="rest_days",
        category="Schedule Metadata",
        definition="Days of rest since the team's previous game in the season.",
        source=schedule_source,
        window="Single-game delta",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="NaN for first game or when previous kickoff missing.",
    )
    yield FeatureSpecEntry(
        name="short_week",
        category="Schedule Metadata",
        definition="Boolean flag for rest shorter than seven days.",
        source=schedule_source,
        window="Single-game",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="False when rest not computed; otherwise boolean.",
    )
    yield FeatureSpecEntry(
        name="kickoff_bucket",
        category="Schedule Metadata",
        definition="Kickoff window classification (early/late/SNF/MNF).",
        source=schedule_source,
        window="Single-game",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Null when kickoff timestamp missing.",
    )

    # Travel
    yield FeatureSpecEntry(
        name="neutral_site",
        category="Travel",
        definition="Boolean indicator for neutral-site games.",
        source=travel_source,
        window="Single-game",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="False when venue resolved; respects schedule neutral flags.",
    )
    yield FeatureSpecEntry(
        name="travel_miles",
        category="Travel",
        definition="Great-circle miles traveled since previous game.",
        source=travel_source,
        window="Single-game",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="NaN when coordinates unavailable or neutral-site fallback required.",
    )
    yield FeatureSpecEntry(
        name="days_since_last",
        category="Travel",
        definition="Calendar days between consecutive games.",
        source=travel_source,
        window="Single-game",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="NaN for first game or missing kickoff data.",
    )
    yield FeatureSpecEntry(
        name="venue_latitude",
        category="Travel",
        definition="Latitude for resolved venue coordinates in decimal degrees.",
        source=travel_source,
        window="Single-game",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="NaN when venue coordinates unavailable or neutral site.",
    )
    yield FeatureSpecEntry(
        name="venue_longitude",
        category="Travel",
        definition="Longitude for resolved venue coordinates in decimal degrees.",
        source=travel_source,
        window="Single-game",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="NaN when venue coordinates unavailable or neutral site.",
    )

    # Play-by-play totals
    totals = [
        ("plays_offense", "Total offensive plays (pass + rush)."),
        ("pass_plays", "Pass attempts logged for the offense."),
        ("rush_plays", "Rush attempts logged for the offense."),
        ("dropbacks", "Quarterback dropbacks including sacks and scrambles."),
        ("sacks", "Sacks taken by the offense."),
        ("success_plays", "Sum of play success indicators used for success rate."),
        ("early_down_plays", "Offensive plays occurring on downs one and two."),
        ("play_action_plays", "Pass plays flagged as play action."),
        ("shotgun_plays", "Offensive plays executed from shotgun."),
        ("no_huddle_plays", "Offensive plays run without a huddle."),
        ("explosive_pass_plays", "Pass plays gaining at least 15 yards."),
        ("explosive_rush_plays", "Rush plays gaining at least 10 yards."),
        ("penalties", "Offensive penalties assessed to the team."),
        ("st_plays", "Special teams plays (punts, kicks, field goals)."),
        ("total_epa", "Offensive expected points added across all plays."),
        ("early_down_epa", "EPA accumulated on early-down offensive plays."),
        ("st_epa", "EPA accumulated on special teams plays."),
    ]
    for name, definition in totals:
        yield FeatureSpecEntry(
            name=name,
            category="Play-by-Play Totals",
            definition=definition,
            source=pbp_source,
            window="Single-week total",
            snapshot_timing=_PLAY_BY_PLAY_SNAPSHOT,
            null_policy="Zero when no qualifying plays; numeric aggregation.",
        )

    # Play-by-play rates
    rates = [
        ("epa_per_play", "EPA per offensive play."),
        ("early_down_epa_per_play", "EPA per early-down offensive play."),
        ("success_rate", "Share of offensive plays marked successful."),
        ("pass_rate", "Share of offensive plays that are passes."),
        ("rush_rate", "Share of offensive plays that are rushes."),
        ("play_action_rate", "Share of pass plays using play action."),
        ("shotgun_rate", "Share of offensive plays from shotgun."),
        ("no_huddle_rate", "Share of offensive plays run without a huddle."),
        ("sack_rate", "Sacks divided by dropbacks."),
        ("explosive_pass_rate", "Explosive pass plays divided by pass attempts."),
        ("explosive_rush_rate", "Explosive rushes divided by rush attempts."),
        ("penalty_rate", "Offensive penalties divided by offensive plays."),
        ("st_epa_per_play", "Special teams EPA per special teams play."),
    ]
    for name, definition in rates:
        yield FeatureSpecEntry(
            name=name,
            category="Play-by-Play Rates",
            definition=definition,
            source=pbp_source,
            window="Single-week rate",
            snapshot_timing=_PLAY_BY_PLAY_SNAPSHOT,
            null_policy="NaN when denominator is zero or missing.",
        )

    # Rolling windows
    rolling_windows = {
        "w4": "Trailing 4 games",
        "w8": "Trailing 8 games",
        "season": "Season-to-date expanding window",
    }
    for base_name, definition in rates:
        for suffix, window_label in rolling_windows.items():
            yield FeatureSpecEntry(
                name=f"{base_name}_{suffix}",
                category="Rolling Windows",
                definition=f"{definition} computed over {window_label.lower()}.",
                source=pbp_source,
                window=window_label,
                snapshot_timing=_PLAY_BY_PLAY_SNAPSHOT,
                null_policy="NaN until sufficient history or denominator available.",
            )

    # Weather
    yield FeatureSpecEntry(
        name="wx_temp",
        category="Weather",
        definition="Forecast or backfilled temperature in degrees Fahrenheit.",
        source=weather_source,
        window="Single-game",
        snapshot_timing=_WEATHER_SNAPSHOT,
        null_policy="NaN when outdoor forecast unavailable; indoor venues left NaN.",
    )
    yield FeatureSpecEntry(
        name="wx_wind",
        category="Weather",
        definition="Sustained wind speed in miles per hour.",
        source=weather_source,
        window="Single-game",
        snapshot_timing=_WEATHER_SNAPSHOT,
        null_policy="0 for indoor venues; NaN when outdoor forecast unavailable.",
    )
    yield FeatureSpecEntry(
        name="precip",
        category="Weather",
        definition="Probability of precipitation expressed as 0-1 fraction.",
        source=weather_source,
        window="Single-game",
        snapshot_timing=_WEATHER_SNAPSHOT,
        null_policy="0 for indoor venues; NaN when outdoor forecast unavailable.",
    )

    # Rule & phase flags
    yield FeatureSpecEntry(
        name="is_postseason",
        category="Rule & Phase Flags",
        definition="Boolean flag identifying postseason contests.",
        source=playoff_source,
        window="Single-game",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="False when postseason designation unavailable.",
        rule_notes="Uses schedule `game_type` when present; falls back to week threshold.",
    )
    yield FeatureSpecEntry(
        name="season_phase",
        category="Rule & Phase Flags",
        definition="Text label describing season phase (`regular` or `postseason`).",
        source=playoff_source,
        window="Single-game",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Defaults to `regular` when postseason flag unavailable.",
    )
    yield FeatureSpecEntry(
        name="kickoff_2024plus",
        category="Rule & Phase Flags",
        definition="Rule-change flag for seasons impacted by the 2024 kickoff overhaul.",
        source=rule_source,
        window="Season scope",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Boolean; flips to True for seasons >= 2024.",
        rule_notes="Applies to all games assigned to seasons 2024 and beyond.",
    )
    yield FeatureSpecEntry(
        name="ot_regular_2025plus",
        category="Rule & Phase Flags",
        definition="Flag for updated regular-season overtime procedure starting 2025.",
        source=rule_source,
        window="Season scope",
        snapshot_timing=_SCHEDULE_SNAPSHOT,
        null_policy="Boolean; True for seasons >= 2025 during regular season weeks.",
        rule_notes="Weeks above regular-season threshold remain False pending playoff flags.",
    )

    # Scores & labels
    yield FeatureSpecEntry(
        name="team_score",
        category="Scores & Labels",
        definition="Final score for the documented team.",
        source=score_source,
        window="Single-game",
        snapshot_timing="Scores only appear after game completion (post-asof).",
        null_policy="NaN until final scores recorded in schedule feed.",
    )
    yield FeatureSpecEntry(
        name="opponent_score",
        category="Scores & Labels",
        definition="Final score for the opponent.",
        source=score_source,
        window="Single-game",
        snapshot_timing="Scores only appear after game completion (post-asof).",
        null_policy="NaN until final scores recorded in schedule feed.",
    )
    yield FeatureSpecEntry(
        name="label_team_win",
        category="Scores & Labels",
        definition="Training label: 1.0 win, 0.5 tie, 0.0 loss.",
        source=score_source,
        window="Single-game",
        snapshot_timing="Available once both team and opponent scores are known.",
        null_policy="NaN until official scores available.",
    )


__all__ = ["FeatureSpecEntry", "FeatureStats", "generate_feature_spec"]
