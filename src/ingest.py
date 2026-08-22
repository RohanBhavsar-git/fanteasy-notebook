"""
FanTeasy Stats — Data ingestion (Phase 1)

Reusable functions for pulling data from nflverse (via nflreadpy) and
Sleeper's public API. Every fetch is cached to data/raw/ so subsequent
calls in the same session (or across sessions) don't re-hit the network.

Usage from a notebook:
    from src.ingest import (
        get_pbp, get_weekly_stats, get_snap_counts,
        get_ngs_data, get_schedule, get_seasonal_rosters,
        get_id_crosswalk, get_sleeper_league, get_sleeper_players,
        get_sleeper_projections,
    )
    pbp = get_pbp([2024, 2025])

Note on the nflverse client:
    This module uses `nflreadpy`, the maintained successor to
    `nfl_data_py` (which nflverse deprecated in 2025). nflreadpy returns
    Polars DataFrames; every function here converts to pandas at the
    boundary so the rest of the pipeline (Phases 2-7) stays pandas-native.
    If you later want Polars end-to-end, drop the _to_pandas() calls.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import polars as pl
import requests

# nflreadpy is the maintained Python client for nflverse data. See:
# https://nflreadpy.nflverse.com/
import nflreadpy as nfl

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths — resolve relative to this file so it works whether called from a
# notebook (which has a different cwd) or from a script.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_OUTPUT = PROJECT_ROOT / "data" / "output"

# Ensure directories exist even if .gitkeep files got deleted
for d in (DATA_RAW, DATA_PROCESSED, DATA_OUTPUT):
    d.mkdir(parents=True, exist_ok=True)

# Sleeper's public API (no auth required, generous rate limits — 1000/min).
# This is the same host + path the dashboard uses in index.html; keeping them
# identical matters because the notebook's projection baseline should be the
# exact numbers the dashboard displays.
SLEEPER_API = "https://api.sleeper.app/v1"
# Newer Sleeper host, used as a fallback for projections only.
SLEEPER_API_ALT = "https://api.sleeper.com"

# The league this project is built for. Change here if you fork the project.
# NOTE: Sleeper mints a NEW league_id each season for dynasty leagues (the old
# one becomes `previous_league_id`). If get_sleeper_league() reports a season
# older than the one you're modeling, update this constant.
DEFAULT_LEAGUE_ID = "1389706592789733376"


# ==========================================================================
# INTERNAL HELPERS
# ==========================================================================
def _to_pandas(df: pl.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Convert an nflreadpy (Polars) frame to pandas.

    If `columns` is given, the subset is selected in Polars *before* the
    conversion — this matters for play-by-play, where the full 380-column
    frame can cost a gigabyte or more in pandas.
    """
    if columns:
        available = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning(f"Columns not present in source data, skipped: {missing}")
        df = df.select(available)
    return df.to_pandas()


def _normalize_id_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Coerce an ID column to clean strings.

    Player-ID columns arrive as floats whenever the source has nulls, which
    turns Sleeper's "4984" into 4984.0 and silently breaks every downstream
    join (Sleeper's own player_id values are strings). This strips the
    trailing ".0" and leaves genuine nulls as pd.NA.
    """
    if col not in df.columns:
        return df
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    return df


# ==========================================================================
# CACHING HELPERS
# ==========================================================================
def _cache_path(name: str, suffix: str = "parquet") -> Path:
    """Build a cache filepath under data/raw/."""
    return DATA_RAW / f"{name}.{suffix}"


def _read_cache_parquet(name: str) -> pd.DataFrame | None:
    """Return cached DataFrame if present, else None."""
    path = _cache_path(name, "parquet")
    if path.exists():
        logger.info(f"[cache hit]  {path.name}")
        return pd.read_parquet(path)
    return None


def _write_cache_parquet(df: pd.DataFrame, name: str) -> None:
    """Persist a DataFrame to parquet under data/raw/."""
    path = _cache_path(name, "parquet")
    df.to_parquet(path, index=False)
    logger.info(f"[cache write] {path.name} ({len(df):,} rows)")


def _read_cache_json(name: str) -> dict | list | None:
    path = _cache_path(name, "json")
    if path.exists():
        logger.info(f"[cache hit]  {path.name}")
        return json.loads(path.read_text())
    return None


def _write_cache_json(obj: dict | list, name: str) -> None:
    path = _cache_path(name, "json")
    path.write_text(json.dumps(obj))
    logger.info(f"[cache write] {path.name}")


# ==========================================================================
# NFLVERSE FETCHERS
# ==========================================================================
def get_pbp(
    seasons: Iterable[int],
    refresh: bool = False,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Play-by-play data — the master truth source. ~50k rows per season.

    Contains every play with fields like: pass_attempt, complete_pass,
    passing_yards, air_yards, yards_after_catch, rush_attempt,
    rushing_yards, rush_touchdown, sack, interception, yardline_100,
    pass_location, run_location, run_gap, receiver_player_id, etc.

    Args:
        seasons: iterable of years, e.g. [2024, 2025]
        refresh: force a re-fetch instead of reading the parquet cache
        columns: optional column subset, selected before the Polars →
            pandas conversion. Two full seasons of pbp is ~380 columns and
            can exceed 1 GB in pandas; pass a subset if memory is tight.

    Returns:
        DataFrame with one row per play. gsis_id-based player IDs.
    """
    seasons = list(seasons)
    cache_name = f"pbp_{'_'.join(str(s) for s in seasons)}"
    if not refresh:
        cached = _read_cache_parquet(cache_name)
        if cached is not None:
            return cached
    logger.info(f"Fetching play-by-play for {seasons}...")
    df = _to_pandas(nfl.load_pbp(seasons), columns)
    _write_cache_parquet(df, cache_name)
    return df


def get_weekly_stats(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """
    Weekly aggregated stats — easier than aggregating pbp yourself.

    One row per (player, week, season). Includes totals like:
        completions, attempts, passing_yards, passing_tds,
        interceptions, sacks, carries, rushing_yards, rushing_tds,
        receptions, targets, receiving_yards, receiving_tds,
        fantasy_points, fantasy_points_ppr.

    Returns:
        DataFrame keyed by (player_id, season, week).
    """
    seasons = list(seasons)
    cache_name = f"weekly_{'_'.join(str(s) for s in seasons)}"
    if not refresh:
        cached = _read_cache_parquet(cache_name)
        if cached is not None:
            return cached
    logger.info(f"Fetching weekly stats for {seasons}...")
    df = _to_pandas(nfl.load_player_stats(seasons, summary_level="week"))
    _write_cache_parquet(df, cache_name)
    return df


def get_snap_counts(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """
    Snap counts — critical for role classification (Phase 3).

    Fields include: offense_snaps, offense_pct, defense_snaps,
    defense_pct, st_snaps, st_pct. One row per (player, game).
    Sourced from Pro Football Reference; available from 2012 on.

    Returns:
        DataFrame with (pfr_player_id, game_id, snap counts). Note the
        ID system here (pfr_player_id) differs from pbp's gsis_id — use
        the crosswalk to join.
    """
    seasons = list(seasons)
    cache_name = f"snaps_{'_'.join(str(s) for s in seasons)}"
    if not refresh:
        cached = _read_cache_parquet(cache_name)
        if cached is not None:
            return cached
    logger.info(f"Fetching snap counts for {seasons}...")
    df = _to_pandas(nfl.load_snap_counts(seasons))
    _write_cache_parquet(df, cache_name)
    return df


def get_ngs_data(
    stat_type: str, seasons: Iterable[int], refresh: bool = False
) -> pd.DataFrame:
    """
    Next Gen Stats — aDOT, separation, time to throw.

    Args:
        stat_type: 'passing', 'receiving', or 'rushing'
        seasons: iterable of years

    Notes:
        NGS is only available from 2016+ and has some latency (a few days
        after the game). Some fields are only populated for a subset of
        players (e.g. separation requires tracking data).

        The argument order here (stat_type first) is kept from the original
        nfl_data_py-based version so notebooks don't need editing — but note
        nflreadpy's own load_nextgen_stats() takes (seasons, stat_type).
    """
    if stat_type not in {"passing", "receiving", "rushing"}:
        raise ValueError(f"stat_type must be passing|receiving|rushing, got {stat_type}")
    seasons = list(seasons)
    cache_name = f"ngs_{stat_type}_{'_'.join(str(s) for s in seasons)}"
    if not refresh:
        cached = _read_cache_parquet(cache_name)
        if cached is not None:
            return cached
    logger.info(f"Fetching NGS {stat_type} for {seasons}...")
    df = _to_pandas(nfl.load_nextgen_stats(seasons=seasons, stat_type=stat_type))
    _write_cache_parquet(df, cache_name)
    return df


def get_schedule(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """
    Game schedule for matchup + weather features.

    Includes game_id, week, season, home_team, away_team, gameday,
    weekday, roof (dome/outdoors/closed/open), surface, temp, wind, stadium.
    """
    seasons = list(seasons)
    cache_name = f"schedule_{'_'.join(str(s) for s in seasons)}"
    if not refresh:
        cached = _read_cache_parquet(cache_name)
        if cached is not None:
            return cached
    logger.info(f"Fetching schedule for {seasons}...")
    df = _to_pandas(nfl.load_schedules(seasons))
    _write_cache_parquet(df, cache_name)
    return df


def get_seasonal_rosters(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """
    Season-level roster — one row per (player, season, team) assignment.

    Useful for team-affiliation lookups and confirming which players
    were active for a given season.
    """
    seasons = list(seasons)
    cache_name = f"rosters_{'_'.join(str(s) for s in seasons)}"
    if not refresh:
        cached = _read_cache_parquet(cache_name)
        if cached is not None:
            return cached
    logger.info(f"Fetching rosters for {seasons}...")
    df = _to_pandas(nfl.load_rosters(seasons))
    _write_cache_parquet(df, cache_name)
    return df


def get_id_crosswalk(refresh: bool = False) -> pd.DataFrame:
    """
    Multi-ID crosswalk — the joiner between nflverse and Sleeper.

    nflverse uses gsis_id everywhere. Sleeper uses its own sleeper_id.
    nflreadpy's load_ff_playerids() (the DynastyProcess player-ID database)
    gives us both, plus many others (ESPN, Yahoo, PFR, PFF, MFL, RotoWire).

    ID columns are normalized to strings here — see _normalize_id_column for
    why that matters. Every join in Phases 2-7 depends on this.

    Returns:
        DataFrame with columns including gsis_id, sleeper_id, name,
        position, team, and other ID systems.
    """
    if not refresh:
        cached = _read_cache_parquet("id_crosswalk")
        if cached is not None:
            return cached
    logger.info("Fetching player ID crosswalk...")
    df = _to_pandas(nfl.load_ff_playerids())
    for col in ("sleeper_id", "gsis_id", "pfr_id", "espn_id", "yahoo_id", "mfl_id"):
        df = _normalize_id_column(df, col)
    _write_cache_parquet(df, "id_crosswalk")
    return df


# ==========================================================================
# SLEEPER FETCHERS
# ==========================================================================
def _sleeper_get(url: str, required: bool = True) -> dict | list | None:
    """
    GET a Sleeper URL.

    Raises by default rather than returning None. A silent empty result is
    worse than a crash here: an empty projections frame reads as "no data
    published yet" when the real cause is a wrong URL or a bad league ID.
    Pass required=False only where a miss is genuinely expected.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        if required:
            raise RuntimeError(f"Sleeper request failed for {url}: {e}") from e
        logger.warning(f"Sleeper request failed for {url}: {e}")
        return None


def get_sleeper_league(league_id: str = DEFAULT_LEAGUE_ID, refresh: bool = False) -> dict:
    """
    Full Sleeper league metadata: settings, scoring_settings,
    roster_positions, season, name, etc.

    We use this so notebook computations honor the league's actual
    scoring rules (matching the dashboard's logic).
    """
    cache_name = f"sleeper_league_{league_id}"
    if not refresh:
        cached = _read_cache_json(cache_name)
        if cached is not None:
            return cached
    logger.info(f"Fetching Sleeper league {league_id}...")
    data = _sleeper_get(f"{SLEEPER_API}/league/{league_id}")
    if not data:
        raise RuntimeError(
            f"Sleeper returned no league for id {league_id}. Dynasty leagues get a "
            "new id each season — check DEFAULT_LEAGUE_ID."
        )
    _write_cache_json(data, cache_name)
    return data


def get_sleeper_players(refresh: bool = False) -> pd.DataFrame:
    """
    Sleeper's full NFL player DB — ~11k entries with player_id, name,
    position, team, injury_status, depth_chart_position, etc.

    This is a LARGE endpoint (~5MB) — Sleeper documents caching it once
    per day rather than per-request. We cache aggressively.

    Returns:
        DataFrame with one row per player, keyed by sleeper_id (string).
    """
    cache_name = "sleeper_players"
    if not refresh:
        cached_raw = _read_cache_json(cache_name)
        if cached_raw is not None:
            return _sleeper_players_to_df(cached_raw)
    logger.info("Fetching Sleeper player DB (this can take a few seconds)...")
    data = _sleeper_get(f"{SLEEPER_API}/players/nfl")
    _write_cache_json(data, cache_name)
    return _sleeper_players_to_df(data)


def _sleeper_players_to_df(raw: dict) -> pd.DataFrame:
    """Turn Sleeper's dict-of-dicts response into a DataFrame."""
    records = []
    for player_id, payload in raw.items():
        if not payload:
            continue
        payload = dict(payload)  # don't mutate the cached dict in place
        payload["sleeper_id"] = str(player_id)
        records.append(payload)
    df = pd.DataFrame(records)
    # Keep a lean set of the most useful columns — full response has ~60
    keep = [
        "sleeper_id", "first_name", "last_name", "full_name",
        "position", "team", "age", "years_exp", "status",
        "injury_status", "injury_body_part", "injury_notes",
        "injury_start_date", "depth_chart_position",
        "depth_chart_order", "fantasy_positions",
        "college", "height", "weight",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    return _normalize_id_column(df, "sleeper_id")


def get_sleeper_projections(
    season: int, week: int, refresh: bool = False
) -> pd.DataFrame:
    """
    Sleeper's projection for a single week — the baseline we're trying
    to beat with our custom model.

    Uses the same host and path as index.html (see line ~1924) so the
    notebook's benchmark matches the numbers the dashboard renders. Falls
    back to Sleeper's newer host if the primary path 404s, and raises if
    both come back empty rather than handing you a blank DataFrame.

    Returns:
        DataFrame with sleeper_id + projected stat fields (pass_yd,
        pass_td, rec, rec_yd, etc.) plus pre-aggregated pts_ppr /
        pts_half_ppr / pts_std.
    """
    cache_name = f"sleeper_proj_{season}_wk{week}"
    if not refresh:
        cached = _read_cache_json(cache_name)
        if cached is not None:
            return _sleeper_projections_to_df(cached)

    logger.info(f"Fetching Sleeper projections for {season} week {week}...")
    data = _sleeper_get(
        f"{SLEEPER_API}/projections/nfl/regular/{season}/{week}", required=False
    )
    if not data:
        logger.info("Primary projections path returned nothing; trying alt host...")
        data = _sleeper_get(
            f"{SLEEPER_API_ALT}/projections/nfl/{season}/{week}"
            "?season_type=regular&position[]=QB&position[]=RB"
            "&position[]=WR&position[]=TE&position[]=K&position[]=DEF",
            required=False,
        )
    if not data:
        raise RuntimeError(
            f"No Sleeper projections returned for {season} week {week}. Either the "
            "week hasn't been published yet or the endpoint moved — check the URL "
            "against what index.html uses before assuming it's a data gap."
        )
    _write_cache_json(data, cache_name)
    return _sleeper_projections_to_df(data)


def _sleeper_projections_to_df(raw: list | dict) -> pd.DataFrame:
    """
    Flatten Sleeper's projection response to a table.

    Handles both shapes Sleeper serves: a list of {player_id, stats} dicts,
    and a dict keyed by player_id whose values are the stat dicts.
    """
    if not raw:
        return pd.DataFrame()
    records = []
    if isinstance(raw, dict):
        for player_id, stats in raw.items():
            record = {"sleeper_id": str(player_id)}
            record.update(stats or {})
            records.append(record)
    else:
        for entry in raw:
            stats = entry.get("stats", {}) or {}
            record = {"sleeper_id": str(entry.get("player_id"))}
            record.update(stats)
            records.append(record)
    df = pd.DataFrame(records)
    return _normalize_id_column(df, "sleeper_id")


def get_sleeper_matchups(league_id: str, week: int, refresh: bool = False) -> pd.DataFrame:
    """
    Real weekly fantasy matchups — /league/{id}/matchups/{week}. One row
    per (roster_id, matchup_id): the roster's ACTUAL total points that
    week (Sleeper's own league-scoring total — ground truth for who won
    that matchup), `starters` (ordered sleeper player_ids, including
    K/DST slots), and `starters_points` (Sleeper's actual per-starter
    points that week, same order).

    Returns an EMPTY DataFrame (not a raise) if the week has no
    matchups — most leagues run a 14-17 week fantasy schedule inside the
    NFL's 18-week regular season, so a late week can be a genuine bye on
    the fantasy calendar rather than a broken fetch.
    """
    cache_name = f"sleeper_matchups_{league_id}_wk{week}"
    if not refresh:
        cached = _read_cache_json(cache_name)
        if cached is not None:
            return _sleeper_matchups_to_df(cached)
    logger.info(f"Fetching Sleeper matchups for league {league_id} week {week}...")
    data = _sleeper_get(f"{SLEEPER_API}/league/{league_id}/matchups/{week}", required=False)
    if data is None:
        data = []
    _write_cache_json(data, cache_name)
    return _sleeper_matchups_to_df(data)


def _sleeper_matchups_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["roster_id", "matchup_id", "points", "starters", "starters_points"])
    rows = [{
        "roster_id": entry.get("roster_id"),
        "matchup_id": entry.get("matchup_id"),
        "points": entry.get("points"),
        "starters": entry.get("starters") or [],
        "starters_points": entry.get("starters_points") or [],
    } for entry in raw]
    return pd.DataFrame(rows)


def get_sleeper_rosters(league_id: str, refresh: bool = False) -> list:
    """
    Real fantasy rosters — /league/{id}/rosters. One entry per roster_id,
    including `players` (every sleeper_id on that roster, any slot) and
    `starters`. A newly-created league returns one empty-`players` entry per
    roster_id until the draft happens — that's real, current league state,
    not a fetch failure, so this doesn't raise on an empty players list the
    way other Sleeper fetchers raise on a fully empty response.

    Kept as the raw list (matches get_sleeper_bracket's convention) since
    downstream code typically just wants a fast set-membership check over
    every rostered player_id, not a flattened table.
    """
    cache_name = f"sleeper_rosters_{league_id}"
    if not refresh:
        cached = _read_cache_json(cache_name)
        if cached is not None:
            return cached
    logger.info(f"Fetching Sleeper rosters for league {league_id}...")
    data = _sleeper_get(f"{SLEEPER_API}/league/{league_id}/rosters")
    _write_cache_json(data, cache_name)
    return data


def get_sleeper_bracket(league_id: str, refresh: bool = False) -> list:
    """
    Real playoff bracket — /league/{id}/winners_bracket. Only meaningful
    once a season's playoffs have started/completed; each entry is one
    bracket match: `m` (match id), `r` (round), `t1`/`t2` (roster_ids, OR
    a reference to an earlier match via `t1_from`/`t2_from` — {"w": m}
    for that match's winner, {"l": m} for its loser).

    Kept as the raw list (not flattened to a DataFrame) since its shape
    is irregular — see playoff_participants_from_bracket() for the one
    thing this project needs out of it.
    """
    cache_name = f"sleeper_bracket_{league_id}"
    if not refresh:
        cached = _read_cache_json(cache_name)
        if cached is not None:
            return cached
    logger.info(f"Fetching Sleeper winners_bracket for league {league_id}...")
    data = _sleeper_get(f"{SLEEPER_API}/league/{league_id}/winners_bracket", required=False)
    if data is None:
        data = []
    _write_cache_json(data, cache_name)
    return data


def playoff_participants_from_bracket(bracket: list) -> set:
    """
    The set of roster_ids that actually made the playoffs, read off a
    completed winners_bracket.

    A match's t1/t2 slot is a REAL seeded participant only when there's
    no t1_from/t2_from key for that slot — a "_from" key means the slot
    is filled by an earlier match's winner or loser, not a new playoff
    entrant. E.g. a bye team enters round 2 directly as a literal t1
    with no t1_from, while everyone advancing out of round 1 reaches
    round 2 only via t2_from. Taking the union of every direct slot
    across the whole bracket recovers exactly the league's playoff
    field, including any top-seed byes, without needing to already know
    the league's playoff_teams count.
    """
    participants = set()
    for match in bracket:
        if "t1_from" not in match and match.get("t1") is not None:
            participants.add(match["t1"])
        if "t2_from" not in match and match.get("t2") is not None:
            participants.add(match["t2"])
    return participants


# ==========================================================================
# CONVENIENCE: full Phase 1 fetch bundle
# ==========================================================================
def fetch_all(
    seasons: Iterable[int],
    include_pbp: bool = True,
    include_ngs: bool = True,
    league_id: str = DEFAULT_LEAGUE_ID,
) -> dict:
    """
    Convenience — pull everything Phase 1 needs in one call.

    Args:
        seasons: iterable of years, e.g. [2024, 2025]
        include_pbp: play-by-play is the largest fetch; disable for
            quick exploratory work when you don't need per-play detail
        include_ngs: Next Gen Stats — only relevant if analyzing
            aDOT / separation / time to throw
        league_id: Sleeper league to fetch

    Returns:
        Dict of DataFrames keyed by source name.
    """
    seasons = list(seasons)
    bundle = {
        "weekly": get_weekly_stats(seasons),
        "snaps": get_snap_counts(seasons),
        "schedule": get_schedule(seasons),
        "rosters": get_seasonal_rosters(seasons),
        "crosswalk": get_id_crosswalk(),
        "sleeper_players": get_sleeper_players(),
        "sleeper_league": get_sleeper_league(league_id),
    }
    if include_pbp:
        bundle["pbp"] = get_pbp(seasons)
    if include_ngs:
        bundle["ngs_passing"] = get_ngs_data("passing", seasons)
        bundle["ngs_receiving"] = get_ngs_data("receiving", seasons)
        bundle["ngs_rushing"] = get_ngs_data("rushing", seasons)
    return bundle
