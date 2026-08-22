"""
FanTeasy Stats -- Phase 8: shared feature-building orchestration.

Factors out the fetch -> score -> Family 1-6 feature sequence that
`02_custom_scoring.ipynb` and `03_usage_features.ipynb` already run cell by
cell, so `scripts/retrain.py` and `scripts/weekly_update.py` (Phase 8) call
ONE shared implementation instead of each re-expressing the same sequence --
see CLAUDE.md's "reusable logic goes in src/, not in notebooks." The
notebooks remain the exploratory, cell-by-cell reference for this same
sequence; nothing here changes what they compute.

Split into two steps, not one, because the two Phase 8 scripts need to
insert themselves at different points:
  - `build_weekly_scored` + `build_raw_features`: Families 1-5 and xFP,
    every one of which is computed independently per (player, week) with
    no cross-season lookback. Safe to run on a single season in isolation.
  - `add_rolling_features` (imported from src.usage, not wrapped here):
    Family 6 groups by (player_id, season) and needs `prev_season_*` to
    look back a full season -- it must run on a frame that ALREADY spans
    the season(s) before the one being featured. `build_feature_table`
    below does this in one shot for retrain (which has every season at
    once); `scripts/weekly_update.py` instead concatenates a small
    multi-season history seed with one season of fresh raw features and
    calls `add_rolling_features` itself, via the exact same mechanism
    `src/export.py::build_target_week_features` already uses to extend a
    historical frame with a not-yet-played target week.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

from src.features import add_pick_six_column, compute_custom_score
from src.ingest import (
    DEFAULT_LEAGUE_ID,
    get_id_crosswalk,
    get_ngs_data,
    get_pbp,
    get_schedule,
    get_sleeper_league,
    get_snap_counts,
    get_weekly_stats,
)
from src.usage import (
    FANTASY_POSITIONS,
    ROLLING_SOURCE_COLUMNS,
    add_context_features,
    add_efficiency_features,
    add_rolling_features,
    add_situational_features,
    add_snap_features,
    add_volume_features,
    add_xfp_features,
)

logger = logging.getLogger(__name__)

# The minimal column set a not-yet-featured season needs to seed
# add_rolling_features/add_trend_features for a LATER season, without
# carrying the full ~340-column feature table -- see src/artifacts.py for
# where this gets trimmed to and why. player_id/position/team/season/week
# are identifying and join columns; offense_pct + ROLLING_SOURCE_COLUMNS
# are exactly what add_rolling_features requires as input (its own
# `required` check); custom_points is the one extra column
# src/export.py::build_xfp_summary needs that ISN'T itself part of
# ROLLING_SOURCE_COLUMNS (xfp is; the actual outcome it's compared against
# is not).
HISTORY_SEED_COLUMNS = list(dict.fromkeys(
    ["player_id", "position", "team", "season", "week", "offense_pct", "custom_points"]
    + list(ROLLING_SOURCE_COLUMNS)
))  # offense_pct is ALSO part of ROLLING_SOURCE_COLUMNS (SNAP_OUTPUT_COLUMNS) --
   # dict.fromkeys dedupes while preserving order, since a duplicate column
   # name here breaks pd.concat downstream with a confusing
   # "Reindexing only valid with uniquely valued Index objects" error.


def _is_unpublished_season_error(exc: Exception) -> bool:
    """
    True if `exc` is nflreadpy's own signal that a requested season's file
    doesn't exist on nflverse-data YET -- an HTTP 404, wrapped by
    nflreadpy's downloader into a bare ConnectionError with the original
    requests.HTTPError chained as __cause__. Distinguishes "this season
    genuinely hasn't started" (expected during the offseason/preseason,
    safe to treat as zero rows) from a real network/auth failure (should
    still fail loudly, per src/ingest.py's own "fail loudly" convention --
    this function does NOT relax that for any other kind of error).
    """
    cause = exc.__cause__
    return (
        isinstance(cause, requests.exceptions.HTTPError)
        and cause.response is not None
        and cause.response.status_code == 404
    )


def build_weekly_scored(seasons: list[int], league_id: str = DEFAULT_LEAGUE_ID) -> pd.DataFrame:
    """
    Phase 1 + 2a: weekly stats, filtered to REG season QB/RB/WR/TE, scored
    with this league's actual custom rules. Mirrors
    `02_custom_scoring.ipynb`'s `weekly_scored.parquet` build.

    add_pick_six_column runs BEFORE compute_custom_score, not after --
    `pass_int_td` is the one active scoring rule with no weekly-stats
    column (pick-sixes have to be derived from pbp: an interception
    returned for a touchdown, charged to the passer), and
    compute_custom_score reads the `pass_int_tds` column it produces.
    Skipping this silently zeroes out every pick-six thrown -- see
    CLAUDE.md's "The scoring code is load-bearing" section.

    A 404 for an unpublished season (see _is_unpublished_season_error) is
    treated as zero rows, not raised -- this is the ONE caller in this
    pipeline that legitimately expects that (scripts/weekly_update.py,
    predicting week 1 of a season with no games played yet). Every other
    caller (retrain.py's HISTORICAL_SEASONS) requests seasons known to be
    published, so a 404 there is a real bug and still propagates.
    """
    try:
        weekly = get_weekly_stats(seasons)
    except ConnectionError as e:
        if len(seasons) == 1 and _is_unpublished_season_error(e):
            logger.info(f"build_weekly_scored: season {seasons[0]} has no published weekly stats yet -- returning zero rows.")
            return pd.DataFrame()
        raise
    reg = weekly[
        (weekly["season_type"] == "REG") & weekly["position"].isin(FANTASY_POSITIONS)
    ].reset_index(drop=True)
    if reg.empty:
        logger.warning(f"build_weekly_scored: zero REG QB/RB/WR/TE rows for seasons {seasons}")
        return reg
    pbp = get_pbp(seasons)
    reg = add_pick_six_column(reg, pbp)
    scoring_settings = get_sleeper_league(league_id)["scoring_settings"]
    reg["custom_points"] = compute_custom_score(reg, scoring_settings)
    return reg


def build_raw_features(
    weekly_scored: pd.DataFrame, seasons: list[int], league_id: str = DEFAULT_LEAGUE_ID
) -> pd.DataFrame:
    """
    Families 1-5 and xFP -- everything add_rolling_features (Family 6)
    needs as input, deliberately stopping short of calling it. See this
    module's docstring for why: Family 6 needs multi-season context that a
    single-season call to this function doesn't have on its own.

    Safe to call with `weekly_scored` empty (e.g. a not-yet-started
    season) -- every add_* function below is a left-merge/derivation over
    its own input rows, so an empty frame in produces an empty frame out,
    not an error.

    KNOWN GAP when `seasons` is a single in-progress season (as
    scripts/weekly_update.py calls this): add_xfp_features's bucket rate
    table is an EXPANDING window over whatever `pbp` this call receives
    (see its own docstring and PROJECT_CONTEXT.md's "xFP rate table...
    two-season compromise" note) -- fed only this season's own pbp, the
    rate table for an early week has just that week's few hundred plays to
    average, not the multi-season history retrain.py's full-history call
    builds it from. xfp/fp_over_expected (and their rolled _ewm3/_s2d/_vol
    versions, a handful of this pipeline's ~180 model features) are
    therefore noisier early in a season under weekly-only inference than
    what the model was trained against, converging toward normal as the
    season accumulates its own plays. Disclosed in the exported JSON's
    meta.caveats (see scripts/weekly_update.py's WEEKLY_EXTRA_CAVEATS)
    rather than fixed here -- a real fix means feeding this call multiple
    prior seasons' pbp too, which is exactly the fetch-cost "current
    season only" is deliberately avoiding.
    """
    if weekly_scored.empty:
        return weekly_scored

    pbp = get_pbp(seasons)
    snaps = get_snap_counts(seasons)
    crosswalk = get_id_crosswalk()
    schedule = get_schedule(seasons)
    scoring_settings = get_sleeper_league(league_id)["scoring_settings"]
    ngs_receiving = get_ngs_data("receiving", seasons)
    ngs_passing = get_ngs_data("passing", seasons)

    df = add_volume_features(weekly_scored, pbp)
    df = add_snap_features(df, snaps, crosswalk)
    df = add_efficiency_features(df, pbp, ngs_receiving, ngs_passing)
    df = add_situational_features(df, pbp)
    df = add_context_features(df, schedule)
    df = add_xfp_features(df, pbp, scoring_settings)
    return df


def build_feature_table(seasons: list[int], league_id: str = DEFAULT_LEAGUE_ID) -> pd.DataFrame:
    """
    Full Phase 1-2b pipeline for retrain: every season fetched and
    featured together, so Family 6's prev_season_*/in-season rolling
    columns see the whole multi-season history in one call. Mirrors
    `03_usage_features.ipynb`'s `weekly_features.parquet` build.
    """
    weekly_scored = build_weekly_scored(seasons, league_id)
    raw = build_raw_features(weekly_scored, seasons, league_id)
    if raw.empty:
        return raw
    return add_rolling_features(raw)
