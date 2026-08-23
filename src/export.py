"""
FanTeasy Stats -- Phase 7: JSON export.

Produces data/output/player_advanced_stats.json: this league's custom model's
projection (point/floor/ceiling), a trailing usage snapshot, a Phase 3'
usage-trend block (current opportunity share, a normalized trend signal, and
a rising/falling/stable direction label per player -- see src/usage.py's
Family 7), and a season-long xFP/luck summary, for the dashboard to consume
-- keyed by Sleeper player_id, since Sleeper's ID is the only one the
dashboard has (it never sees gsis_id; that's an internal, nflverse-side
detail of this pipeline).

Scope: this league's real 2026 rosters, union the top ~300 remaining
QB/RB/WR/TE by this model's own point projection. K/DST are excluded
entirely -- src/model.py's projection model has never covered them (see its
module docstring and CLAUDE.md's scope boundaries), and faking a projection
for a position with zero model coverage would be exactly the "overstated
model" failure the dashboard's own no-fake-data principle exists to prevent.
The dashboard already shows Sleeper's own K/DST numbers directly, unaffected
by this file.

Predicting a real, not-yet-played week (there is no historical fold to hold
out against it) reuses the existing point-in-time-safe feature pipeline
unmodified, rather than inventing a new one:
  1. Append one STUB row per candidate player for (target_season,
     target_week) to the real historical feature table, with every Family
     1-4/xFP source column left null -- honestly "not yet known," not
     fabricated.
  2. Re-run add_context_features() (src/usage.py) over the combined frame.
     Family 5 needs only the real, already-published schedule -- no play
     data required -- so this works for an unplayed week exactly as it does
     for a played one.
  3. Re-run add_rolling_features() (src/usage.py) over the combined frame.
     Grouped by (player_id, season), the stub row is the FIRST row in a new
     (player, 2026) group: every in-season _ewm3/_vol/_s2d column comes out
     null by the exact same shift(1)-into-an-empty-group mechanics that
     already make week 1 of every OTHER season null in the training data --
     no special-casing required. prev_season_* correctly carries the
     player's real, complete prior-season (2025) averages forward, because
     that computation only ever looks at season S's own rows, which are
     unaffected by whether season S+1 has a real game yet.
Nothing about this needed new feature logic -- only a correctly-shaped stub
row and reusing the exact functions every historical week already goes
through.

Two models per position, both untuned (matching every other model in this
project -- no hyperparameter search anywhere in this pipeline), both trained
on ALL available history (2018 through the week before the target week, with
no fold held out -- there's nothing to hold out against a week that hasn't
been played yet):
  - a plain regression model for `projection.point` (LightGBM,
    objective='regression') -- the Formulation A model from src/model.py,
    trained fresh on the full window instead of walk-forward folds.
  - `objective='quantile'` models at alpha=0.10/0.90 for
    `projection.floor`/`ceiling`, widened by the ALREADY-DERIVED CQR
    constants published in PROJECT_CONTEXT.md's Phase 6 findings (not
    re-derived here -- they're deterministic given the same model, data, and
    random_state, so re-running the ~140-fold calibration reservoir again
    would just reproduce the same eight numbers at real cost for no new
    information).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.model import FEATURE_COLUMNS, _cast_categoricals, predict_with_models, train_final_models
from src.simulate import simulate_matchup, simulate_season

logger = logging.getLogger(__name__)

EXPORT_POSITIONS = ("QB", "RB", "WR", "TE")

# Sleeper's player DB uses its own team codes, which differ from nflverse's
# (schedule/pbp) codes for exactly one currently-active team: the Rams.
# Everything else matches. Checked directly against both sources rather than
# assumed -- see the Phase 7 notebook's data-check cell.
SLEEPER_TO_NFLVERSE_TEAM = {"LAR": "LA"}

# CQR widening for the 10th-90th interval, per position -- reused verbatim
# from the calibration already published in PROJECT_CONTEXT.md's Phase 6
# findings (derived from a calibration split of 2018 Wk5-2024 Wk4, strictly
# before any locked evaluation fold). Not re-derived here.
CQR_WIDEN_BY_10_90 = {"QB": 2.309, "RB": 0.730, "WR": 0.606, "TE": 0.467}

# CQR widening for the 25th-75th interval -- src/simulate.py's per-player
# distributions need this pair too (pred_q25_cqr/pred_q75_cqr), which
# nothing before Phase 8 (Round 2) ever consumed, so no export.py constant
# for it existed yet. Derived from PROJECT_CONTEXT.md's already-published
# Phase 6 CQR table (same calibration run as CQR_WIDEN_BY_10_90 above, same
# split) rather than re-run: that table reports width BEFORE/AFTER
# widening for both interval pairs, and widen_by = (width_after -
# width_before) / 2 since the constant is added to each side. Verified
# against the table's own 10-90 row before trusting this on the 25-75 row:
# recomputing CQR_WIDEN_BY_10_90 this same way from the table's rounded
# before/after widths reproduces QB 2.31 / RB 0.73 / WR 0.605 / TE 0.47 --
# matching the hardcoded constants above to within the table's own
# 2-decimal rounding. Not re-derived on every retrain, same reasoning as
# CQR_WIDEN_BY_10_90 -- see scripts/retrain.py's module docstring.
CQR_WIDEN_BY_25_75 = {"QB": 1.265, "RB": 0.465, "WR": 0.380, "TE": 0.355}

# Formulation A's already-published 8-season MAE per position, against
# Sleeper's projection and the season-to-date-average baseline -- from
# PROJECT_CONTEXT.md's Phase 6 findings data-volume table. Static facts about
# an already-validated model, not something to recompute per export.
PERFORMANCE_BY_POSITION = {
    "QB": {"model_mae": 6.38, "sleeper_mae": 5.54, "s2d_mae": 6.61},
    "RB": {"model_mae": 4.18, "sleeper_mae": 3.87, "s2d_mae": 4.25},
    "WR": {"model_mae": 3.93, "sleeper_mae": 3.77, "s2d_mae": 3.99},
    "TE": {"model_mae": 3.06, "sleeper_mae": 2.88, "s2d_mae": 3.12},
}

CAVEATS = [
    "Sleeper's projections are more accurate than this model at every position.",
    "Floor/ceiling intervals are conformally calibrated; ceilings run conservative.",
    "The red_zone_share trend signal is real but noisier than snap/target/carry "
    "share -- red-zone opportunities are low-volume, and a 'rising' read reverts "
    "more often than it holds even at the validated window and threshold.",
]

# A curated subset of ROLLING_OUTPUT_COLUMNS for the `usage` block -- role
# and workload trend, not the full ~180-column feature set. Genuinely null
# for positions the underlying stat doesn't apply to (e.g. target_share_ewm3
# for a QB), same as everywhere else in this pipeline -- not a bug to fix.
#
# Includes both the in-season _ewm3 trend AND the matching prev_season_*
# baseline as SEPARATE, distinctly-named fields, deliberately not coalesced
# into one key: at week 1 of a season, every _ewm3 column is null by
# construction (nothing has been played yet this season -- the same
# mechanism that makes every OTHER season's week 1 null in training data),
# so a week-1 export would otherwise ship an all-null usage block. Blending
# last season's average into the SAME key a reader would reasonably assume
# means "this season's trend" would be a labeling problem even though the
# underlying number is real, not fabricated -- so both are exposed under
# their own honest names instead, and the dashboard can choose how to
# display/fall back between them.
#
# xfp_vol (Phase 8 round 3) -- expanding std of xFP, i.e. how much this
# player's OPPORTUNITY-implied points have swung week to week. Added as
# this pipeline's "boom/bust" / volatility signal for the dashboard's
# Player Detail page: no column here is fantasy-POINTS volatility itself
# (Family 6 deliberately excludes custom_points, the model's own target,
# from ROLLING_SOURCE_COLUMNS -- see src/model.py's module docstring), and
# xfp_vol is the closest already-computed, points-denominated proxy to
# it -- unlike a single usage-share's volatility, xFP already blends
# targets, carries, and field position into one points-scale number, so
# its volatility reads as "swings in scoring OPPORTUNITY" rather than one
# narrow usage metric. Null for QB by construction (xfp has no passing
# counterpart, see add_xfp_features's docstring) -- same disclosed gap
# already true of every other xFP-derived field in this export.
USAGE_EXPORT_COLUMNS = [
    "target_share_ewm3", "touch_share_ewm3", "offense_pct_ewm3",
    "snap_share_delta_3wk", "rz_target_share_ewm3", "rz_carry_share_ewm3",
    "prev_season_target_share", "prev_season_touch_share", "prev_season_offense_pct",
    "xfp_vol",
]

# Phase 3' trend block: src/usage.py's internal column-name feature ->
# the human-readable key the export uses. rz_opportunity_share is renamed
# rather than reused verbatim because it means something a reader has to
# infer from Family 4's raw column names (rz_targets + rz_carries over
# their respective team totals) -- "red_zone_share" says what it is without
# that context. The others keep their existing names since they're already
# plain English.
TREND_FEATURE_LABELS = {
    "offense_pct": "snap_share",
    "target_share": "target_share",
    "carry_share": "carry_share",
    "rz_opportunity_share": "red_zone_share",
}

# ==========================================================================
# PHASE 4: RADAR PERCENTILES
# ==========================================================================
# Six axes per position, each an ALREADY-COMPUTED Family 1-4 `_s2d` column
# (season-to-date expanding mean, point-in-time safe -- see
# add_rolling_features's docstring) -- deliberately not `_ewm3` (the
# trailing-3-week value the Opportunity Shares panel and the trend
# indicator already show elsewhere on Player Detail): a "position profile"
# is meant to characterize what KIND of player this is over a real season
# sample, not this week's hot/cold streak, and `_s2d` is the more stable
# input for a percentile RANK specifically -- a 3-week-EWM percentile would
# jump around week to week for reasons that have nothing to do with the
# player's underlying role. Nothing here is a new feature computation: no
# NGS-only column (avg_separation/avg_cushion/time_to_throw) is used, so
# every axis is available for every position it's assigned to.
#
# NOTEBOOK_OUTLINE.md's Phase 4 sketch names several axes this pipeline has
# never actually computed (Big Play Rate, TD Rate, Sack %, Explosive Runs,
# Contested Catch %, YPRR, Blocking Snaps, Dome %, ST TDs) -- that sketch
# predates any real feature work (see CLAUDE.md: "Phases 3-8 were written
# before anything ran, so their code snippets are intent, not tested
# code"). The axes below are chosen from what Families 1-4/xFP actually
# produce, per position, rather than forced to match that list.
#
# `unit` is a plain display suffix, not a format spec -- index.html just
# concatenates raw + unit, no per-axis-name branching needed there. `pct`
# marks columns that are 0-1 fractions needing x100 before either the UI
# or a human reads them as a percentage; everything else (yards, EPA,
# CPOE's percentage-POINT delta, counts) is already in its natural,
# human-facing scale.
RADAR_METRICS = {
    "QB": [
        {"column": "pass_attempts_s2d", "label": "Pass Volume", "unit": " att/gm", "pct": False},
        {"column": "designed_rush_attempts_s2d", "label": "Rush Volume", "unit": " att/gm", "pct": False},
        {"column": "yards_per_carry_s2d", "label": "Yards / Carry", "unit": " yds", "pct": False},
        {"column": "scramble_rate_s2d", "label": "Scramble Rate", "unit": "%", "pct": True},
        {"column": "epa_per_dropback_s2d", "label": "EPA / Dropback", "unit": " EPA", "pct": False},
        {"column": "cpoe_s2d", "label": "Comp % Over Expected", "unit": " pts", "pct": False},
    ],
    "RB": [
        {"column": "touches_s2d", "label": "Touch Volume", "unit": " touches/gm", "pct": False},
        {"column": "touch_share_s2d", "label": "Touch Share", "unit": "%", "pct": True},
        {"column": "target_share_s2d", "label": "Target Share", "unit": "%", "pct": True},
        {"column": "yards_per_carry_s2d", "label": "Yards / Carry", "unit": " yds", "pct": False},
        {"column": "goal_line_carry_share_s2d", "label": "Goal-Line Share", "unit": "%", "pct": True},
        {"column": "offense_pct_s2d", "label": "Snap Share", "unit": "%", "pct": True},
    ],
    "WR": [
        {"column": "target_share_s2d", "label": "Target Share", "unit": "%", "pct": True},
        {"column": "air_yards_share_s2d", "label": "Air Yards Share", "unit": "%", "pct": True},
        {"column": "adot_s2d", "label": "Avg Depth of Target", "unit": " yds", "pct": False},
        {"column": "catch_rate_s2d", "label": "Catch Rate", "unit": "%", "pct": True},
        {"column": "yac_per_reception_s2d", "label": "YAC / Reception", "unit": " yds", "pct": False},
        {"column": "rz_target_share_s2d", "label": "Red-Zone Target Share", "unit": "%", "pct": True},
    ],
    "TE": [
        {"column": "target_share_s2d", "label": "Target Share", "unit": "%", "pct": True},
        {"column": "offense_pct_s2d", "label": "Snap Share", "unit": "%", "pct": True},
        {"column": "catch_rate_s2d", "label": "Catch Rate", "unit": "%", "pct": True},
        {"column": "adot_s2d", "label": "Avg Depth of Target", "unit": " yds", "pct": False},
        {"column": "rz_target_share_s2d", "label": "Red-Zone Target Share", "unit": "%", "pct": True},
        {"column": "yac_per_reception_s2d", "label": "YAC / Reception", "unit": " yds", "pct": False},
    ],
}

# FLEX-eligible slot names and each position's assumed share of a FLEX
# start -- must stay byte-for-byte in sync with index.html's
# positionStarterCount() (search that name). Not shared code (index.html
# has no Python runtime to import from and this pipeline has no JS runtime
# to import from) -- kept in lockstep instead by
# tests/test_export.py::test_position_starter_counts_matches_frontend_logic,
# which pins this function's output against this league's REAL
# roster_positions so either side drifting from the other fails a test,
# not just a silent mismatch a reader would have to notice by eye.
_FLEX_SLOT_NAMES = {"FLEX", "WRRB_FLEX", "REC_FLEX"}
_FLEX_SHARE_BY_POSITION = {"WR": 0.5, "RB": 0.35, "TE": 0.15}
_DEFAULT_STARTER_SHARE = {"QB": 1, "RB": 2.5, "WR": 3, "TE": 1.2}


def position_starter_counts(roster_positions: list[str], n_teams: int) -> dict[str, int]:
    """
    League-wide startable-slot count per EXPORT_POSITIONS position -- the
    radar's percentile pool is ranked against exactly these many players,
    not every player at the position (which would include deep backups
    and compress every real contributor into the top of the range).

    Direct Python port of index.html's positionStarterCount(), which
    already powers the Weekly Production chart's Top-N baseline -- same
    direct-slot + FLEX-share + SUPER_FLEX logic, so "startable" means the
    same thing in both places. `round()` here uses round-half-up (matching
    JS's Math.round(), not Python's own round-half-to-even) so the two
    stay identical even on a future league config that lands exactly on
    a .5 boundary -- this league's real roster_positions (QB=14, RB=33,
    WR=35, TE=16 at 14 teams) don't currently hit one, but the port
    shouldn't silently diverge the day one does.
    """
    import math

    counts = {}
    for position in EXPORT_POSITIONS:
        direct = roster_positions.count(position)
        flex_share = sum(
            _FLEX_SHARE_BY_POSITION.get(position, 0.0)
            for slot in roster_positions if slot in _FLEX_SLOT_NAMES
        )
        super_flex_share = roster_positions.count("SUPER_FLEX") if position == "QB" else 0
        raw = direct + flex_share + super_flex_share
        if raw == 0:
            raw = _DEFAULT_STARTER_SHARE.get(position, 1)
        counts[position] = max(1, math.floor(raw * n_teams + 0.5))
    return counts


# ==========================================================================
# STEP 1: build the point-in-time-safe feature row for the target week
# ==========================================================================
def normalize_team_code(team: str) -> str:
    return SLEEPER_TO_NFLVERSE_TEAM.get(team, team)


def get_export_candidates(
    historical_features: pd.DataFrame, sleeper_players: pd.DataFrame, crosswalk: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """
    The universe of players eligible for a projection, built starting from
    this project's OWN gsis_id-space data -- not from Sleeper's player list
    -- so the crosswalk match rate reported here measures something real
    (of every QB/RB/WR/TE we have actual history for, how many can be found
    in Sleeper's system) rather than being tautologically 100% because
    unmatched players were filtered out before the rate was ever computed.

    Two distinct reasons a position-eligible player can still drop out,
    reported separately since they mean different things:
      - no Sleeper crosswalk match at all (a real ID-matching gap)
      - a Sleeper match with no current team on file (a genuinely unsigned
        free agent -- there's no game to attach pregame context to, which
        isn't a data problem to fix, just a real player with no upcoming
        game to project)

    A player with zero real NFL history (never appears in
    `historical_features`) is never a candidate at all -- there's nothing
    for add_rolling_features to build prev_season_*/trend features from, so
    a true rookie is honestly out of scope here, not defaulted to a
    league-average guess.

    Returns:
        (candidates, report) -- candidates has player_id (gsis_id),
        position, team (normalized to nflverse's codes). report has
        n_position_eligible, n_crosswalk_matched, n_with_current_team,
        crosswalk_match_rate.
    """
    position_eligible = (
        historical_features[historical_features["position"].isin(EXPORT_POSITIONS)][["player_id", "position"]]
        .drop_duplicates(subset=["player_id"])
    )

    cw = crosswalk.dropna(subset=["gsis_id", "sleeper_id"]).drop_duplicates(subset=["gsis_id"])
    merged = position_eligible.merge(cw[["gsis_id", "sleeper_id"]], left_on="player_id", right_on="gsis_id", how="left")
    n_crosswalk_matched = int(merged["sleeper_id"].notna().sum())

    team_lookup = sleeper_players[["sleeper_id", "team"]].dropna(subset=["team"])
    merged = merged.merge(team_lookup, on="sleeper_id", how="left")
    merged["team"] = merged["team"].map(normalize_team_code)
    n_with_current_team = int(merged["team"].notna().sum())

    candidates = merged.dropna(subset=["team"])[["player_id", "position", "team"]].reset_index(drop=True)
    report = {
        "n_position_eligible": len(position_eligible),
        "n_crosswalk_matched": n_crosswalk_matched,
        "n_with_current_team": n_with_current_team,
        "crosswalk_match_rate": n_crosswalk_matched / len(position_eligible) if len(position_eligible) else float("nan"),
    }
    return candidates, report


def build_target_week_features(
    historical_features: pd.DataFrame,
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> pd.DataFrame:
    """
    Returns the COMBINED frame (real history + one stub row per candidate
    for target_season/target_week), with Family 5 context and Family 6
    rolling features recomputed over the whole thing -- see the module
    docstring for why this reuses add_context_features/add_rolling_features
    unmodified rather than writing new future-facing feature logic.

    Categorical columns (roof, surface) are cast ONCE here, on the combined
    frame, so the later train/predict split always shares the same category
    set -- the same reason src/model.py's walk_forward_predict casts before
    splitting into folds, not after.
    """
    from src.usage import (
        ROLLING_SOURCE_COLUMNS, add_context_features, add_rolling_features, add_trend_features,
    )

    stub = candidates.copy()
    stub["season"] = target_season
    stub["week"] = target_week
    for col in ["offense_pct"] + list(ROLLING_SOURCE_COLUMNS):
        stub[col] = np.nan

    combined = pd.concat([historical_features, stub], ignore_index=True, sort=False)
    combined = add_context_features(combined, schedule)
    combined = add_rolling_features(combined)
    # Phase 3' trend signal -- downstream of the model's own feature set
    # (add_trend_features's outputs are never added to FEATURE_COLUMNS), so
    # running it here doesn't change what predict_target_week trains on.
    combined = add_trend_features(combined)
    combined = _cast_categoricals(combined, FEATURE_COLUMNS)
    return combined


# ==========================================================================
# STEP 2: train final (no-holdout) models and predict the target week
# ==========================================================================
def predict_target_week(
    combined_features: pd.DataFrame,
    target_season: int,
    target_week: int,
    feature_cols: list[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Per position: one regression model (for `point`) and one q10/q90
    quantile pair (for `floor`/`ceiling`, CQR-widened), each trained on
    every row strictly before (target_season, target_week) -- there is no
    fold to hold out against a week that hasn't been played, so this trains
    on everything available, once, rather than walk-forward folds.

    Training and prediction are delegated to src/model.py's
    train_final_models/predict_with_models -- the same two functions
    Phase 8's retrain.yml uses to produce a model artifact and
    weekly-update.yml uses to predict FROM one (see
    predict_target_week_from_artifact below). Keeping this function as a
    thin train-then-predict wrapper over those two, rather than its own
    parallel fit/predict logic, means the "trained fresh right now" and
    "trained earlier, loaded from disk" code paths can never quietly
    diverge in how floor/ceiling get built.

    Returns: player_id, position, point, floor, ceiling -- one row per
    (position, player) in combined_features' target week.
    """
    train_mask = (
        (combined_features["season"] < target_season)
        | ((combined_features["season"] == target_season) & (combined_features["week"] < target_week))
    )
    test_mask = (combined_features["season"] == target_season) & (combined_features["week"] == target_week)

    train_df = combined_features[train_mask]
    test_df = combined_features[test_mask]
    models = train_final_models(train_df, feature_cols=feature_cols, positions=EXPORT_POSITIONS)
    return predict_with_models(test_df, models, CQR_WIDEN_BY_10_90, feature_cols=feature_cols)


def predict_target_week_from_artifact(
    combined_features: pd.DataFrame,
    target_season: int,
    target_week: int,
    artifact: dict,
) -> pd.DataFrame:
    """
    Inference-only counterpart to predict_target_week: no training here at
    all, just predict_with_models against models ALREADY TRAINED and saved
    by retrain.yml (see src/artifacts.py::load_model_artifact). This is
    what weekly-update.yml calls -- it keeps the model fixed between
    retrains, so week-to-week output changes come from new data, not a
    retrained model.

    Uses the artifact's OWN feature_columns/cqr_widen_by_10_90 (not this
    module's FEATURE_COLUMNS/CQR_WIDEN_BY_10_90 constants) -- the artifact
    is self-describing so a weekly run always matches whatever retrain.yml
    actually trained, even if this module's constants are edited later.

    Returns: same shape as predict_target_week.
    """
    feature_cols = artifact["feature_columns"]
    test_mask = (combined_features["season"] == target_season) & (combined_features["week"] == target_week)
    test_df = combined_features[test_mask]
    return predict_with_models(test_df, artifact["models"], artifact["cqr_widen_by_10_90"], feature_cols=feature_cols)


# ==========================================================================
# STEP 3: usage snapshot and season-long xFP summary
# ==========================================================================
def build_usage_snapshot(combined_features: pd.DataFrame, target_season: int, target_week: int) -> pd.DataFrame:
    """The target week's rolling-usage columns (USAGE_EXPORT_COLUMNS), one
    row per player -- this IS the model's own input, not a separate
    computation, so it's guaranteed consistent with the projection."""
    test_mask = (combined_features["season"] == target_season) & (combined_features["week"] == target_week)
    cols = ["player_id"] + USAGE_EXPORT_COLUMNS
    return combined_features.loc[test_mask, cols].reset_index(drop=True)


def build_trend_snapshot(combined_features: pd.DataFrame, target_season: int, target_week: int) -> pd.DataFrame:
    """
    Phase 3' trend block, one row per player: for each of
    TREND_SOURCE_FEATURES, the current (recent, ewm3) opportunity share
    stated plainly, the normalized trend signal, and its direction label.
    Same "this IS the model's own input, not a separate computation" note
    as build_usage_snapshot -- combined_features already has
    add_trend_features applied (see build_target_week_features).

    "Current" reuses the SAME <feat>_ewm3 column already exposed under
    USAGE_EXPORT_COLUMNS (e.g. target_share_ewm3) -- there's no raw
    this-week value to show for a not-yet-played target week, so "current"
    honestly means "most recent known," the same recency estimate the
    signal itself is built from. Null at week 1 of a season for the same
    reason every other in-season rolling column is null there (nothing
    played yet this season) -- not a bug, see add_rolling_features's
    season-boundary docstring note.

    The <feat>_ewm3 columns are renamed to trend_<feat>_current on the way
    out -- target_share_ewm3 and offense_pct_ewm3 are ALSO already columns
    on the separate `usage` frame (USAGE_EXPORT_COLUMNS), and
    assemble_player_advanced_stats merges both onto the same row. Returning
    them under the same name would collide on that merge and silently
    become target_share_ewm3_x/_y instead of raising -- renamed here so
    there's nothing to collide with, rather than relying on callers to get
    the merge order right forever.
    """
    from src.usage import TREND_SOURCE_FEATURES

    test_mask = (combined_features["season"] == target_season) & (combined_features["week"] == target_week)
    cols = (
        ["player_id"]
        + [f"{f}_ewm3" for f in TREND_SOURCE_FEATURES]
        + [f"{f}_trend_signal" for f in TREND_SOURCE_FEATURES]
        + [f"{f}_trend_direction" for f in TREND_SOURCE_FEATURES]
    )
    out = combined_features.loc[test_mask, cols].reset_index(drop=True)
    return out.rename(columns={f"{f}_ewm3": f"trend_{f}_current" for f in TREND_SOURCE_FEATURES})


def build_radar_snapshot(
    combined_features: pd.DataFrame,
    target_season: int,
    target_week: int,
    roster_positions: list[str],
    n_teams: int,
) -> dict:
    """
    Phase 4: one radar entry per candidate player_id (gsis_id-keyed, same
    convention as every other pre-crosswalk builder in this module).

    Eligibility gate: the player's OWN `games_played` (point-in-time-safe,
    from add_rolling_features -- see its docstring) must be >=
    RADAR_MIN_GAMES, reusing Phase 3's already-validated MIN_GAMES_FOR_TREND
    floor rather than a second, separately-justified number (same reasoning
    Family 7 already used for reusing EWM_HALFLIFE). This is a single,
    whole-player gate, not per-axis: a radar with some axes plotted and
    others silently missing would draw a misleading shape on the chart, so
    a player short on games gets an honest "not enough games yet" object
    instead of a partial one.

    Percentile pool: for each position, the top position_starter_counts()
    players (this league's real roster_positions/n_teams) by season-to-date
    total custom_points among players who ALSO clear RADAR_MIN_GAMES --
    "startable" means both "ranks near the top of the position" and "has
    enough of a sample to rank honestly." Every candidate at the position
    (pool member or not) gets percentiled against this same pool, so a
    thin-sample or bench player's radar still reads as "where would this
    profile rank among this league's actual starters," which is the whole
    point of restricting the pool -- a bench-inclusive denominator would
    compress every real contributor into the top of the range.

    Percentile convention: scipy's percentileofscore(kind='mean') -- a
    tied raw value gets credit for half its tied group rather than being
    arbitrarily broken toward one side.

    Returns: {player_id: radar_dict}, where radar_dict is either
      {"eligible": False, "games_played": int, "min_games": RADAR_MIN_GAMES}
    or
      {"eligible": True, "games_played": int, "min_games": RADAR_MIN_GAMES,
       "pool_size": int,
       "axes": [{"label": str, "unit": str, "percentile": int, "raw": float}, ...]}
    (raw/percentile null on axes where the pool itself has no data for that
    column -- doesn't happen for this pipeline's current RADAR_METRICS,
    since every listed column is populated for every position it's
    assigned to, but handled rather than assumed).
    """
    from scipy.stats import percentileofscore

    from src.usage import MIN_GAMES_FOR_TREND as RADAR_MIN_GAMES

    target_mask = (combined_features["season"] == target_season) & (combined_features["week"] == target_week)
    all_axis_cols = sorted({axis["column"] for axes in RADAR_METRICS.values() for axis in axes})
    target_rows = combined_features.loc[
        target_mask, ["player_id", "position", "games_played"] + all_axis_cols
    ].reset_index(drop=True)

    history_mask = (combined_features["season"] == target_season) & (combined_features["week"] < target_week)
    season_points = (
        combined_features.loc[history_mask].groupby("player_id")["custom_points"].sum()
    )

    starter_counts = position_starter_counts(roster_positions, n_teams)

    out: dict = {}
    for position, axes in RADAR_METRICS.items():
        pos_rows = target_rows[target_rows["position"] == position]
        if pos_rows.empty:
            continue

        eligible_ids = pos_rows.loc[pos_rows["games_played"] >= RADAR_MIN_GAMES, "player_id"]
        pool_points = season_points.reindex(eligible_ids).dropna().sort_values(ascending=False)
        pool_ids = set(pool_points.head(starter_counts[position]).index)
        pool_rows = pos_rows[pos_rows["player_id"].isin(pool_ids)]
        pool_size = len(pool_rows)

        for _, row in pos_rows.iterrows():
            games_played = int(row["games_played"])
            if games_played < RADAR_MIN_GAMES or pool_size == 0:
                out[row["player_id"]] = {
                    "eligible": False, "games_played": games_played, "min_games": RADAR_MIN_GAMES,
                }
                continue

            axis_out = []
            for axis in axes:
                col = axis["column"]
                raw = row[col]
                pool_vals = pool_rows[col].dropna()
                if pd.isna(raw) or pool_vals.empty:
                    pct, raw_display = None, None
                else:
                    pct = round(percentileofscore(pool_vals, raw, kind="mean"))
                    raw_display = round(float(raw) * 100 if axis["pct"] else float(raw), 2)
                axis_out.append({
                    "label": axis["label"], "unit": axis["unit"], "percentile": pct, "raw": raw_display,
                })

            out[row["player_id"]] = {
                "eligible": True, "games_played": games_played, "min_games": RADAR_MIN_GAMES,
                "pool_size": pool_size, "axes": axis_out,
            }

    return out


# A display judgment call, not empirically derived the way MIN_GAMES_FOR_TREND
# was (see Phase 3' findings) -- 1-2 plays in a zone could be one broken
# play, one scramble drill, one garbage-time snap, not a real tendency.
# Below this, a zone is flagged `sparse` rather than dropped or merged: the
# play genuinely happened, so hiding it would understate real (if noisy)
# usage, but rendering it at full visual weight would overstate confidence
# in a 1-2-play sample. The UI's job is to show it, just not as solidly.
HEATMAP_SPARSE_THRESHOLD = 3


def build_heatmap_snapshot(
    combined_features: pd.DataFrame,
    pbp: pd.DataFrame,
    target_season: int,
    target_week: int,
) -> dict:
    """
    Phase 5: one heatmap entry per candidate player_id (gsis_id-keyed,
    same convention as build_radar_snapshot). Same eligibility gate as
    radar -- reuses the player's own `games_played` from combined_features
    (point-in-time-safe, from add_rolling_features) against
    MIN_GAMES_FOR_TREND, a single whole-player gate: a heatmap with some
    zones plotted from a real sample and others from a 1-play fluke would
    misrepresent the player's actual usage pattern, so a short-on-games
    player gets the same honest ineligible object as radar, not a partial
    picture.

    `groups` is one entry per HEATMAP_POSITION_KINDS kind for this
    player's position (QB: passing only; RB: rushing AND receiving,
    matching getHeatmapTitle()'s "Rushing Direction & Receiving"; WR/TE:
    receiving only) -- shares are computed WITHIN each kind (a QB's
    passing shares sum to 1.0 on their own; an RB's rushing shares and
    receiving shares are two separate 1.0s), never combined into one pool,
    since a target and a carry don't share a denominator any more than
    Family 4's rz_target_share/rz_carry_share do (see PROJECT_CONTEXT.md's
    design decisions).

    `pbp` is scoped to target_season's weeks strictly before target_week
    (already-played weeks only -- pbp for an unplayed week doesn't exist
    yet regardless, this is a defensive belt-and-suspenders filter, same
    reasoning as every other point-in-time-safe cutoff in this pipeline).

    Returns: {player_id: heatmap_dict}, where heatmap_dict is either
      {"eligible": False, "games_played": int, "min_games": MIN_GAMES_FOR_TREND}
    or
      {"eligible": True, "games_played": int, "min_games": MIN_GAMES_FOR_TREND,
       "sparse_threshold": HEATMAP_SPARSE_THRESHOLD,
       "groups": [{"kind": str, "total_plays": int,
                   "zones": [{"id", "label", "count", "share", "sparse"}, ...]}, ...]}
    (sparse_threshold rides along per-player, redundant but simple, so
    index.html's "~ marks a zone with fewer than N plays" copy reads N
    from the export instead of a second, hardcoded copy of the same
    number that could quietly drift from this constant.)
    """
    from src.usage import (
        HEATMAP_POSITION_KINDS, HEATMAP_ZONE_LABELS, MIN_GAMES_FOR_TREND, passing_zone_plays,
        receiving_zone_plays, rushing_zone_plays,
    )

    target_mask = (combined_features["season"] == target_season) & (combined_features["week"] == target_week)
    target_rows = combined_features.loc[target_mask, ["player_id", "position", "games_played"]]

    # `pbp` can legitimately arrive completely empty (zero columns, not
    # just zero rows) -- weekly_update.py passes that when the current
    # season hasn't started yet and get_pbp itself can't be called (see
    # its own comment for why). Guarded here, before touching any pbp
    # column, rather than assumed to always have the real nflverse schema
    # -- and skipped rather than passed into the zone functions, since a
    # season with zero real pbp can only mean games_played is 0 for every
    # candidate anyway (5+ games played is impossible without pbp for
    # those games existing), so every candidate is ineligible below
    # regardless of what the zone functions would have returned.
    season_pbp = pbp[(pbp["season"] == target_season) & (pbp["week"] < target_week)] if not pbp.empty else pbp
    if season_pbp.empty:
        plays_by_kind = {kind: pd.DataFrame(columns=["player_id", "zone_a", "zone_b"]) for kind in ("receiving", "passing", "rushing")}
    else:
        plays_by_kind = {
            "receiving": receiving_zone_plays(season_pbp),
            "passing": passing_zone_plays(season_pbp),
            "rushing": rushing_zone_plays(season_pbp),
        }
    # Pre-grouped ONCE per kind (not per player) -- looking up a player's
    # zone counts is then a single groupby-result lookup, not an O(plays)
    # scan repeated for every candidate.
    counts_by_kind = {
        kind: {pid: g.groupby(["zone_a", "zone_b"]).size() for pid, g in frame.groupby("player_id")}
        for kind, frame in plays_by_kind.items()
    }

    out: dict = {}
    for _, row in target_rows.iterrows():
        kinds = HEATMAP_POSITION_KINDS.get(row["position"])
        if kinds is None:
            continue
        player_id = row["player_id"]
        games_played = int(row["games_played"])

        if games_played < MIN_GAMES_FOR_TREND:
            out[player_id] = {
                "eligible": False, "games_played": games_played, "min_games": MIN_GAMES_FOR_TREND,
            }
            continue

        groups = []
        for kind in kinds:
            counts = counts_by_kind[kind].get(player_id)
            if counts is None or counts.empty:
                continue
            total = int(counts.sum())
            zones = [
                {
                    "id": f"{zone_a}|{zone_b}",
                    "label": f"{HEATMAP_ZONE_LABELS[zone_a]} · {HEATMAP_ZONE_LABELS[zone_b]}",
                    "count": int(count),
                    "share": round(int(count) / total, 3),
                    "sparse": count < HEATMAP_SPARSE_THRESHOLD,
                }
                for (zone_a, zone_b), count in counts.sort_values(ascending=False).items()
            ]
            groups.append({"kind": kind, "total_plays": total, "zones": zones})

        out[player_id] = {
            "eligible": True, "games_played": games_played, "min_games": MIN_GAMES_FOR_TREND,
            "sparse_threshold": HEATMAP_SPARSE_THRESHOLD, "groups": groups,
        }

    return out


def build_xfp_summary(historical_features: pd.DataFrame, xfp_season: int) -> pd.DataFrame:
    """
    Real, completed-season xFP vs. actual custom_points, summed over
    xfp_season's REG weeks -- the most recently COMPLETED season, since the
    target (upcoming) season has no games played yet to summarize. This is
    intentionally the ACTUAL per-week xfp values (not the lagged _ewm3/_s2d
    rolling versions used as model inputs), since it's a season retrospective
    for a human reader, not a model feature.
    """
    season_rows = historical_features[historical_features["season"] == xfp_season]
    summary = (
        season_rows.groupby("player_id")[["xfp", "custom_points"]]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"xfp": "season_xfp", "custom_points": "season_actual"})
    )
    summary["fp_over_expected"] = summary["season_actual"] - summary["season_xfp"]
    return summary


def build_weekly_xfp(historical_features: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """
    Per-player, per-week REAL xfp for target_season's played weeks so far
    (Phase 8 round 3) -- src/usage.py::add_xfp_features's per-play-derived
    value for that exact week, not the lagged xfp_ewm3/xfp_s2d already in
    the `usage` block. Powers the Weekly Production chart's optional xFP
    line: this is the one field in this export that's a per-week time
    series rather than a single upcoming-week snapshot, because a chart
    line needs one value per already-played week, not one number
    summarizing the whole season (build_xfp_summary) or the model's own
    upcoming-week prediction (predict_target_week*).

    Deliberately scoped to target_season, not the full multi-season
    historical_features table -- the Weekly Production chart only ever
    shows ONE season's bars at a time, and a stray prior-season week
    would just be dead data the chart never reads.

    Rows with a null xfp (QB, always -- xfp has no passing counterpart;
    or any other player-week xfp itself came back null for) are dropped
    here, not zero-filled, matching every other null-means-absent
    convention in this export.

    Returns: player_id, week, xfp -- one row per (player, played week).
    Can be empty (e.g. the target season has no games played yet).
    """
    season_rows = historical_features[historical_features["season"] == target_season]
    return season_rows.dropna(subset=["xfp"])[["player_id", "week", "xfp"]].reset_index(drop=True)


# ==========================================================================
# STEP 4: scope -- real 2026 rosters, union top ~300 by point projection
# ==========================================================================
def get_export_scope(
    rostered_gsis_ids: set, predictions: pd.DataFrame, top_n: int = 300
) -> tuple[pd.DataFrame, dict]:
    """
    Scope = every rostered player with a prediction, union the top_n
    REMAINING players by `point`. Rostered players are never displaced by
    the top_n cutoff -- a manager's own roster should never silently vanish
    from their own dashboard because a free agent projects higher.

    Returns (scoped predictions, {"n_rostered", "n_top_n", "n_total"}).
    """
    is_rostered = predictions["player_id"].isin(rostered_gsis_ids)
    rostered = predictions[is_rostered]
    remaining = predictions[~is_rostered].sort_values("point", ascending=False).head(top_n)
    scoped = pd.concat([rostered, remaining], ignore_index=True)
    report = {"n_rostered": len(rostered), "n_top_n": len(remaining), "n_total": len(scoped)}
    return scoped, report


# ==========================================================================
# STEP 5: assemble the JSON payload
# ==========================================================================
def assemble_player_advanced_stats(
    scoped_predictions: pd.DataFrame,
    usage: pd.DataFrame,
    trend: pd.DataFrame,
    xfp_summary: pd.DataFrame,
    weekly_xfp: pd.DataFrame,
    radar: dict,
    heatmap: dict,
    crosswalk: pd.DataFrame,
    target_season: int,
    target_week: int,
    seasons_trained: list[int],
    model_version: str,
    performance: dict = PERFORMANCE_BY_POSITION,
    caveats: list = CAVEATS,
) -> tuple[dict, dict]:
    """
    Joins everything onto scoped_predictions and crosswalks gsis_id ->
    sleeper_id AT THIS FINAL STEP, since Sleeper's ID is what the JSON is
    keyed by but every upstream computation is in gsis_id space.

    `performance` defaults to this module's PERFORMANCE_BY_POSITION
    constant (the one-time Phase 6 walk-forward result, for manual/notebook
    use) but Phase 8's weekly_update.py passes the LOADED ARTIFACT's own
    `performance` dict instead -- retrain.yml recomputes walk-forward MAE
    vs. baselines on every run (see scripts/retrain.py), so the exported
    JSON should report whatever the artifact that produced its predictions
    actually measured, not a number that could be stale by several retrains.

    `weekly_xfp` (build_weekly_xfp's output) is grouped into a
    {player_id: {week_str: xfp}} dict BEFORE the per-row loop below,
    rather than merged the way usage/trend/xfp_summary are -- those are
    already one row per player, but weekly_xfp is one row per (player,
    week), so a plain merge would multiply scoped_predictions' rows
    instead of adding a column to them.

    `radar` (build_radar_snapshot's output) is ALREADY a {player_id: dict}
    mapping, not a DataFrame -- its per-player value is a nested
    eligible/axes structure, not a flat set of columns a `.merge()` could
    add, same reasoning as weekly_xfp_by_player just built above. A
    candidate absent from `radar` (shouldn't happen -- build_radar_snapshot
    covers every RADAR_METRICS position for every target-week candidate,
    this is defensive, not expected) falls back to an honest "not
    eligible, 0 games" object rather than a KeyError.

    `heatmap` (build_heatmap_snapshot's output) is the same shape of
    {player_id: dict} mapping as `radar`, same fallback reasoning.

    Returns (payload, crosswalk_report) -- crosswalk_report has
    {"n_scoped", "n_matched", "match_rate"} so the match rate gets reported,
    not just assumed.
    """
    from src.usage import MIN_GAMES_FOR_TREND, TREND_SOURCE_FEATURES

    weekly_xfp_by_player = {}
    if not weekly_xfp.empty:
        for player_id, group in weekly_xfp.dropna(subset=["xfp"]).groupby("player_id"):
            weekly_xfp_by_player[player_id] = {
                str(int(w)): round(float(x), 2) for w, x in zip(group["week"], group["xfp"])
            }

    cw = crosswalk.dropna(subset=["gsis_id", "sleeper_id"]).drop_duplicates(subset=["gsis_id"])
    merged = scoped_predictions.merge(usage, on="player_id", how="left")
    merged = merged.merge(trend, on="player_id", how="left")
    merged = merged.merge(xfp_summary, on="player_id", how="left")
    n_scoped = len(merged)
    merged = merged.merge(cw[["gsis_id", "sleeper_id"]], left_on="player_id", right_on="gsis_id", how="inner")
    n_matched = len(merged)

    players = {}
    for _, row in merged.iterrows():
        players[row["sleeper_id"]] = {
            "projection": {
                "point": round(float(row["point"]), 2),
                "floor": round(float(row["floor"]), 2),
                "ceiling": round(float(row["ceiling"]), 2),
            },
            "usage": {
                col: (None if pd.isna(row[col]) else round(float(row[col]), 4))
                for col in USAGE_EXPORT_COLUMNS
            },
            "trend": {
                TREND_FEATURE_LABELS[feat]: {
                    "current": (
                        None if pd.isna(row[f"trend_{feat}_current"])
                        else round(float(row[f"trend_{feat}_current"]), 4)
                    ),
                    "signal": None if pd.isna(row[f"{feat}_trend_signal"]) else round(float(row[f"{feat}_trend_signal"]), 3),
                    "direction": None if pd.isna(row[f"{feat}_trend_direction"]) else row[f"{feat}_trend_direction"],
                }
                for feat in TREND_SOURCE_FEATURES
            },
            "xfp": {
                "season_xfp": None if pd.isna(row.get("season_xfp")) else round(float(row["season_xfp"]), 2),
                "season_actual": None if pd.isna(row.get("season_actual")) else round(float(row["season_actual"]), 2),
                "fp_over_expected": (
                    None if pd.isna(row.get("fp_over_expected")) else round(float(row["fp_over_expected"]), 2)
                ),
            },
            "weekly_xfp": weekly_xfp_by_player.get(row["player_id"], {}),
            "radar": radar.get(
                row["player_id"], {"eligible": False, "games_played": 0, "min_games": MIN_GAMES_FOR_TREND}
            ),
            "heatmap": heatmap.get(
                row["player_id"], {"eligible": False, "games_played": 0, "min_games": MIN_GAMES_FOR_TREND}
            ),
        }

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "season": target_season,
            "week": target_week,
            "model_version": model_version,
            "seasons_trained": seasons_trained,
            "performance": performance,
            "caveats": caveats,
        },
        "players": players,
    }
    crosswalk_report = {
        "n_scoped": n_scoped, "n_matched": n_matched,
        "match_rate": n_matched / n_scoped if n_scoped else float("nan"),
    }
    return payload, crosswalk_report


# ==========================================================================
# STEP 6: validation
# ==========================================================================
def validate_export(payload: dict, crosswalk: pd.DataFrame) -> dict:
    """
    Fails loudly (raises AssertionError) rather than returning a partial
    "mostly fine" report -- a bad export silently shipped to the dashboard is
    worse than a notebook cell that stops.

    Checks:
      - every top-level players[] key is a real Sleeper ID (present in the
        crosswalk's sleeper_id column)
      - projection.point/floor/ceiling are never null
      - floor <= point <= ceiling on every row
    """
    real_sleeper_ids = set(crosswalk["sleeper_id"].dropna())
    players = payload["players"]

    unknown_keys = [pid for pid in players if pid not in real_sleeper_ids]
    assert not unknown_keys, f"{len(unknown_keys)} player keys are not real Sleeper IDs: {unknown_keys[:5]}"

    null_projection = [
        pid for pid, p in players.items()
        if p["projection"]["point"] is None or p["projection"]["floor"] is None or p["projection"]["ceiling"] is None
    ]
    assert not null_projection, f"{len(null_projection)} players have a null projection field: {null_projection[:5]}"

    out_of_order = [
        pid for pid, p in players.items()
        if not (p["projection"]["floor"] <= p["projection"]["point"] <= p["projection"]["ceiling"])
    ]
    assert not out_of_order, f"{len(out_of_order)} players violate floor <= point <= ceiling: {out_of_order[:5]}"

    bad_radar_percentiles = [
        pid for pid, p in players.items()
        if p["radar"]["eligible"]
        and any(ax["percentile"] is not None and not (0 <= ax["percentile"] <= 100) for ax in p["radar"]["axes"])
    ]
    assert not bad_radar_percentiles, (
        f"{len(bad_radar_percentiles)} players have an out-of-range radar percentile: {bad_radar_percentiles[:5]}"
    )

    bad_heatmap_shares = [
        pid for pid, p in players.items()
        if p["heatmap"]["eligible"]
        and any(
            abs(sum(z["share"] for z in g["zones"]) - 1.0) > 0.01 or any(z["count"] < 0 for z in g["zones"])
            for g in p["heatmap"]["groups"]
        )
    ]
    assert not bad_heatmap_shares, (
        f"{len(bad_heatmap_shares)} players have a heatmap group whose zone shares don't sum to ~1.0 "
        f"or contain a negative count: {bad_heatmap_shares[:5]}"
    )

    return {
        "n_players": len(players),
        "all_keys_real_sleeper_ids": True,
        "no_null_projections": True,
        "floor_le_point_le_ceiling": True,
        "radar_percentiles_in_range": True,
        "heatmap_shares_sum_to_one": True,
    }


# ==========================================================================
# STEP 7: simulation wiring (Phase 8 round 2) -- src/simulate.py was
# computed but never written anywhere; this is that wiring.
# ==========================================================================
# Validated on 204 real historical matchups (win probability) and 8
# (season, snapshot-week) combinations across 2 completed seasons (playoff
# odds) -- 112 nominal team-observations, but the 4 snapshots per season
# follow the SAME 14 teams through ONE realized season, so the true
# independent sample is closer to 2 than 112. See PROJECT_CONTEXT.md's
# Phase 6.5 findings for the full calibration report. Enough to catch a
# broken simulator (garbage in, garbage probabilities out would show up
# immediately), not enough to certify these exact numbers -- every UI
# surface that shows a probability from this block must show this caveat
# alongside it, not bury it in a tooltip only.
SIMULATION_CALIBRATION_CAVEAT = (
    "Calibration was checked on ~2 independent seasons of historical matchups and "
    "playoff snapshots -- enough to catch a broken simulator, not enough to certify "
    "these exact numbers."
)
# The simulator's win-probability favorite matches a naive "higher projected
# total wins" rule in 191/204 (93.6%) of validated matchups -- BY
# CONSTRUCTION, not because the simulator out-picks the naive rule.
# Correlation/variance change a simulated total's SPREAD, not its MEAN, so
# the two methods can only possibly disagree on genuine toss-up games. What
# simulation adds over the naive rule is the probability itself (and
# playoff odds, which the naive rule has no equivalent of at all) -- not
# better picks. See PROJECT_CONTEXT.md's Phase 6.5 findings.
SIMULATION_ACCURACY_CAVEAT = (
    "This does not out-pick a simple 'higher projected total wins' rule -- it agrees "
    "with that rule 93.6% of the time by construction. What it adds is the probability "
    "itself, not better picks."
)
SIMULATION_N_SIMS_MATCHUP = 10_000  # matches Phase 6.5's own validated setting
SIMULATION_N_SIMS_SEASON = 3_000    # matches Phase 6.5's own validated setting


def build_team_game_id_lookup(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    (season, week, team) -> game_id, REG only, stacking each schedule row's
    home and away perspective. src/simulate.py's sample_player_week needs
    game_id for its game-environment correlation mechanism; nothing in
    src/usage.py's CONTEXT_OUTPUT_COLUMNS carries it through (that family
    strips the schedule down to spread/total/weather, not the raw game
    identifier) -- this is the one piece of schedule data this pipeline
    needed for the first time in Phase 8 round 2.
    """
    reg = schedule[schedule["game_type"] == "REG"]
    home = reg[["season", "week", "home_team", "game_id"]].rename(columns={"home_team": "team"})
    away = reg[["season", "week", "away_team", "game_id"]].rename(columns={"away_team": "team"})
    return pd.concat([home, away], ignore_index=True)


def build_starter_quantile_rows(
    starter_sleeper_ids: list,
    season: int,
    week: int,
    quantiles_by_gsis: pd.DataFrame,
    sleeper_proj_points: dict,
    sleeper_to_gsis: dict,
    team_by_sleeper: dict,
    game_id_by_team_week: dict,
) -> pd.DataFrame:
    """
    One roster's real starters (Sleeper player_ids, K/DST included) -> the
    game_id + 5-quantile-column frame src/simulate.py's
    sample_player_week/simulate_matchup/simulate_season all need.

    Per starter:
      - model-covered (QB/RB/WR/TE with a row in quantiles_by_gsis, found
        via the crosswalk): that player's own 5 CQR-calibrated quantile
        points (predict_quantiles_with_models's output for this exact
        week -- callers predicting multiple weeks, e.g. simulate_season's
        remaining schedule, must pass THAT week's own quantiles_by_gsis,
        not one week's reused for every week).
      - everyone else (K/DST, or a skill player missing model coverage --
        e.g. not in this week's candidate pool): src/simulate.py's own
        documented convention -- all 5 quantile columns set to Sleeper's
        OWN point projection for that player, so interpolating always
        returns that constant. Falls back to 0.0 (not dropped) if Sleeper
        has no projection either -- the same "unprojected means 0, not
        missing" convention index.html's own resolvePts(...) ?? 0 uses.

    game_id comes from the starter's CURRENT team (Sleeper's own team
    field, normalized to nflverse's code) joined against
    game_id_by_team_week for (season, week). A starter on a bye that week,
    or with no resolvable team, is dropped -- there's no real game to
    attach them to, so nothing to simulate for that one slot. This makes
    a simulated roster total a slight underestimate on a bye week for one
    of its players, same direction of bias as any other "can't score
    without a game" convention.

    Returns: game_id, pred_q10_cqr, pred_q25_cqr, pred_q50, pred_q75_cqr,
    pred_q90_cqr -- one row per resolvable starter. Can be empty (e.g.
    every starter unresolvable) -- callers must treat that as "nothing to
    simulate for this roster," not fail.
    """
    quantiles_indexed = (
        quantiles_by_gsis.drop_duplicates(subset=["player_id"]).set_index("player_id")
        if not quantiles_by_gsis.empty else quantiles_by_gsis
    )
    quantile_cols = ["pred_q10_cqr", "pred_q25_cqr", "pred_q50", "pred_q75_cqr", "pred_q90_cqr"]

    rows = []
    for sleeper_id in starter_sleeper_ids or []:
        if not sleeper_id or sleeper_id == "0":
            continue  # Sleeper uses "0" to mark an empty lineup slot
        team = normalize_team_code(team_by_sleeper.get(sleeper_id) or "")
        game_id = game_id_by_team_week.get((season, week, team))
        if game_id is None:
            continue

        gsis_id = sleeper_to_gsis.get(sleeper_id)
        if gsis_id is not None and not quantiles_by_gsis.empty and gsis_id in quantiles_indexed.index:
            q = quantiles_indexed.loc[gsis_id]
            row = {"game_id": game_id, **{c: float(q[c]) for c in quantile_cols}}
        else:
            constant = float(sleeper_proj_points.get(sleeper_id, 0.0))
            row = {"game_id": game_id, **{c: constant for c in quantile_cols}}
        rows.append(row)

    return pd.DataFrame(rows, columns=["game_id"] + quantile_cols)


def build_matchup_simulation(matchups: pd.DataFrame, lineup_fn, n_sims: int = SIMULATION_N_SIMS_MATCHUP) -> list:
    """
    Per-matchup win probability for one week, via src.simulate.simulate_matchup.

    Args:
        matchups: Sleeper's own matchups frame for the target week
            (get_sleeper_matchups's output) -- roster_id, matchup_id.
        lineup_fn: callable roster_id -> DataFrame (build_starter_quantile_rows's
            output) for THIS week. Kept as an injected callback, same
            reasoning as simulate_season's own lineup_builder parameter.
        n_sims: passed through to simulate_matchup.

    Skips (not fails) a matchup that can't be simulated -- an odd-sized
    group (shouldn't happen for a real fantasy week, but malformed data
    shouldn't abort the whole export) or a lineup with zero resolvable
    starters -- rather than raising over one team's data gap.

    Win probabilities are rounded to WHOLE percent -- see
    SIMULATION_CALIBRATION_CAVEAT's own precision note; a decimal implies
    precision this validation doesn't support.

    Returns: list of {matchup_id, roster_a, roster_b, win_prob_a,
    win_prob_b, n_sims}, one entry per simulatable matchup.
    """
    results = []
    for matchup_id, group in matchups.groupby("matchup_id"):
        roster_ids = group["roster_id"].tolist()
        if len(roster_ids) != 2:
            continue
        roster_a, roster_b = roster_ids
        lineup_a, lineup_b = lineup_fn(roster_a), lineup_fn(roster_b)
        if lineup_a.empty or lineup_b.empty:
            continue
        sim = simulate_matchup(lineup_a, lineup_b, n_sims=n_sims)
        results.append({
            "matchup_id": int(matchup_id),
            "roster_a": int(roster_a),
            "roster_b": int(roster_b),
            "win_prob_a": round(sim["team_a_win_prob"] * 100),
            "win_prob_b": round(sim["team_b_win_prob"] * 100),
            "n_sims": n_sims,
        })
    return results


def build_playoff_odds(
    remaining_weeks: list,
    starting_standings: pd.DataFrame,
    lineup_builder,
    playoff_teams: int,
    n_sims: int = SIMULATION_N_SIMS_SEASON,
) -> dict:
    """
    Per-roster playoff-qualification odds via src.simulate.simulate_season,
    rounded to whole percent (see build_matchup_simulation's docstring for
    why). Thin wrapper -- simulate_season already does the real work; this
    just reshapes its output into the {roster_id: pct} JSON-key shape the
    export needs (JSON object keys are always strings, so roster_id is
    stringified here, not left as simulate_season's own int).

    Returns {} if starting_standings is empty (nothing to simulate) --
    callers should treat an empty dict as "no odds available," same as
    build_matchup_simulation returning an empty list.
    """
    if starting_standings.empty:
        return {}
    result_df = simulate_season(remaining_weeks, starting_standings, lineup_builder, playoff_teams, n_sims=n_sims)
    return {str(int(row.roster_id)): round(row.playoff_prob * 100) for row in result_df.itertuples()}


def assemble_simulation_block(matchups: list, playoff_odds: dict, week: int) -> dict | None:
    """
    Packages build_matchup_simulation/build_playoff_odds's output into the
    payload's top-level `simulation` key (a sibling of `meta`/`players`,
    not nested under either -- these are per-MATCHUP and per-ROSTER
    numbers, not per-player, so they don't fit the players[] schema).

    Returns None (not an empty dict) when there is truly nothing to show --
    no matchups AND no playoff odds, e.g. the 2026 pre-draft league before
    a real schedule exists -- so the dashboard's null-safe path is a single
    `if (!advanced.simulation)` check, not "is every sub-field also empty."
    """
    if not matchups and not playoff_odds:
        return None
    return {
        "week": week,
        "matchups": matchups,
        "playoff_odds": playoff_odds,
        "matchup_n_sims": SIMULATION_N_SIMS_MATCHUP,
        "season_n_sims": SIMULATION_N_SIMS_SEASON,
        "calibration_caveat": SIMULATION_CALIBRATION_CAVEAT,
        "accuracy_caveat": SIMULATION_ACCURACY_CAVEAT,
    }


def validate_simulation(simulation: dict | None) -> dict:
    """
    Lightweight sanity check for the simulation block -- deliberately
    looser than validate_export's hard per-player assertions, since this
    block is allowed to be partially populated (matchups without playoff
    odds once the regular season ends, or vice versa is never expected but
    isn't fatal either) or entirely None. Still fails loudly on genuinely
    nonsensical output -- a probability outside [0, 100], or a matchup's
    two win probabilities not summing to ~100 -- since that would mean the
    simulation math itself is broken, not just sparse.
    """
    if simulation is None:
        return {"present": False}

    for m in simulation["matchups"]:
        total = m["win_prob_a"] + m["win_prob_b"]
        assert 0 <= m["win_prob_a"] <= 100 and 0 <= m["win_prob_b"] <= 100, (
            f"matchup {m['matchup_id']}: win probabilities out of [0, 100] range: {m}"
        )
        # Whole-percent rounding of two complementary probabilities can
        # legitimately land one point off 100 (e.g. 50/50 rounds fine, but
        # 33.4/66.6 rounds to 33/67 = 100 while 50.5/49.5 rounds to 51/50 =
        # 101) -- allow a small tolerance rather than demanding exact 100.
        assert 99 <= total <= 101, f"matchup {m['matchup_id']}: win probabilities sum to {total}, not ~100: {m}"

    for roster_id, pct in simulation["playoff_odds"].items():
        assert 0 <= pct <= 100, f"roster {roster_id}: playoff_prob {pct} out of [0, 100] range"

    return {
        "present": True,
        "n_matchups": len(simulation["matchups"]),
        "n_rosters_with_playoff_odds": len(simulation["playoff_odds"]),
    }
