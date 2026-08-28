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

from src.model import (
    ALL_FEATURE_COLUMNS, FEATURE_COLUMNS_BY_POSITION, _cast_categoricals, predict_quantiles_with_models,
    predict_with_models, train_final_models,
)
from src.simulate import player_point_in_time_metrics, sample_player_week, simulate_matchup, simulate_season

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
    "Floor/ceiling intervals are conformally calibrated; ceilings run conservative, "
    "so boom probabilities (Monte Carlo) read a bit low for the same reason.",
    "The red_zone_share trend signal is real but noisier than snap/target/carry "
    "share -- red-zone opportunities are low-volume, and a 'rising' read reverts "
    "more often than it holds even at the validated window and threshold.",
    "Team Tendencies describes a team's offense (pass rate over expected, pace, "
    "red-zone play-calling, target distribution), not any one player's usage.",
]

# Surfaced both in meta.caveats (above, terse) and as its own meta field
# (fuller, for the UI to show inline right next to boom/bust/start-over-
# replacement numbers -- same "surface it where the numbers appear" pattern
# SIMULATION_CALIBRATION_CAVEAT/SIMULATION_ACCURACY_CAVEAT below already use
# for the matchup-simulation block). Numbers are PROJECT_CONTEXT.md's own
# Phase 6 CQR findings, not re-derived here: 10-90 interval coverage after
# CQR lands at 82.6-86.0% against an 80% target (QB/RB/WR/TE respectively),
# and the correction overshoots the ceiling more than the floor because the
# floor was the worse-calibrated side before correction -- widening it by
# the amount needed pulled the ceiling along with it. src/simulate.py's own
# module docstring documents the same asymmetry for anyone sampling from
# these quantiles directly.
MONTE_CARLO_CALIBRATION_CAVEAT = (
    "These probabilities come from the same CQR-calibrated quantiles as the floor/ceiling "
    "projection. Interval coverage runs 82-86% against an 80% target, and the correction is "
    "asymmetric: the floor was the worse-calibrated side before correction, so widening it by "
    "the needed amount pulled the ceiling along with it, leaving ceilings more conservative "
    "than floors. Boom rates (the upper tail) read a bit low as a result -- treat them as a "
    "lower bound, not an exact probability."
)

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


def get_archive_candidates(historical_features: pd.DataFrame, crosswalk: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Candidate pool for a SEASON ARCHIVE (scripts/archive_season.py), not a
    live upcoming week -- see get_export_candidates for that path, which
    this deliberately does NOT reuse unmodified.

    The one real difference: `team` comes from this player's own most
    recent REAL row in historical_features (whatever team they were on
    the last time they actually played), not from Sleeper's CURRENT
    snapshot the way get_export_candidates resolves it via `sleeper_players`.
    That distinction matters enormously here and barely at all on the live
    path: a live "predict the upcoming week" candidate genuinely needs a
    CURRENT team (there's no game to attach pregame context to for a free
    agent with no team today), but an archive candidate's `team` is only
    ever used to build a stub row for a week that's already over --
    add_context_features finds no real schedule entry for a week that
    never happened regardless of which team is filled in, so the value
    only needs to be non-null, not current.

    Using get_export_candidates' current-team filter here would silently
    drop every retired/moved-on player from a historical archive --
    checked directly against this league's real rosters before choosing
    this fix, not assumed: it would have excluded 47-89% of a season's
    REAL rostered players depending on how long ago the season was (2021:
    53% dropped; 2025: 11% dropped even for the most recent completed
    season), which would defeat an archive's entire purpose.

    Returns: (candidates, report), same shape as get_export_candidates.
    `n_with_current_team` is kept as the same report key for schema
    consistency, though here it's always identical to n_crosswalk_matched
    by construction -- a real historical row always has a real team, so
    once a player clears the crosswalk match there's nothing left to drop.
    """
    position_eligible = historical_features[
        historical_features["position"].isin(EXPORT_POSITIONS)
    ][["player_id", "position", "season", "week", "team"]]

    last_known_team = (
        position_eligible.sort_values(["player_id", "season", "week"])
        .drop_duplicates(subset=["player_id"], keep="last")[["player_id", "position", "team"]]
    )
    last_known_team["team"] = last_known_team["team"].map(normalize_team_code)
    n_position_eligible = len(last_known_team)

    cw = crosswalk.dropna(subset=["gsis_id", "sleeper_id"]).drop_duplicates(subset=["gsis_id"])
    merged = last_known_team.merge(cw[["gsis_id", "sleeper_id"]], left_on="player_id", right_on="gsis_id", how="left")
    n_crosswalk_matched = int(merged["sleeper_id"].notna().sum())

    candidates = merged.dropna(subset=["sleeper_id"])[["player_id", "position", "team"]].reset_index(drop=True)
    report = {
        "n_position_eligible": n_position_eligible,
        "n_crosswalk_matched": n_crosswalk_matched,
        "n_with_current_team": n_crosswalk_matched,
        "crosswalk_match_rate": n_crosswalk_matched / n_position_eligible if n_position_eligible else float("nan"),
    }
    return candidates, report


def get_season_team_map(historical_features: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Team-per-season resolution for a SEASON ARCHIVE's export -- distinct
    from get_archive_candidates' `team` column just above. That one is
    deliberately LAST-EVER (see its own docstring): built only to give
    build_target_week_features a non-null stub value, where current-ness
    never matters. This answers a different question the Draft Prep view
    actually needs: what team was this player ON DURING `season`
    specifically. For any archived season that isn't the most recent one
    in historical_features, "last-ever" would silently resolve to a LATER
    team instead of that season's real one.

    Resolved from the player's last real row WITHIN `season` (their
    end-of-season team, so a single in-season trade doesn't produce two
    conflicting answers).

    Returns one row per player_id: [player_id, season_team]. A player with
    zero real rows in `season` simply isn't a key -- callers should treat
    a missing player_id as "no season team on file," not assume one.
    """
    season_rows = historical_features[historical_features["season"] == season][["player_id", "week", "team"]]
    resolved = (
        season_rows.sort_values(["player_id", "week"])
        .drop_duplicates(subset=["player_id"], keep="last")[["player_id", "team"]]
        .rename(columns={"team": "season_team"})
        .reset_index(drop=True)
    )
    resolved["season_team"] = resolved["season_team"].map(normalize_team_code)
    return resolved


def build_target_week_features(
    historical_features: pd.DataFrame,
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    target_season: int,
    target_week: int,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns the COMBINED frame (real history + one stub row per candidate
    for target_season/target_week), with Family 5 context and Family 6
    rolling features recomputed over the whole thing -- see the module
    docstring for why this reuses add_context_features/add_rolling_features
    unmodified rather than writing new future-facing feature logic.

    `pbp` feeds Team Tendencies (src/team_tendencies.py, a QB model
    feature -- see src/model.py::FEATURE_COLUMNS_BY_POSITION), the one
    family here that can't be derived from historical_features' own
    already-computed columns the way opponent strength reuses `xfp` --
    it needs real play-by-play directly. Scoping this to just the target
    season's own pbp is sufficient (team tendencies resets every season,
    same as opponent strength -- see add_team_tendency_features's own
    docstring) even though historical_features itself may span multiple
    seasons. Callers pass an EMPTY DataFrame (not None) when pbp can't be
    fetched yet (season hasn't started -- see build_heatmap_snapshot's own
    comment for the identical condition); add_team_tendency_features
    guards that itself.

    Categorical columns (roof, surface) are cast ONCE here, on the combined
    frame, so the later train/predict split always shares the same category
    set -- the same reason src/model.py's walk_forward_predict casts before
    splitting into folds, not after.
    """
    from src.team_tendencies import add_team_tendency_features
    from src.usage import (
        ROLLING_SOURCE_COLUMNS, add_context_features, add_opponent_strength_features,
        add_rolling_features, add_trend_features,
    )

    stub = candidates.copy()
    stub["season"] = target_season
    stub["week"] = target_week
    for col in ["offense_pct"] + list(ROLLING_SOURCE_COLUMNS):
        stub[col] = np.nan

    combined = pd.concat([historical_features, stub], ignore_index=True, sort=False)
    combined = add_context_features(combined, schedule)
    # Needs xfp -- already real for every historical row and NaN on the
    # stub row (from the ROLLING_SOURCE_COLUMNS fill above, xfp included) --
    # same reasoning as add_context_features just above: the stub row's own
    # (team, target_week) contributes nothing to the team-position-week
    # aggregate build_defense_strength_table computes, but the opponent's
    # REAL prior weeks already in historical_features are exactly what the
    # stub row's own opp_def_xfp_allowed_* values end up reflecting.
    combined = add_opponent_strength_features(combined, schedule)
    combined = add_rolling_features(combined)
    # Phase 3' trend signal -- downstream of the model's own feature set
    # (add_trend_features's outputs are never added to FEATURE_COLUMNS), so
    # running it here doesn't change what predict_target_week trains on.
    combined = add_trend_features(combined)
    combined = add_team_tendency_features(combined, pbp)
    combined = _cast_categoricals(combined, ALL_FEATURE_COLUMNS)
    return combined


# ==========================================================================
# STEP 2: train final (no-holdout) models and predict the target week
# ==========================================================================
def predict_target_week(
    combined_features: pd.DataFrame,
    target_season: int,
    target_week: int,
    feature_cols: dict[str, list[str]] = FEATURE_COLUMNS_BY_POSITION,
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
    module's FEATURE_COLUMNS_BY_POSITION/CQR_WIDEN_BY_10_90 constants) --
    the artifact is self-describing so a weekly run always matches
    whatever retrain.yml actually trained, even if this module's
    constants are edited later.

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


def build_matchup_snapshot(combined_features: pd.DataFrame, target_season: int, target_week: int) -> pd.DataFrame:
    """
    Family 5B (opponent defensive strength): the target week's `opponent`
    and OPPONENT_STRENGTH_OUTPUT_COLUMNS, one row per player -- this IS
    the model's own input for RB/WR/TE, not a separate computation, same
    "this IS the model's own input" pattern as build_usage_snapshot.
    `opponent` itself is populated for QB too (it comes from the real
    schedule, not from xfp) even though the four strength columns are
    null for QB -- see add_opponent_strength_features's own scope-gap
    docstring note (no expected-passing-points model exists to rate a
    defense against opposing QBs).
    """
    from src.usage import OPPONENT_STRENGTH_OUTPUT_COLUMNS

    test_mask = (combined_features["season"] == target_season) & (combined_features["week"] == target_week)
    cols = ["player_id", "opponent"] + OPPONENT_STRENGTH_OUTPUT_COLUMNS
    return combined_features.loc[test_mask, cols].reset_index(drop=True)


def build_defense_rankings(
    combined_features: pd.DataFrame, schedule: pd.DataFrame, target_season: int, target_week: int
) -> dict:
    """
    Every team's CURRENT (as of target_week -- i.e. reflecting only real
    games strictly before target_week) opponent-adjusted xFP allowed, per
    OPP_STRENGTH_POSITIONS position, ranked most-favorable-to-face first
    (highest adj_s2d = allows the most = best matchup for an offensive
    player at that position). Powers the dashboard's standalone "which
    defenses are favorable/unfavorable this week" panel; build_matchup_
    snapshot above covers the per-player indicator, this covers the
    league-wide view neither a per-player frame nor a per-player dict can
    show on its own.

    Ranked on allowed_adj_s2d (season-to-date, opponent-adjusted) rather
    than the recency-weighted ewm3 variant -- same "a season-characterizing
    read wants the more stable input" reasoning RADAR_METRICS's own choice
    of _s2d over _ewm3 already uses for the position-profile radar. Both
    raw and adjusted, both ewm3 and s2d, ride along per team so the UI can
    show the full picture, not just the one number used to sort.

    Returns: {position: [{"team", "raw_ewm3", "raw_s2d", "adj_ewm3",
    "adj_s2d", "rank"}, ...]}, sorted most-favorable-first. A team with
    fewer than the implicit games-played floor this season simply isn't
    in the list yet -- not enough history to rank it honestly, the same
    absence-means-no-data convention as everywhere else in this pipeline
    (not a fabricated rank).
    """
    from src.usage import OPP_STRENGTH_POSITIONS, build_defense_strength_table

    table = build_defense_strength_table(combined_features, schedule)
    week_rows = table[(table["season"] == target_season) & (table["week"] == target_week)]

    out: dict = {}
    for position in OPP_STRENGTH_POSITIONS:
        pos_rows = week_rows[
            (week_rows["position"] == position) & week_rows["allowed_adj_s2d"].notna()
        ].sort_values("allowed_adj_s2d", ascending=False)
        out[position] = [
            {
                "team": row["team"],
                "raw_ewm3": round(float(row["allowed_ewm3"]), 2),
                "raw_s2d": round(float(row["allowed_s2d"]), 2),
                "adj_ewm3": round(float(row["allowed_adj_ewm3"]), 2),
                "adj_s2d": round(float(row["allowed_adj_s2d"]), 2),
                "rank": rank,
            }
            for rank, (_, row) in enumerate(pos_rows.iterrows(), start=1)
        ]
    return out


def build_season_defense_rankings(historical_features: pd.DataFrame, schedule: pd.DataFrame, season: int) -> dict:
    """
    Season-ARCHIVE counterpart to build_defense_rankings -- a full,
    COMPLETED season's real xFP allowed per game, per position, per team.

    Deliberately NOT built by calling build_defense_strength_table at the
    archive's stub week (one past the season's real end, the same
    hypothetical week build_radar_snapshot/build_heatmap_snapshot already
    use): that function's `allowed` side is point-in-time-safe by
    construction, meaning a team's OWN value at week W reflects its games
    STRICTLY BEFORE W -- for that mechanism to reach "the whole real
    season," it needs a row at "one past the end" to shift() the last real
    week's value into, which itself requires resolving that hypothetical
    week's OPPONENT from the schedule. There isn't one -- the season is
    over, so the stub week has no real game, and every team's `allowed`
    row for it drops out of build_defense_strength_table entirely (no
    match in `_team_week_opponent`). A season retrospective isn't
    predicting anything, so there's nothing to protect from leakage here --
    same reasoning build_xfp_summary already uses for reading REAL,
    unshifted per-week xfp rather than the lagged `_ewm3`/`_s2d` columns:
    "this is a season retrospective for a human reader, not a model
    feature."

    Opponent-adjusted the same single-pass way as
    build_defense_strength_table (subtract how much stronger/weaker, on
    average, the offenses a defense faced were than the league average),
    just computed over the whole real season's games at once, in
    xFP-per-game units throughout (a full-season SUM would put allowed and
    opponent-strength on mismatched scales across teams that played
    different numbers of real games due to a bye).

    Returns the same shape as build_defense_rankings:
    {position: [{"team", "raw_ewm3", "raw_s2d", "adj_ewm3", "adj_s2d",
    "rank"}, ...]}, most-favorable-first. raw_ewm3 == raw_s2d and
    adj_ewm3 == adj_s2d here -- there's no separate "recency" vs.
    "season-to-date" distinction for a season that's already over, but the
    same field names are kept so index.html's panel rendering needs no
    archive-specific branch.
    """
    from src.usage import OPP_STRENGTH_POSITIONS, _team_week_opponent

    season_rows = historical_features[
        (historical_features["season"] == season) & historical_features["position"].isin(OPP_STRENGTH_POSITIONS)
    ]
    generated_weekly = (
        season_rows.groupby(["team", "position", "week"])["xfp"].sum(min_count=1)
        .reset_index(name="generated")
    )
    team_avg_generated = (
        generated_weekly.groupby(["team", "position"])["generated"].mean()
        .reset_index(name="avg_generated")
    )
    league_avg_generated = (
        team_avg_generated.groupby("position")["avg_generated"].mean()
    )

    team_week_opp = _team_week_opponent(schedule[schedule["season"] == season])
    allowed_weekly = generated_weekly.merge(
        team_week_opp, on=["team", "week"], how="left"
    ).dropna(subset=["opponent"])
    allowed_avg = (
        allowed_weekly.groupby(["opponent", "position"])["generated"].mean()
        .reset_index().rename(columns={"opponent": "team", "generated": "avg_allowed"})
    )

    # Opponent adjustment: for each defense's real games this season,
    # average the OPPONENT's own season-long per-game generated average
    # (a stable, already-computed full-season number, not a single
    # week's noisy value), then compare that to the league-wide average.
    opponents_faced = allowed_weekly.merge(
        team_avg_generated.rename(columns={"team": "team_opp", "avg_generated": "opp_avg_generated"}),
        left_on=["team", "position"], right_on=["team_opp", "position"], how="left",
    )
    avg_opponent_strength = (
        opponents_faced.groupby(["opponent", "position"])["opp_avg_generated"].mean()
        .reset_index().rename(columns={"opponent": "team"})
    )

    out: dict = {}
    for position in OPP_STRENGTH_POSITIONS:
        pos_allowed = allowed_avg[allowed_avg["position"] == position].set_index("team")["avg_allowed"]
        pos_opp_strength = avg_opponent_strength[avg_opponent_strength["position"] == position].set_index("team")["opp_avg_generated"]
        league_avg = league_avg_generated.get(position, float("nan"))

        rows = []
        for team, allowed in pos_allowed.items():
            sos_correction = pos_opp_strength.get(team, float("nan")) - league_avg
            adj_allowed = allowed - sos_correction
            rows.append({"team": team, "raw": round(float(allowed), 2), "adj": round(float(adj_allowed), 2)})
        rows.sort(key=lambda r: r["adj"], reverse=True)
        out[position] = [
            {"team": r["team"], "raw_ewm3": r["raw"], "raw_s2d": r["raw"],
             "adj_ewm3": r["adj"], "adj_s2d": r["adj"], "rank": rank}
            for rank, r in enumerate(rows, start=1)
        ]
    return out


# ==========================================================================
# TEAM TENDENCIES -- real offense-level identity from pbp (src/team_tendencies.py)
# ==========================================================================
# The honest version of an earlier "coaching scheme" idea -- see
# src/team_tendencies.py's own module docstring for why this measures the
# TEAM, not a coordinator. Surfaced both in meta.caveats (terse) and as its
# own meta field (full), matching MONTE_CARLO_CALIBRATION_CAVEAT's existing
# "surface it where the numbers appear" pattern.
TEAM_TENDENCY_CAVEAT = (
    "These describe the offense a team runs -- pass rate over expected, pace, "
    "red-zone play-calling, target distribution by position -- not any one "
    "player's usage within it. A player who joins this team inherits the "
    "environment, not the incumbent's role."
)


def _team_tendency_metric_block(
    row, ewm3_col: str, s2d_col: str, sample_col: str | None, decimals: int = 4, sample_unit: str = "plays"
) -> dict:
    """
    One metric's {ewm3, s2d, sample_<unit>, sparse} block, from a
    build_team_tendency_table (or season-long equivalent) row. `sample_col`
    is None for target-distribution shares, which share ONE sample count
    across all three positions -- that shared count is attached once at
    the target_distribution level instead (see build_team_tendencies),
    not duplicated into every position's own block.

    `sample_unit` controls both the JSON key (sample_plays vs. sample_games)
    and which sparsity floor applies -- plays_per_game's own sample is a
    GAMES count (how many weekly totals went into the average), a much
    lower-variance quantity than a real PLAY count, so it's held to
    TEAM_TENDENCY_SPARSE_GAMES instead of TEAM_TENDENCY_SPARSE_PLAYS.
    """
    from src.team_tendencies import TEAM_TENDENCY_SPARSE_GAMES, TEAM_TENDENCY_SPARSE_PLAYS

    threshold = TEAM_TENDENCY_SPARSE_GAMES if sample_unit == "games" else TEAM_TENDENCY_SPARSE_PLAYS

    block = {
        "ewm3": None if pd.isna(row[ewm3_col]) else round(float(row[ewm3_col]), decimals),
        "s2d": None if pd.isna(row[s2d_col]) else round(float(row[s2d_col]), decimals),
    }
    if sample_col is not None:
        sample = None if pd.isna(row[sample_col]) else int(row[sample_col])
        block[f"sample_{sample_unit}"] = sample
        block["sparse"] = sample is not None and sample < threshold
    return block


def _team_tendency_row_to_dict(row) -> dict:
    """Shared shape builder for both the live (point-in-time) and season-
    archive team tendency dicts -- both build_team_tendency_table's real
    output and build_season_team_tendencies' own season-long frame use
    these exact column names (see that function's docstring for why)."""
    from src.team_tendencies import TEAM_TENDENCY_METRIC_SAMPLE, TEAM_TENDENCY_SPARSE_PLAYS

    target_sample = row.get("team_targets_s2d_n")
    target_sample = None if pd.isna(target_sample) else int(target_sample)

    return {
        "proe": _team_tendency_metric_block(row, "proe_ewm3", "proe_s2d", TEAM_TENDENCY_METRIC_SAMPLE["proe"]),
        "pace_seconds_per_play": _team_tendency_metric_block(
            row, "seconds_per_play_ewm3", "seconds_per_play_s2d", TEAM_TENDENCY_METRIC_SAMPLE["seconds_per_play"], decimals=2
        ),
        "plays_per_game": _team_tendency_metric_block(
            row, "plays_per_game_ewm3", "plays_per_game_s2d", TEAM_TENDENCY_METRIC_SAMPLE["plays_per_game"],
            decimals=1, sample_unit="games",
        ),
        "red_zone": {
            "inside_20_pass_rate": _team_tendency_metric_block(
                row, "rz20_pass_rate_ewm3", "rz20_pass_rate_s2d", TEAM_TENDENCY_METRIC_SAMPLE["rz20_pass_rate"]
            ),
            "inside_10_pass_rate": _team_tendency_metric_block(
                row, "rz10_pass_rate_ewm3", "rz10_pass_rate_s2d", TEAM_TENDENCY_METRIC_SAMPLE["rz10_pass_rate"]
            ),
        },
        "target_distribution": {
            "rb": _team_tendency_metric_block(row, "target_share_rb_ewm3", "target_share_rb_s2d", None),
            "wr": _team_tendency_metric_block(row, "target_share_wr_ewm3", "target_share_wr_s2d", None),
            "te": _team_tendency_metric_block(row, "target_share_te_ewm3", "target_share_te_s2d", None),
            "sample_targets": target_sample,
            "sparse": target_sample is not None and target_sample < TEAM_TENDENCY_SPARSE_PLAYS,
        },
    }


def build_team_tendencies(
    combined_features: pd.DataFrame, pbp: pd.DataFrame, target_season: int, target_week: int
) -> dict:
    """
    Every team's CURRENT (as of target_week -- real games strictly before
    target_week only) offense-level identity: PROE, pace, red-zone
    play-calling split, target distribution by position. Powers the
    dashboard's Team Tendencies tab. Not a ranked list (unlike
    build_defense_rankings) -- there's no single good/bad direction for
    pace or PROE the way there is for "xFP allowed," so this returns a
    plain {team: {...}} dict, the same per-entity-dict shape radar/heatmap
    already use.

    A team missing from the returned dict simply has no prior in-season
    game yet (week 1 of a season, or an early-season target week) -- not
    enough real plays to show anything honest, the same absence-means-no-
    data convention as build_defense_rankings.

    `pbp` can legitimately arrive completely empty (zero columns, not just
    zero rows) -- weekly_update.py passes that when the current season
    hasn't started yet (see build_heatmap_snapshot's own comment for why).
    Guarded here, before touching any pbp column, same reasoning: a season
    with zero real pbp can only mean every team has zero prior games this
    season, so every team would be absent below regardless.
    """
    if pbp.empty:
        return {}

    from src.team_tendencies import build_team_tendency_table

    table = build_team_tendency_table(combined_features, pbp)
    week_rows = table[(table["season"] == target_season) & (table["week"] == target_week)]
    week_rows = week_rows[week_rows["proe_ewm3"].notna()]

    return {row["team"]: _team_tendency_row_to_dict(row) for _, row in week_rows.iterrows()}


def build_season_team_tendencies(historical_features: pd.DataFrame, pbp: pd.DataFrame, season: int) -> dict:
    """
    Season-ARCHIVE counterpart to build_team_tendencies -- a full completed
    season's real team identity, unshifted (same "nothing left to leak,
    this is a retrospective for a human reader" reasoning
    build_season_defense_rankings already documents). ewm3 and s2d are
    identical here (both just the real season-long weighted average) --
    kept as two separate keys anyway so index.html's rendering needs no
    archive-specific branch, the exact convention build_season_defense_
    rankings already established.

    Weighted by each week's own real sample size (a 70-play week counts
    more than a 40-play week), not a plain mean of weekly rates.
    """
    from src.team_tendencies import _team_week_pace, _team_week_play_rates, _team_week_target_distribution

    weekly_scored = historical_features[historical_features["season"] == season]
    pbp_season = pbp[pbp["season"] == season]

    rates = _team_week_play_rates(pbp_season)
    pace = _team_week_pace(pbp_season)
    position_lookup = weekly_scored[["player_id", "position", "season"]].drop_duplicates(subset=["player_id", "season"])
    targets = _team_week_target_distribution(pbp_season, position_lookup)

    def _weighted(df: pd.DataFrame, value_col: str, weight_col: str) -> pd.Series:
        w = df[weight_col].fillna(0)
        return (df[value_col] * w).groupby(df["team"]).sum() / w.groupby(df["team"]).sum().replace(0, float("nan"))

    out: dict = {}
    teams = sorted(set(rates["team"]) | set(pace["team"]) | set(targets["team"]))
    proe = _weighted(rates, "proe_raw", "neutral_plays")
    plays_per_game = rates.groupby("team")["total_plays"].mean()
    seconds_per_play = _weighted(pace, "seconds_per_play_raw", "pace_gaps")
    rz20 = _weighted(rates, "rz20_pass_rate_raw", "rz20_plays")
    rz10 = _weighted(rates, "rz10_pass_rate_raw", "rz10_plays")
    target_rb = _weighted(targets, "target_share_rb_raw", "team_targets")
    target_wr = _weighted(targets, "target_share_wr_raw", "team_targets")
    target_te = _weighted(targets, "target_share_te_raw", "team_targets")
    games_n = rates.groupby("team").size()
    neutral_n = rates.groupby("team")["neutral_plays"].sum()
    pace_n = pace.groupby("team")["pace_gaps"].sum()
    rz20_n = rates.groupby("team")["rz20_plays"].sum()
    rz10_n = rates.groupby("team")["rz10_plays"].sum()
    target_n = targets.groupby("team")["team_targets"].sum()

    for team in teams:
        row = pd.Series({
            "proe_ewm3": proe.get(team), "proe_s2d": proe.get(team),
            "plays_per_game_ewm3": plays_per_game.get(team), "plays_per_game_s2d": plays_per_game.get(team),
            "seconds_per_play_ewm3": seconds_per_play.get(team), "seconds_per_play_s2d": seconds_per_play.get(team),
            "rz20_pass_rate_ewm3": rz20.get(team), "rz20_pass_rate_s2d": rz20.get(team),
            "rz10_pass_rate_ewm3": rz10.get(team), "rz10_pass_rate_s2d": rz10.get(team),
            "target_share_rb_ewm3": target_rb.get(team), "target_share_rb_s2d": target_rb.get(team),
            "target_share_wr_ewm3": target_wr.get(team), "target_share_wr_s2d": target_wr.get(team),
            "target_share_te_ewm3": target_te.get(team), "target_share_te_s2d": target_te.get(team),
            "games_s2d_n": games_n.get(team),
            "neutral_plays_s2d_n": neutral_n.get(team), "pace_gaps_s2d_n": pace_n.get(team),
            "rz20_plays_s2d_n": rz20_n.get(team), "rz10_plays_s2d_n": rz10_n.get(team),
            "team_targets_s2d_n": target_n.get(team),
        })
        if pd.isna(row["proe_ewm3"]):
            continue
        out[team] = _team_tendency_row_to_dict(row)

    return out


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


def build_weekly_matchup(historical_features: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """
    Per-player, per-week REAL Family 5B matchup for target_season's played
    weeks -- the same per-row `opponent`/OPPONENT_STRENGTH_OUTPUT_COLUMNS
    values add_opponent_strength_features already computed for that exact
    week (the point-in-time-safe number that week's row actually used, not
    re-derived here), plus a RANK among that same week's other opponents
    (build_matchup_snapshot/build_defense_rankings only ever rank the one
    upcoming target week; this ranks every played week, the same "one
    value per already-played week" shape build_weekly_xfp already
    established for the analogous xFP-over-time need).

    This is what makes a real per-player matchup indicator possible for a
    SEASON ARCHIVE, which has no single "current/upcoming week" the live
    dashboard's `matchup` block relies on (see build_matchup_snapshot's
    own docstring, and build_season_defense_rankings' for why the archive
    stub week itself can't resolve a real opponent) -- and, for the LIVE
    export, also lets index.html show a real matchup when browsing an
    ALREADY-PLAYED week of the CURRENT season, not just the upcoming one.
    Nothing here is a forecast: every value describes a game that has
    already been played, so there's no leakage concern in reporting it
    plainly, the same reasoning build_weekly_xfp already uses.

    Ranking mechanics: for each (week, position), the DISTINCT opponents
    that week (deduped first -- every candidate facing the same opponent
    that week carries an identical `opp_def_xfp_allowed_adj_s2d` value, by
    construction, so ranking the un-deduped rows would let a heavily-
    targeted defense's rank be computed multiple times, harmlessly
    redundant but wasteful) are ranked by `opp_def_xfp_allowed_adj_s2d`
    descending -- same "most favorable to face first" convention as
    build_defense_rankings. `method="min"` ties tied opponents at the
    same rank rather than an arbitrary tiebreak order.

    Rows with no resolvable opponent (a bye week) are dropped entirely,
    same as build_weekly_xfp drops a null xfp. A row WITH a real opponent
    but no strength rating yet (early season, or a QB row -- see the scope
    gap in add_opponent_strength_features's own docstring) keeps its real
    `opponent` but gets a null rank -- a real fact (who they played) is
    never hidden just because a derived rating doesn't exist for it.

    Returns: player_id, week, opponent, opp_def_xfp_allowed_ewm3/_s2d/
    _adj_ewm3/_adj_s2d, rank, pool_size -- one row per (player, played
    week with a real opponent). Can be empty.

    `historical_features` can genuinely have NO `opponent` column at all --
    scripts/weekly_update.py's own real pre-draft/week-1 case, where
    `raw_current` is a truly empty (zero-column) frame and the artifact's
    `history_seed` never carries Family 5B columns forward across a season
    boundary by design (see src/pipeline.py::HISTORY_SEED_COLUMNS's own
    comment -- this family resets every season, same as Family 6, so there
    is nothing legitimate to seed). Handled explicitly here rather than
    letting a bare KeyError surface from `.dropna(subset=["opponent"])`.
    """
    from src.usage import OPPONENT_STRENGTH_OUTPUT_COLUMNS

    output_cols = ["player_id", "week", "opponent"] + OPPONENT_STRENGTH_OUTPUT_COLUMNS + ["rank", "pool_size"]
    if "opponent" not in historical_features.columns:
        return pd.DataFrame(columns=output_cols)

    season_rows = historical_features[historical_features["season"] == target_season]
    base = season_rows.dropna(subset=["opponent"])[
        ["player_id", "week", "position", "opponent"] + OPPONENT_STRENGTH_OUTPUT_COLUMNS
    ].reset_index(drop=True)

    distinct = base.dropna(subset=["opp_def_xfp_allowed_adj_s2d"]).drop_duplicates(
        subset=["week", "position", "opponent"]
    ).copy()
    distinct["rank"] = (
        distinct.groupby(["week", "position"])["opp_def_xfp_allowed_adj_s2d"]
        .rank(ascending=False, method="min").astype(int)
    )
    distinct["pool_size"] = distinct.groupby(["week", "position"])["opponent"].transform("size")

    return base.merge(
        distinct[["week", "position", "opponent", "rank", "pool_size"]],
        on=["week", "position", "opponent"], how="left",
    )


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
    xfp_season: int,
    seasons_trained: list[int],
    model_version: str,
    performance: dict = PERFORMANCE_BY_POSITION,
    caveats: list = CAVEATS,
    season_team: pd.DataFrame | None = None,
    player_sim_metrics: pd.DataFrame | None = None,
    matchup: pd.DataFrame | None = None,
    defense_rankings: dict | None = None,
    weekly_matchup: pd.DataFrame | None = None,
    team_tendencies: dict | None = None,
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

    `xfp_season` is which season `xfp_summary`/`weekly_xfp` actually
    describe -- NOT always `target_season`. weekly_update.py's caller
    falls back to `target_season - 1` while the current season has zero
    real games played yet (see its own xfp_season comment), so the same
    export can have `meta.season == 2026` while `xfp.fp_over_expected`
    for every player is real 2025 data. Recorded here explicitly, as its
    own `meta.xfp_season` field, rather than left for a reader to infer
    from `target_season` alone -- index.html's Players table reads this
    to label the FP Over Exp column "(2025)" whenever it differs from the
    season actually being displayed, so a real prior-season luck number
    is never silently mistaken for describing the season on screen.

    `season_team` (get_season_team_map's output, [player_id, season_team])
    is optional and archive-only -- scripts/weekly_update.py's live export
    never passes it, so every player's "team" key there is null; a caller
    that wants the season this player actually played for (not their
    current Sleeper team, which the dashboard already has from the live
    player DB) passes it explicitly. A player missing from `season_team`
    (shouldn't happen for a real archive candidate) gets null, not a
    KeyError.

    `player_sim_metrics` (build_player_simulation_metrics's output:
    player_id, game_id, boom_prob, bust_prob, thresholds, and the 5 CQR
    quantile columns) is optional -- scripts/archive_season.py never
    passes it, same "hypothetical week, not a real one" reasoning that
    already keeps the top-level `simulation` block null for archives (see
    that module's own comment). When provided, each covered player gets a
    `monte_carlo` block; everyone else gets null, not a guessed number.

    `team_tendencies` (build_team_tendencies's or build_season_team_
    tendencies's output) is Team Tendencies' league-wide wiring -- optional
    for the same "hypothetical week" reason as player_sim_metrics, written
    ONLY as its own top-level key (sibling to `players`), NOT joined onto
    any individual player -- see src/team_tendencies.py's own docstring for
    why these deliberately describe the offense, not any one player's row.

    `matchup` (build_matchup_snapshot's output) and `defense_rankings`
    (build_defense_rankings's or build_season_defense_rankings's output)
    are Family 5B's opponent-defensive-strength wiring -- both optional for
    the same "hypothetical week" reason as player_sim_metrics.
    `defense_rankings` is used here only to look up each player's OWN
    opponent's rank at their OWN position (a number that means nothing
    without the other 31 teams to compare against, which is why it isn't
    just another column merged onto `matchup`) -- the full table is ALSO
    written into the payload as its own top-level `defense_rankings` key
    (sibling to `players`), for the dashboard's standalone "which defenses
    are favorable this week" panel, which needs the whole league, not one
    player's opponent.

    `weekly_matchup` (build_weekly_matchup's output) is grouped into
    {player_id: {week_str: matchup_dict}} the same way `weekly_xfp` is --
    a REAL per-played-week matchup history, same field shape as the single
    `matchup` block, so a season archive (which has no single "current
    week" for `matchup` itself to describe -- see build_matchup_snapshot's
    docstring) can still show a real matchup for whichever week a reader
    is looking at.

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

    # {player_id: {week_str: matchup_dict}} -- same shape/field names as
    # the single upcoming-week `matchup` block below, one per REAL played
    # week instead of one for the target week. Lets index.html show a real
    # matchup for a season archive (no single "current week" exists there)
    # or for an already-played week of a live, in-progress season.
    weekly_matchup_by_player = {}
    if weekly_matchup is not None and not weekly_matchup.empty:
        for player_id, group in weekly_matchup.groupby("player_id"):
            weekly_matchup_by_player[player_id] = {
                str(int(row["week"])): {
                    "opponent": row["opponent"],
                    "def_xfp_allowed_ewm3": (
                        None if pd.isna(row["opp_def_xfp_allowed_ewm3"]) else round(float(row["opp_def_xfp_allowed_ewm3"]), 2)
                    ),
                    "def_xfp_allowed_s2d": (
                        None if pd.isna(row["opp_def_xfp_allowed_s2d"]) else round(float(row["opp_def_xfp_allowed_s2d"]), 2)
                    ),
                    "def_xfp_allowed_adj_ewm3": (
                        None if pd.isna(row["opp_def_xfp_allowed_adj_ewm3"]) else round(float(row["opp_def_xfp_allowed_adj_ewm3"]), 2)
                    ),
                    "def_xfp_allowed_adj_s2d": (
                        None if pd.isna(row["opp_def_xfp_allowed_adj_s2d"]) else round(float(row["opp_def_xfp_allowed_adj_s2d"]), 2)
                    ),
                    "rank": None if pd.isna(row["rank"]) else int(row["rank"]),
                    "pool_size": None if pd.isna(row["pool_size"]) else int(row["pool_size"]),
                }
                for _, row in group.iterrows()
            }

    # {(position, team): rank} -- flattened once here rather than looked up
    # per player from the nested {position: [...]} shape build_defense_
    # rankings returns.
    rank_lookup = {}
    if defense_rankings:
        for position, teams in defense_rankings.items():
            for entry in teams:
                rank_lookup[(position, entry["team"])] = {"rank": entry["rank"], "pool_size": len(teams)}

    cw = crosswalk.dropna(subset=["gsis_id", "sleeper_id"]).drop_duplicates(subset=["gsis_id"])
    merged = scoped_predictions.merge(usage, on="player_id", how="left")
    merged = merged.merge(trend, on="player_id", how="left")
    merged = merged.merge(xfp_summary, on="player_id", how="left")
    if season_team is not None:
        merged = merged.merge(season_team, on="player_id", how="left")
    if player_sim_metrics is not None:
        merged = merged.merge(player_sim_metrics, on="player_id", how="left")
    if matchup is not None:
        merged = merged.merge(matchup, on="player_id", how="left")
    n_scoped = len(merged)
    merged = merged.merge(cw[["gsis_id", "sleeper_id"]], left_on="player_id", right_on="gsis_id", how="inner")
    n_matched = len(merged)

    players = {}
    for _, row in merged.iterrows():
        players[row["sleeper_id"]] = {
            "team": None if pd.isna(row.get("season_team")) else row["season_team"],
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
            "weekly_matchup": weekly_matchup_by_player.get(row["player_id"], {}),
            "radar": radar.get(
                row["player_id"], {"eligible": False, "games_played": 0, "min_games": MIN_GAMES_FOR_TREND}
            ),
            "heatmap": heatmap.get(
                row["player_id"], {"eligible": False, "games_played": 0, "min_games": MIN_GAMES_FOR_TREND}
            ),
            "monte_carlo": (
                None if player_sim_metrics is None or pd.isna(row.get("boom_prob"))
                else {
                    "boom_prob": round(float(row["boom_prob"]), 3),
                    "bust_prob": round(float(row["bust_prob"]), 3),
                    "thresholds": {k: round(float(v), 3) for k, v in row["thresholds"].items()},
                    "quantiles": {
                        "q10": round(float(row["pred_q10_cqr"]), 2),
                        "q25": round(float(row["pred_q25_cqr"]), 2),
                        "q50": round(float(row["pred_q50"]), 2),
                        "q75": round(float(row["pred_q75_cqr"]), 2),
                        "q90": round(float(row["pred_q90_cqr"]), 2),
                    },
                    "game_id": None if pd.isna(row.get("game_id")) else row["game_id"],
                }
            ),
            "matchup": (
                None if matchup is None or pd.isna(row.get("opponent"))
                else {
                    "opponent": row["opponent"],
                    "def_xfp_allowed_ewm3": (
                        None if pd.isna(row.get("opp_def_xfp_allowed_ewm3"))
                        else round(float(row["opp_def_xfp_allowed_ewm3"]), 2)
                    ),
                    "def_xfp_allowed_s2d": (
                        None if pd.isna(row.get("opp_def_xfp_allowed_s2d"))
                        else round(float(row["opp_def_xfp_allowed_s2d"]), 2)
                    ),
                    "def_xfp_allowed_adj_ewm3": (
                        None if pd.isna(row.get("opp_def_xfp_allowed_adj_ewm3"))
                        else round(float(row["opp_def_xfp_allowed_adj_ewm3"]), 2)
                    ),
                    "def_xfp_allowed_adj_s2d": (
                        None if pd.isna(row.get("opp_def_xfp_allowed_adj_s2d"))
                        else round(float(row["opp_def_xfp_allowed_adj_s2d"]), 2)
                    ),
                    **(rank_lookup.get((row["position"], row["opponent"])) or {"rank": None, "pool_size": None}),
                }
            ),
        }

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "season": target_season,
            "week": target_week,
            "xfp_season": xfp_season,
            "model_version": model_version,
            "seasons_trained": seasons_trained,
            "performance": performance,
            "caveats": caveats,
            "monte_carlo_caveat": MONTE_CARLO_CALIBRATION_CAVEAT,
            "monte_carlo_n_sims": SIMULATION_N_SIMS_MATCHUP,
            "team_tendency_caveat": TEAM_TENDENCY_CAVEAT,
        },
        "players": players,
        "defense_rankings": defense_rankings,
        "team_tendencies": team_tendencies,
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

    n_with_monte_carlo = sum(1 for p in players.values() if p.get("monte_carlo") is not None)
    bad_monte_carlo = [
        pid for pid, p in players.items()
        if p.get("monte_carlo") is not None
        and (
            not (0 <= p["monte_carlo"]["boom_prob"] <= 1)
            or not (0 <= p["monte_carlo"]["bust_prob"] <= 1)
            or any(not (0 <= v <= 1) for v in p["monte_carlo"]["thresholds"].values())
            or not (p["monte_carlo"]["quantiles"]["q10"] <= p["monte_carlo"]["quantiles"]["q50"]
                    <= p["monte_carlo"]["quantiles"]["q90"])
        )
    ]
    assert not bad_monte_carlo, (
        f"{len(bad_monte_carlo)} players have an invalid monte_carlo block "
        f"(probability outside [0, 1], or q10/q50/q90 out of order): {bad_monte_carlo[:5]}"
    )

    bad_matchup_ranks = [
        pid for pid, p in players.items()
        if p.get("matchup") and p["matchup"]["rank"] is not None
        and not (1 <= p["matchup"]["rank"] <= p["matchup"]["pool_size"])
    ]
    assert not bad_matchup_ranks, (
        f"{len(bad_matchup_ranks)} players have a matchup.rank outside [1, pool_size]: {bad_matchup_ranks[:5]}"
    )

    bad_weekly_matchup_ranks = [
        pid for pid, p in players.items()
        if any(
            m["rank"] is not None and not (1 <= m["rank"] <= m["pool_size"])
            for m in (p.get("weekly_matchup") or {}).values()
        )
    ]
    assert not bad_weekly_matchup_ranks, (
        f"{len(bad_weekly_matchup_ranks)} players have a weekly_matchup entry with rank outside "
        f"[1, pool_size]: {bad_weekly_matchup_ranks[:5]}"
    )

    defense_rankings = payload.get("defense_rankings") or {}
    bad_defense_rankings = [
        position for position, teams in defense_rankings.items()
        if sorted(t["rank"] for t in teams) != list(range(1, len(teams) + 1))
    ]
    assert not bad_defense_rankings, (
        f"defense_rankings ranks aren't a clean 1..N sequence for: {bad_defense_rankings}"
    )

    n_with_matchup = sum(1 for p in players.values() if p.get("matchup") is not None)
    n_with_weekly_matchup = sum(1 for p in players.values() if p.get("weekly_matchup"))

    team_tendencies = payload.get("team_tendencies") or {}
    bad_team_tendency_shares = [
        team for team, block in team_tendencies.items()
        if any(
            v["ewm3"] is not None and not (0 <= v["ewm3"] <= 1)
            for v in [
                block["red_zone"]["inside_20_pass_rate"], block["red_zone"]["inside_10_pass_rate"],
                block["target_distribution"]["rb"], block["target_distribution"]["wr"], block["target_distribution"]["te"],
            ]
        )
    ]
    assert not bad_team_tendency_shares, (
        f"team_tendencies has a share outside [0, 1] for: {bad_team_tendency_shares[:5]}"
    )

    return {
        "n_players": len(players),
        "all_keys_real_sleeper_ids": True,
        "no_null_projections": True,
        "floor_le_point_le_ceiling": True,
        "radar_percentiles_in_range": True,
        "heatmap_shares_sum_to_one": True,
        "n_with_monte_carlo": n_with_monte_carlo,
        "monte_carlo_probabilities_in_range": True,
        "n_with_matchup": n_with_matchup,
        "matchup_ranks_valid": True,
        "n_with_weekly_matchup": n_with_weekly_matchup,
        "weekly_matchup_ranks_valid": True,
        "defense_rankings_positions": list(defense_rankings.keys()),
        "n_team_tendencies": len(team_tendencies),
        "team_tendency_shares_in_range": True,
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


def build_player_simulation_metrics(
    combined_features: pd.DataFrame,
    schedule: pd.DataFrame,
    target_season: int,
    target_week: int,
    artifact: dict,
    n_sims: int = SIMULATION_N_SIMS_MATCHUP,
) -> pd.DataFrame:
    """
    Per-candidate point-in-time Monte Carlo metrics (src/simulate.py's
    player_point_in_time_metrics) PLUS the ingredients -- game_id and the
    full 5-point CQR-calibrated quantile distribution -- every candidate
    needs so the dashboard can run an ad-hoc start-over-replacement
    comparison against ANY other exported candidate, not just this week's
    real fantasy starters (build_starter_quantile_rows/
    build_matchup_simulation's scope).

    Deliberately does NOT export the raw n_sims draws themselves: at
    ~465 exported candidates that would be millions of numbers for one
    week's export (versus 5 quantiles + one game_id per player here). The
    one-factor Gaussian copula src/simulate.py::sample_player_week
    implements is small and exact, so the CLIENT can re-run its own fresh
    Monte Carlo from these ingredients (matching game_id => correlated
    draws, exactly like the Python-side simulator) instead of needing
    Python's specific draws replayed -- a Monte Carlo ESTIMATE of a
    well-defined probability doesn't need to share a random seed with
    another estimate of the same quantity to both be valid. index.html's
    client-side start-over-replacement mirrors this same copula in JS.

    Uses predict_quantiles_with_models (src/model.py) directly rather than
    the single point/floor/ceiling predict_with_models everything else in
    this module uses -- the simulator's own 5-quantile input contract,
    already used by build_simulation_block's real-matchup path, just
    applied here to the FULL target-week candidate pool instead of one
    week's real starters.

    A candidate whose team isn't resolvable to a real game this week (bye,
    or an unresolvable team) gets `game_id=None` in the OUTPUT -- honestly
    "no real game to correlate with" -- but still needs SOME game_id to
    hand sample_player_week (which correlates same-game_id rows with each
    other): given a private, per-player synthetic id instead, so a bucket
    of exactly one player behaves as pure idiosyncratic noise (nothing
    else shares that id to correlate against) without special-casing
    sample_player_week itself.

    Returns: player_id, game_id (nullable), boom_prob, bust_prob,
    thresholds (dict), pred_q10_cqr, pred_q25_cqr, pred_q50, pred_q75_cqr,
    pred_q90_cqr -- one row per candidate with model coverage this week
    (QB/RB/WR/TE only, same scope as every other quantile-based
    computation in this pipeline). Empty DataFrame if there's nothing to
    predict, mirroring predict_target_week_from_artifact's own
    empty-input behavior.
    """
    empty_cols = [
        "player_id", "game_id", "boom_prob", "bust_prob", "thresholds",
        "pred_q10_cqr", "pred_q25_cqr", "pred_q50", "pred_q75_cqr", "pred_q90_cqr",
    ]
    test_mask = (combined_features["season"] == target_season) & (combined_features["week"] == target_week)
    test_df = combined_features[test_mask]
    quantiles = predict_quantiles_with_models(
        test_df, artifact["models"], artifact["cqr_widen_by_10_90"], artifact["cqr_widen_by_25_75"],
        feature_cols=artifact["feature_columns"],
    )
    if quantiles.empty:
        return pd.DataFrame(columns=empty_cols)
    quantiles = quantiles.reset_index(drop=True)

    team_game_id_lookup = build_team_game_id_lookup(schedule)
    game_id_by_team_week = dict(zip(
        zip(team_game_id_lookup["season"], team_game_id_lookup["week"], team_game_id_lookup["team"]),
        team_game_id_lookup["game_id"],
    ))
    team_by_player = dict(zip(test_df["player_id"], test_df["team"]))

    resolved_game_id = quantiles["player_id"].map(team_by_player).map(
        lambda t: game_id_by_team_week.get((target_season, target_week, t))
    )
    # sample_player_week correlates every row sharing one game_id -- a
    # shared None/NaN bucket would wrongly correlate every bye-week/
    # unresolved player with EVERY OTHER one, so each gets its own private
    # placeholder id instead of the real (missing) one.
    sim_game_id = resolved_game_id.where(resolved_game_id.notna(), "no_game_" + quantiles["player_id"].astype(str))

    sim_input = quantiles.copy()
    sim_input["game_id"] = sim_game_id
    draws = sample_player_week(sim_input, n_sims=n_sims)
    metrics = player_point_in_time_metrics(draws, quantiles["position"])

    out = quantiles[["player_id", "pred_q10_cqr", "pred_q25_cqr", "pred_q50", "pred_q75_cqr", "pred_q90_cqr"]].copy()
    out["game_id"] = resolved_game_id
    out["boom_prob"] = [m["boom_prob"] for m in metrics]
    out["bust_prob"] = [m["bust_prob"] for m in metrics]
    out["thresholds"] = [m["thresholds"] for m in metrics]
    return out[empty_cols]


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
