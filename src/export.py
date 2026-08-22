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
USAGE_EXPORT_COLUMNS = [
    "target_share_ewm3", "touch_share_ewm3", "offense_pct_ewm3",
    "snap_share_delta_3wk", "rz_target_share_ewm3", "rz_carry_share_ewm3",
    "prev_season_target_share", "prev_season_touch_share", "prev_season_offense_pct",
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

    Returns (payload, crosswalk_report) -- crosswalk_report has
    {"n_scoped", "n_matched", "match_rate"} so the match rate gets reported,
    not just assumed.
    """
    from src.usage import TREND_SOURCE_FEATURES

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

    return {
        "n_players": len(players),
        "all_keys_real_sleeper_ids": True,
        "no_null_projections": True,
        "floor_le_point_le_ceiling": True,
    }
