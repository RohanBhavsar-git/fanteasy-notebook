"""
FanTeasy Stats — Phase 6 projection model, Formulation A (Order of Work step 6)

Direct prediction of `custom_points` from the Family 1-6 feature table, one
LightGBM model per position, evaluated with expanding-window walk-forward
validation against four baselines: season-to-date average, trailing 3-week
average, Sleeper's projection, and trailing xFP.

No hyperparameter tuning here on purpose -- LightGBM's own defaults, so the
untuned model's standing against the baselines is visible before any tuning
could flatter it. Formulation B (residual vs. Sleeper) is a separate step.

Design note on feature selection -- this is the single most leak-sensitive
part of this module:
    weekly_features.parquet has ~360 columns, and MOST of them are unsafe as
    model inputs -- they're the current week's own OUTCOME (raw box-score
    stats, custom_points itself, and every Family 1/2/3/4 "this week" output
    like target_share, offense_pct, adot, rz_targets -- all of these are
    computed FROM week N's own plays, so using them to predict week N's own
    points is circular, not predictive).

    The safe feature set is exactly two things:
      - ROLLING_OUTPUT_COLUMNS (Family 6): every one of these is already
        `.shift(1)`-ed by construction (see usage.py's leakage tests), so
        by definition none of them can see week N's own outcome.
      - CONTEXT_OUTPUT_COLUMNS (Family 5): is_home, days_rest, spread,
        game_total, team_implied_total, roof, surface, temp, wind. These
        describe week N's PREGAME circumstances (schedule, Vegas lines,
        weather forecast) -- genuinely knowable before kickoff, so unlike
        every other family's "current week" columns, using them as-is
        (not lagged) is correct, not a leak. Unlike ROLLING_OUTPUT_COLUMNS,
        this family is NOT used as one shared block -- see
        FEATURE_COLUMNS_BY_POSITION's own comment for why it's split into
        VEGAS_SCHEDULE_OUTPUT_COLUMNS/WEATHER_OUTPUT_COLUMNS and included
        per position, not per family.
    xfp/fp_over_expected (current week) are deliberately excluded even
    though they sound like features -- xfp for week N requires knowing the
    player's ACTUAL week-N opportunities (targets/carries that week), which
    isn't known before kickoff. Only xfp_ewm3/xfp_s2d/xfp_vol (the rolling,
    lagged versions) are safe, and those are already in ROLLING_OUTPUT_COLUMNS.
    Sleeper's own projection is deliberately NOT a feature here either --
    it's reserved as an independent baseline to compare against (and the
    starting point for Formulation B), not folded into Formulation A's inputs.

Design note on the Sleeper-projection baseline:
    Sleeper's projection endpoint returns projected STAT values (rec, rec_yd,
    pass_td, fum, ...) using the exact same key vocabulary as the league's
    own scoring_settings -- verified directly (every nonzero offensive
    scoring key has a same-named projection column). That means the
    projected point total for this league's rules is a plain weighted sum
    over the projection row, with NO detour through compute_custom_score()'s
    nflverse-column mapping (that machinery exists to translate nflverse's
    weekly-stat column names to Sleeper's key names; Sleeper's own
    projections are already in Sleeper's key names).
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src.team_tendencies import TEAM_TENDENCY_OUTPUT_COLUMNS
from src.usage import (
    OPPONENT_STRENGTH_OUTPUT_COLUMNS,
    ROLLING_OUTPUT_COLUMNS,
    VEGAS_SCHEDULE_OUTPUT_COLUMNS,
    WEATHER_OUTPUT_COLUMNS,
)

logger = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")

# OPPONENT_STRENGTH_OUTPUT_COLUMNS (Family 5B -- opponent defensive
# strength by position) is, like the Family 5 context columns below, a
# describes-THIS-WEEK'S-MATCHUP family rather than a describes-the-player-
# over-time one: each value already IS a trailing, shift(1)-safe summary
# of the player's OPPONENT (not the player), so it's used as-is here
# rather than being fed through Family 6's rolling treatment a second
# time -- doing that would average together different opponents across
# different weeks, the same reason CONTEXT_OUTPUT_COLUMNS itself is
# excluded from ROLLING_SOURCE_COLUMNS (see usage.py's comment on that
# exclusion). Null for QB by construction -- see
# add_opponent_strength_features's docstring.
#
# Unlike CONTEXT_OUTPUT_COLUMNS (see FEATURE_COLUMNS_BY_POSITION below),
# this family is NOT position-differentiated -- Family 5B's own walk-
# forward test already found a real WR/TE gain and a null-by-construction
# QB / noise RB result (see PROJECT_CONTEXT.md's Family 5B findings), so
# it stays in the shared base rather than needing its own per-position
# split.
FEATURE_COLUMNS = (
    list(ROLLING_OUTPUT_COLUMNS) + list(OPPONENT_STRENGTH_OUTPUT_COLUMNS)
)

# Per-position feature sets. Two families ride on top of the shared
# FEATURE_COLUMNS base, both position-differentiated because a single
# shared list can't help one position while excluding another that the
# same column set hurts:
#
# Team Tendencies (src/team_tendencies.py) -- walk-forward tested (same
# eval_min_season window/methodology as every other feature family in
# this pipeline; see PROJECT_CONTEXT.md's Team Tendencies findings for
# the full table): a real, substantial MAE improvement for QB (-0.18),
# noise for RB/WR (+0.004/+0.006), and a real DEGRADATION for TE (+0.024).
# A QB throws every pass his team throws, so team-wide pace/PROE gate QB
# volume about as directly as a feature can; for RB/WR/TE, the player-
# level rolling shares already in FEATURE_COLUMNS (target_share_ewm3 and
# friends) already capture THAT PLAYER'S OWN slice of the team total, so
# the team-wide aggregate is redundant at best and actively misleads the
# model at TE specifically.
#
# Game context (Family 5) -- CONTEXT_OUTPUT_COLUMNS split into
# VEGAS_SCHEDULE_OUTPUT_COLUMNS and WEATHER_OUTPUT_COLUMNS (see
# usage.py's own comment on that split) after a block-level ablation of
# the whole family masked a real, opposite-signed pair of effects at RB
# (Vegas helped, weather hurt, and the two nearly canceled into a false
# "no signal" reading at the block level -- +/-0.008 combined vs. -0.027
# for Vegas alone and +0.015 for weather alone). Walk-forward tested per
# subfamily per position, same methodology, full numbers in
# PROJECT_CONTEXT.md's Context Columns findings:
#
#   position | Vegas/schedule alone | weather alone
#   ---------|-----------------------|---------------
#   QB       | -0.087 (real)         | -0.001 (noise alone -- see below)
#   RB       | -0.027 (real)         | +0.015 (real degradation)
#   WR       | -0.005 (noise)        | -0.014 (small real)
#   TE       | +0.022 (real degrad.) | +0.028 (real degradation)
#
# QB is the one exception to "include only what tested positive alone":
# Vegas/schedule and Team Tendencies overlap heavily for QB (a factorial
# test found each factor's solo effect of ~-0.23 shrinks to ~-0.08 once
# the other is already present -- redundant, not additive, though neither
# is fully subsumed by the other). Weather's OWN solo effect for QB is
# ~0, which would argue for dropping it -- but dropping weather from an
# already-Vegas+TT QB feature set measured +0.073 MAE WORSE than keeping
# all three together (6.2472 vs. the committed 6.1738): a real
# interaction, not redundancy. So QB keeps weather despite its flat solo
# number -- don't drop it on that number alone without re-checking this
# interaction.
FEATURE_COLUMNS_BY_POSITION: dict[str, list[str]] = {
    "QB": (
        list(FEATURE_COLUMNS)
        + list(VEGAS_SCHEDULE_OUTPUT_COLUMNS)
        + list(WEATHER_OUTPUT_COLUMNS)  # kept despite a ~0 solo effect -- see the interaction note above
        + list(TEAM_TENDENCY_OUTPUT_COLUMNS)
    ),
    "RB": list(FEATURE_COLUMNS) + list(VEGAS_SCHEDULE_OUTPUT_COLUMNS),
    # weather deliberately excluded for RB -- measured +0.015 MAE (real degradation), not just untested
    "WR": list(FEATURE_COLUMNS) + list(WEATHER_OUTPUT_COLUMNS),
    # Vegas/schedule deliberately excluded for WR -- measured -0.005 MAE (noise, doesn't clear the bar)
    "TE": list(FEATURE_COLUMNS),
    # both Vegas/schedule (+0.022) and weather (+0.028) deliberately excluded for TE -- both real degradations;
    # TEAM_TENDENCY_OUTPUT_COLUMNS also excluded -- tested worse separately, see the Team Tendencies findings above
}

# The union across every position's own list -- for the ONE step that
# still needs a single flat column set: casting categorical dtypes
# (roof/surface) consistently across a combined, not-yet-position-split
# frame BEFORE it gets divided into each position's own columns (see
# src/export.py::build_target_week_features and walk_forward_predict's
# own "cast once, before splitting into folds" reasoning).
ALL_FEATURE_COLUMNS = sorted({col for cols in FEATURE_COLUMNS_BY_POSITION.values() for col in cols})

TARGET_BASELINE_OUTPUT_COLUMNS = ["baseline_season_to_date_avg", "baseline_trailing_3wk_avg"]

MIN_TRAIN_ROWS = 30  # skip a fold if the expanding training window is this thin or thinner


# ==========================================================================
# BASELINE 1-2: season-to-date average / trailing 3-week average of custom_points
# ==========================================================================
def add_target_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add baseline_season_to_date_avg and baseline_trailing_3wk_avg for
    custom_points -- the same in-season, `.shift(1)`-ed, season-boundary-
    respecting methodology as add_rolling_features, applied to the TARGET
    column itself, which Family 6 deliberately excludes from
    ROLLING_SOURCE_COLUMNS (it's the thing being predicted, not a feature).

    Idempotent: existing output columns are dropped before recomputing.
    """
    required = ["player_id", "season", "week", "custom_points"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_target_baselines: df is missing columns {missing}")

    out = df.drop(columns=[c for c in TARGET_BASELINE_OUTPUT_COLUMNS if c in df.columns])

    sorted_df = out.sort_values(["player_id", "season", "week"])
    group_keys = [sorted_df["player_id"], sorted_df["season"]]
    grouped = sorted_df.groupby(["player_id", "season"], sort=False)

    def _shift_within_group(s: pd.Series) -> pd.Series:
        return s.groupby(group_keys, sort=False).shift(1)

    s2d_raw = grouped["custom_points"].expanding().mean().droplevel([0, 1])
    roll3_raw = grouped["custom_points"].rolling(3, min_periods=1).mean().droplevel([0, 1])

    results = {
        "baseline_season_to_date_avg": _shift_within_group(s2d_raw),
        "baseline_trailing_3wk_avg": _shift_within_group(roll3_raw),
    }
    new_cols = pd.DataFrame(
        {name: series.reindex(out.index) for name, series in results.items()}, index=out.index
    )
    return pd.concat([out, new_cols], axis=1)


# ==========================================================================
# BASELINE 3: Sleeper's projection, scored with this league's own rules
# ==========================================================================
def sleeper_projected_points(proj_df: pd.DataFrame, scoring_settings: dict) -> pd.Series:
    """
    Weighted sum of Sleeper's projected stat columns using this league's
    scoring weights. proj_df's columns already use Sleeper's own key
    vocabulary (rec, rec_yd, pass_td, ...) -- the same one scoring_settings
    is keyed by -- so no column-name translation is needed here, unlike
    compute_custom_score() which translates nflverse's weekly-stat column
    names into that vocabulary first.
    """
    points = pd.Series(0.0, index=proj_df.index)
    for key, weight in scoring_settings.items():
        if not weight or key not in proj_df.columns:
            continue
        points += proj_df[key].fillna(0) * float(weight)
    return points


def add_sleeper_baseline(
    df: pd.DataFrame, seasons_weeks: list[tuple[int, int]], crosswalk: pd.DataFrame,
    scoring_settings: dict, get_sleeper_projections_fn,
) -> pd.DataFrame:
    """
    Add baseline_sleeper_proj: Sleeper's own pregame projection for that
    exact week, scored with this league's rules. Fetches (and caches, via
    get_sleeper_projections_fn) every (season, week) in seasons_weeks.

    Not a leak by construction -- Sleeper published this projection before
    kickoff of the week it's for; it isn't derived from that week's outcome.

    Idempotent: existing output column is dropped before recomputing.
    """
    out = df.drop(columns=["baseline_sleeper_proj"], errors="ignore")

    cw = crosswalk.dropna(subset=["sleeper_id", "gsis_id"]).drop_duplicates(subset=["sleeper_id"])

    frames = []
    for season, week in seasons_weeks:
        proj = get_sleeper_projections_fn(season, week)
        proj = proj.merge(cw[["sleeper_id", "gsis_id"]], on="sleeper_id", how="inner")
        proj["baseline_sleeper_proj"] = sleeper_projected_points(proj, scoring_settings)
        proj["season"] = season
        proj["week"] = week
        frames.append(proj[["gsis_id", "season", "week", "baseline_sleeper_proj"]])

    sleeper_table = pd.concat(frames, ignore_index=True).rename(columns={"gsis_id": "player_id"})
    dupes = sleeper_table.duplicated(subset=["player_id", "season", "week"], keep=False)
    if dupes.any():
        raise ValueError(
            f"{dupes.sum()} rows collide on (player_id, season, week) in the Sleeper "
            "projection table after the sleeper_id -> gsis_id join."
        )

    return out.merge(sleeper_table, on=["player_id", "season", "week"], how="left")


# ==========================================================================
# WALK-FORWARD VALIDATION
# ==========================================================================
def chronological_folds(
    df: pd.DataFrame, warmup_weeks: int, eval_min_season: int | None = None
) -> list[tuple[int, int]]:
    """
    (season, week) pairs to evaluate on, in chronological order, skipping
    the first `warmup_weeks` weeks of the EARLIEST season in df (no
    training history exists yet for those). Later seasons are not
    warmed-up again -- by the time a later season starts, the expanding
    training window already has a full prior season in it.

    `eval_min_season`: if set, folds before this season are excluded from
    the returned (evaluated) list entirely, WITHOUT touching df itself --
    those earlier seasons' rows remain available as training history for
    every returned fold (walk_forward_predict's train_mask is unaffected
    by this parameter). This is what lets a data-volume experiment hold
    the EVALUATION population fixed (e.g. always 2024-2025) while varying
    how much pre-2024 history the model gets to train on -- the only way
    to isolate "does more training history help" from "are we now also
    being scored on a different, maybe easier or harder, set of weeks."
    """
    weeks = sorted(set(map(tuple, df[["season", "week"]].drop_duplicates().to_numpy())))
    first_season = weeks[0][0]
    folds = [(s, w) for s, w in weeks if not (s == first_season and w <= warmup_weeks)]
    if eval_min_season is not None:
        folds = [(s, w) for s, w in folds if s >= eval_min_season]
    return folds


def walk_forward_predict(
    df: pd.DataFrame,
    position: str,
    feature_cols: list[str] = FEATURE_COLUMNS,
    target_col: str = "custom_points",
    warmup_weeks: int = 4,
    eval_min_season: int | None = None,
    lgb_params: dict | None = None,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward prediction for one position. For each
    fold (season, week) after the warmup period: train a FRESH LightGBM
    model on every row STRICTLY BEFORE that fold (any earlier week, any
    earlier season -- chronological order across the whole timeline, not
    reset per season, since training data is not a "trend feature" the
    way Family 6's columns are), predict that fold's rows. No shuffling,
    no KFold -- each fold's test rows are never in that fold's own
    training set, by construction.

    `eval_min_season`: see chronological_folds -- restricts which folds
    get EVALUATED without restricting what a later fold can TRAIN on.
    Earlier seasons in `df` still feed every evaluated fold's training
    window even when eval_min_season excludes them from being predicted.

    No hyperparameter tuning: `lgb_params` defaults to None, meaning
    LightGBM's own out-of-the-box defaults (objective='regression' is the
    only thing pinned, since the default objective needs to be explicit).

    Returns:
        Long frame with player_id, player_display_name, season, week,
        target_col, and `pred_model_a` -- one row per (player, evaluated
        week), out-of-sample.
    """
    pos_df = df[df["position"] == position].sort_values(["season", "week"]).reset_index(drop=True)
    folds = chronological_folds(pos_df, warmup_weeks, eval_min_season)
    # Cast once on the full frame, before splitting into folds, so train
    # and test always share the same category set -- casting per-fold
    # separately would let a category present only in a later fold's test
    # set silently vanish from an earlier fold's training categories.
    pos_df = _cast_categoricals(pos_df, feature_cols)

    params = {"objective": "regression", "verbosity": -1, "random_state": 42}
    if lgb_params:
        params.update(lgb_params)

    id_cols = [c for c in ["player_id", "player_display_name", "season", "week"] if c in pos_df.columns]

    preds = []
    for season, week in folds:
        train_mask = (pos_df["season"] < season) | ((pos_df["season"] == season) & (pos_df["week"] < week))
        test_mask = (pos_df["season"] == season) & (pos_df["week"] == week)
        train = pos_df.loc[train_mask]
        test = pos_df.loc[test_mask]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue

        model = lgb.LGBMRegressor(**params)
        model.fit(train[feature_cols], train[target_col])
        pred = model.predict(test[feature_cols])

        result = test[id_cols + [target_col]].copy()
        result["pred_model_a"] = pred
        preds.append(result)

    if not preds:
        return pd.DataFrame(columns=id_cols + [target_col, "pred_model_a"])
    return pd.concat(preds, ignore_index=True)


def walk_forward_predict_residual(
    df: pd.DataFrame,
    position: str,
    baseline_col: str,
    feature_cols: list[str] = FEATURE_COLUMNS,
    target_col: str = "custom_points",
    warmup_weeks: int = 4,
    eval_min_season: int | None = None,
    lgb_params: dict | None = None,
) -> pd.DataFrame:
    """
    Formulation B: train on (target_col - baseline_col) instead of
    target_col directly, then reconstruct pred_custom_points =
    predicted_residual + baseline_col at prediction time. Same idea
    whether baseline_col is Sleeper's projection (the spec's own
    Formulation B) or a trailing average (the earlier season_to_date_avg
    diagnostic that established this pattern) -- isolates whether the
    feature set adds anything on top of whatever the baseline already
    knows, rather than asking the model to learn that baseline's
    information over again from scratch.

    Rows where baseline_col is null can't have a defined residual target
    and are dropped before training (and so are absent from the
    returned predictions too) -- reusing walk_forward_predict internally
    rather than duplicating its fold/training logic.

    Returns:
        Same shape as walk_forward_predict, plus `residual_target` (the
        ACTUAL residual, not predicted -- check this isn't degenerate
        before trusting the reconstruction) and `pred_custom_points`
        (predicted_residual + baseline_col -- what to actually compare
        against target_col).
    """
    valid = df[df[baseline_col].notna()].copy()
    residual_col = "_residual_target"
    valid[residual_col] = valid[target_col] - valid[baseline_col]

    preds = walk_forward_predict(
        valid, position, feature_cols=feature_cols, target_col=residual_col,
        warmup_weeks=warmup_weeks, eval_min_season=eval_min_season, lgb_params=lgb_params,
    )
    preds = preds.rename(columns={"pred_model_a": "pred_residual", residual_col: "residual_target"})
    preds = preds.merge(
        valid[["player_id", "season", "week", target_col, baseline_col]],
        on=["player_id", "season", "week"], how="left",
    )
    preds["pred_custom_points"] = preds["pred_residual"] + preds[baseline_col]
    return preds


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _quantile_col(q: float) -> str:
    return f"pred_q{round(q * 100)}"


def walk_forward_predict_quantile(
    df: pd.DataFrame,
    position: str,
    feature_cols: list[str] = FEATURE_COLUMNS,
    target_col: str = "custom_points",
    quantiles: tuple[float, ...] = QUANTILES,
    warmup_weeks: int = 4,
    eval_min_season: int | None = None,
    lgb_params: dict | None = None,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward prediction, LightGBM quantile regression
    (`objective='quantile'`), one FRESH model per quantile per fold -- same
    fold structure as walk_forward_predict (train on every row strictly
    before the fold, predict that fold, no shuffling, no KFold), just fit
    len(quantiles) times per fold instead of once.

    LightGBM fits each quantile as an INDEPENDENT model -- there's no
    constraint tying pred_q25 <= pred_q50 <= pred_q75 across the separate
    fits, so on any given row the predictions can come out non-monotonic
    ("quantile crossing"). This function does not fix that; check with
    quantile_crossing_rate() and fix with fix_quantile_crossing() if it
    occurs.

    Returns:
        Wide frame: id_cols, target_col, and one pred_q{int(q*100)} column
        per quantile (e.g. pred_q10, pred_q25, ...) -- one row per
        (player, evaluated week), out-of-sample.
    """
    pos_df = df[df["position"] == position].sort_values(["season", "week"]).reset_index(drop=True)
    folds = chronological_folds(pos_df, warmup_weeks, eval_min_season)
    pos_df = _cast_categoricals(pos_df, feature_cols)

    base_params = {"objective": "quantile", "verbosity": -1, "random_state": 42}
    if lgb_params:
        base_params.update(lgb_params)

    id_cols = [c for c in ["player_id", "player_display_name", "season", "week"] if c in pos_df.columns]
    q_cols = [_quantile_col(q) for q in quantiles]

    preds = []
    for season, week in folds:
        train_mask = (pos_df["season"] < season) | ((pos_df["season"] == season) & (pos_df["week"] < week))
        test_mask = (pos_df["season"] == season) & (pos_df["week"] == week)
        train = pos_df.loc[train_mask]
        test = pos_df.loc[test_mask]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue

        result = test[id_cols + [target_col]].copy()
        for q, col in zip(quantiles, q_cols):
            params = dict(base_params, alpha=q)
            model = lgb.LGBMRegressor(**params)
            model.fit(train[feature_cols], train[target_col])
            result[col] = model.predict(test[feature_cols])
        preds.append(result)

    if not preds:
        return pd.DataFrame(columns=id_cols + [target_col] + q_cols)
    return pd.concat(preds, ignore_index=True)


def quantile_crossing_rate(preds: pd.DataFrame, quantiles: tuple[float, ...] = QUANTILES) -> dict:
    """
    Fraction of rows where the independently-fit quantile predictions are
    NOT monotonically non-decreasing (pred_q10 <= pred_q25 <= ... <=
    pred_q90). Crossing is expected occasionally with independent quantile
    fits; this just measures how often.
    """
    cols = [_quantile_col(q) for q in quantiles]
    vals = preds[cols].to_numpy(dtype=float)
    is_monotonic = np.all(np.diff(vals, axis=1) >= 0, axis=1)
    return {
        "n_rows": len(preds),
        "n_crossed": int((~is_monotonic).sum()),
        "crossing_rate": float((~is_monotonic).mean()) if len(preds) else np.nan,
    }


def fix_quantile_crossing(preds: pd.DataFrame, quantiles: tuple[float, ...] = QUANTILES) -> pd.DataFrame:
    """
    Rearrangement fix for quantile crossing (Chernozhukov, Fernandez-Val &
    Galichon 2010): sort each row's independently-fit quantile predictions
    into non-decreasing order and reassign them back to the same columns.
    This does not change any column's marginal calibration -- it's the
    same multiset of predicted values per row, just reordered so pred_q10
    is always <= pred_q90 -- it only removes crossing, it doesn't pull the
    values toward each other or shrink the interval.
    """
    out = preds.copy()
    cols = [_quantile_col(q) for q in quantiles]
    sorted_vals = np.sort(out[cols].to_numpy(dtype=float), axis=1)
    for i, col in enumerate(cols):
        out[col] = sorted_vals[:, i]
    return out


def quantile_coverage(
    preds: pd.DataFrame, target_col: str = "custom_points", quantiles: tuple[float, ...] = QUANTILES,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Calibration check: for each quantile q, the fraction of ACTUAL outcomes
    at or below pred_q{q} ("actual_coverage") should be close to q itself
    ("target_coverage") if the model is well-calibrated. A quantile whose
    actual_coverage is far from its target is untrustworthy for floor/
    ceiling calls regardless of how good the point-estimate metrics look.

    `suffix` reads pred_q{q}{suffix} instead of pred_q{q} -- e.g.
    suffix="_cqr" to check coverage of conformal-widened bounds (see
    apply_conformal_widening) using the exact same tie-breaking
    convention as the unadjusted check, so a before/after comparison
    never silently reintroduces the '<' vs '<=' inconsistency fixed here.
    """
    rows = []
    for q in quantiles:
        col = _quantile_col(q) + suffix
        valid = preds[preds[col].notna()]
        below = (valid[target_col] <= valid[col]).mean() if len(valid) else np.nan
        rows.append({
            "quantile": q, "target_coverage": q, "actual_coverage": below, "n": len(valid),
        })
    return pd.DataFrame(rows)


def quantile_interval_coverage(
    preds: pd.DataFrame, target_col: str = "custom_points", lower_q: float = 0.1, upper_q: float = 0.9,
    suffix: str = "",
) -> dict:
    """
    Three-way split for the interval between lower_q and upper_q (e.g. the
    10th-90th, meant to cover ~80% of outcomes): fraction of actuals below
    the lower prediction, within the interval, and above the upper
    prediction. Rows are dropped if either bound is null.

    Uses the same '<=' convention as quantile_coverage() for "at or below"
    a quantile, so below_q{lower} here always matches
    quantile_coverage()'s actual_coverage for the same quantile exactly.
    This isn't a rounding nuance: a target with real point-mass at a
    specific value (e.g. TE's custom_points is exactly 0.0 for ~19% of
    player-weeks -- DNP/inactive/no recorded stats) produces enough
    predicted-vs-actual ties at a quantile near that value that '<=' vs
    '<' changes the reported figure by double digits, not fractions of a
    percent.

    `suffix`: same purpose as in quantile_coverage -- pass "_cqr" to
    report coverage of the conformal-widened bounds.
    """
    lower_col, upper_col = _quantile_col(lower_q) + suffix, _quantile_col(upper_q) + suffix
    valid = preds[preds[lower_col].notna() & preds[upper_col].notna()]
    below = (valid[target_col] <= valid[lower_col]).mean() if len(valid) else np.nan
    above = (valid[target_col] > valid[upper_col]).mean() if len(valid) else np.nan
    within = (1 - below - above) if len(valid) else np.nan
    return {
        "lower_q": lower_q, "upper_q": upper_q, "n": len(valid),
        "target_within": upper_q - lower_q,
        f"below_q{round(lower_q * 100)}": below,
        "within_interval": within,
        f"above_q{round(upper_q * 100)}": above,
    }


# ==========================================================================
# CONFORMALIZED QUANTILE REGRESSION (CQR) -- Phase 6.5 prerequisite
# ==========================================================================
def conformity_scores(
    calib_preds: pd.DataFrame, target_col: str, lower_q: float, upper_q: float
) -> np.ndarray:
    """
    CQR conformity score (Romano, Patterson & Candes 2019) for the
    interval [lower_q, upper_q], one per calibration row: how far the
    ACTUAL outcome fell outside the PREDICTED interval.
        E_i = max(pred_lower_i - y_i, y_i - pred_upper_i)
    Positive if y_i fell outside the interval on either side; a negative
    margin if y_i fell inside it. calib_preds should already have
    quantile crossing fixed (fix_quantile_crossing) so pred_lower_i <=
    pred_upper_i for every row -- this formula doesn't require it, but a
    crossed row's "interval" isn't a real interval to begin with.

    No boolean tie-breaking here (unlike quantile_coverage): this is a
    continuous max() of two differences, so point-mass in the target
    (e.g. TE's ~19% exact-zero custom_points) doesn't create a </<=
    ambiguity in the score itself -- it just means many calibration rows
    can land at exactly E_i = 0 when both the prediction and the actual
    are 0, which is a real, correctly-computed conformity score, not a
    tie to break.
    """
    lower_col, upper_col = _quantile_col(lower_q), _quantile_col(upper_q)
    return np.maximum(
        (calib_preds[lower_col] - calib_preds[target_col]).to_numpy(),
        (calib_preds[target_col] - calib_preds[upper_col]).to_numpy(),
    )


def conformal_quantile(scores: np.ndarray, target_coverage: float) -> float:
    """
    The finite-sample-corrected empirical quantile of calibration
    conformity scores that gives CQR its coverage GUARANTEE at finite n
    (Romano, Patterson & Candes 2019): the ceil((n+1) * target_coverage)
    -th order statistic, clipped to the largest observed score if the
    calibration set is too small for that rank to exist. A naive
    np.quantile(scores, target_coverage) uses n instead of (n+1) and
    under-widens slightly, losing the finite-sample guarantee (the two
    converge as n grows, so this only matters when the calibration set
    is small).
    """
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    if n == 0:
        raise ValueError("conformal_quantile: empty calibration score set")
    rank = min(int(np.ceil((n + 1) * target_coverage)), n)
    return float(sorted_scores[rank - 1])


def apply_conformal_widening(
    preds: pd.DataFrame, lower_q: float, upper_q: float, widen_by: float
) -> pd.DataFrame:
    """
    Applies the CQR adjustment: subtract widen_by from the lower bound,
    add it to the upper bound, for every row -- a single, constant,
    additive correction (not scaled by the prediction's own magnitude).
    Writes NEW pred_q{lower}_cqr / pred_q{upper}_cqr columns; the
    original pred_q{lower}/pred_q{upper} columns are left untouched so
    before/after can be compared on the same frame.
    """
    out = preds.copy()
    lower_col, upper_col = _quantile_col(lower_q), _quantile_col(upper_q)
    out[f"{lower_col}_cqr"] = out[lower_col] - widen_by
    out[f"{upper_col}_cqr"] = out[upper_col] + widen_by
    return out


def interval_breach_by_prediction_bucket(
    calib_preds: pd.DataFrame, target_col: str, lower_q: float, upper_q: float, n_bins: int = 3,
) -> pd.DataFrame:
    """
    Diagnostic for whether a single GLOBAL widening constant is an
    appropriate fix, or whether undercoverage is concentrated at one end
    of the prediction range: buckets the calibration set into n_bins
    equal-count groups by predicted median (pred_q50), and reports the
    RAW (pre-CQR) breach rate below lower_q and above upper_q within
    each bucket. Similar breach rates across buckets means one global
    constant is a reasonable fit; a breach rate concentrated in one
    bucket means a single constant over-widens easy-to-predict weeks and
    under-widens hard ones.
    """
    lower_col, upper_col = _quantile_col(lower_q), _quantile_col(upper_q)
    out = calib_preds.copy()
    out["_bucket"] = pd.qcut(out["pred_q50"], n_bins, labels=[f"bucket_{i + 1}_of_{n_bins}" for i in range(n_bins)])
    rows = []
    for bucket, g in out.groupby("_bucket", observed=True):
        below = (g[target_col] <= g[lower_col]).mean()
        above = (g[target_col] > g[upper_col]).mean()
        rows.append({
            "bucket": bucket, "n": len(g),
            "pred_q50_min": g["pred_q50"].min(), "pred_q50_max": g["pred_q50"].max(),
            "below": below, "within": 1 - below - above, "above": above,
        })
    return pd.DataFrame(rows)


def _cast_categoricals(pos_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """LightGBM's sklearn wrapper only accepts int/float/bool columns as-is;
    string columns (roof, surface) need pandas 'category' dtype for its
    native categorical handling."""
    pos_df = pos_df.copy()
    for col in feature_cols:
        if pos_df[col].dtype == object or pd.api.types.is_string_dtype(pos_df[col]):
            pos_df[col] = pos_df[col].astype("category")
    return pos_df


def _rank_features_by_shap(
    X: pd.DataFrame, y: pd.Series, feature_cols: list[str], lgb_params: dict
) -> pd.Series:
    """Train one model on (X, y) and return feature_cols ranked by mean
    |SHAP value|, descending, computed on the SAME data the model trained
    on (X) -- not on any held-out fold. This is the selector step of
    walk_forward_predict_selected: a ranking derived only from information
    available going into that fold's training window."""
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X[feature_cols], y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[feature_cols])
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    return mean_abs_shap.sort_values(ascending=False)


def shap_top_features_median_model(
    df: pd.DataFrame,
    position: str,
    feature_cols: list[str] = FEATURE_COLUMNS,
    target_col: str = "custom_points",
    top_n: int = 20,
    lgb_params: dict | None = None,
) -> pd.Series:
    """
    SHAP feature-importance ranking for ONE quantile-median (alpha=0.5)
    model trained on this position's entire dataset -- a final
    explainability snapshot, not a walk-forward accuracy evaluation. Uses
    every row on purpose (same reasoning as stable_feature_ranking:
    explaining the final model is a one-time setup question, not part of
    the walk-forward accuracy measurement, so there's no leakage concern
    in using all available data for it).

    Returns the top_n features by mean |SHAP|, descending.
    """
    pos_df = df[df["position"] == position].reset_index(drop=True)
    pos_df = _cast_categoricals(pos_df, feature_cols)
    params = {"objective": "quantile", "alpha": 0.5, "verbosity": -1, "random_state": 42}
    if lgb_params:
        params.update(lgb_params)
    ranking = _rank_features_by_shap(pos_df, pos_df[target_col], feature_cols, params)
    return ranking.head(top_n)


def walk_forward_predict_selected(
    df: pd.DataFrame,
    position: str,
    all_feature_cols: list[str],
    n_features: int,
    target_col: str = "custom_points",
    warmup_weeks: int = 4,
    lgb_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward prediction with feature selection derived FRESH inside
    each fold, from that fold's training data only -- fixes the look-ahead
    bias in a single dataset-wide SHAP ranking (an early fold's "top N"
    would otherwise be chosen partly using information from weeks that
    fold couldn't have seen yet).

    Per fold: (1) train a selector model on all_feature_cols using that
    fold's training rows, (2) rank by mean |SHAP| on those SAME training
    rows, (3) keep the top n_features, (4) train a SECOND, fresh model
    using ONLY those n_features on the same training rows, (5) predict the
    test week with that second model. Two fits per fold, not one --
    correctness over speed, and it's still fast at this data size.

    Returns:
        (preds, selections) --
        preds: same shape as walk_forward_predict's return.
        selections: long frame of (season, week, feature, shap_rank) for
        every fold -- the raw material for picking one final, stable
        feature list (e.g. by average rank across folds) if this feature
        count turns out to win the ablation.
    """
    pos_df = df[df["position"] == position].sort_values(["season", "week"]).reset_index(drop=True)
    folds = chronological_folds(pos_df, warmup_weeks)
    pos_df = _cast_categoricals(pos_df, all_feature_cols)

    params = {"objective": "regression", "verbosity": -1, "random_state": 42}
    if lgb_params:
        params.update(lgb_params)

    id_cols = [c for c in ["player_id", "player_display_name", "season", "week"] if c in pos_df.columns]

    preds = []
    selections = []
    for season, week in folds:
        train_mask = (pos_df["season"] < season) | ((pos_df["season"] == season) & (pos_df["week"] < week))
        test_mask = (pos_df["season"] == season) & (pos_df["week"] == week)
        train = pos_df.loc[train_mask]
        test = pos_df.loc[test_mask]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            continue

        ranking = _rank_features_by_shap(train, train[target_col], all_feature_cols, params)
        selected = ranking.index[:n_features].tolist()
        selections.append(pd.DataFrame({
            "season": season, "week": week,
            "feature": ranking.index[:n_features],
            "shap_rank": range(1, len(selected) + 1),
        }))

        model = lgb.LGBMRegressor(**params)
        model.fit(train[selected], train[target_col])
        pred = model.predict(test[selected])

        result = test[id_cols + [target_col]].copy()
        result["pred_model_a"] = pred
        preds.append(result)

    preds_df = (
        pd.concat(preds, ignore_index=True) if preds
        else pd.DataFrame(columns=id_cols + [target_col, "pred_model_a"])
    )
    selections_df = (
        pd.concat(selections, ignore_index=True) if selections
        else pd.DataFrame(columns=["season", "week", "feature", "shap_rank"])
    )
    return preds_df, selections_df


def stable_feature_ranking(selections: pd.DataFrame, all_feature_cols: list[str]) -> pd.Series:
    """
    Aggregate walk_forward_predict_selected's per-fold rankings into ONE
    final ranking, for picking a fixed feature list to hard-code -- average
    rank across every fold, treating a feature that never made a fold's
    top-N as ranked last (len(all_feature_cols)) in that fold rather than
    leaving it out of the average entirely. Lower is better.

    Deliberately uses ALL folds' worth of information (unlike the
    per-fold-selected ablation runs, which each fold sees only its own
    training data) -- once feature count is decided, choosing the final,
    permanent feature LIST is a one-time setup decision, not part of the
    walk-forward accuracy evaluation, so it's fine and standard practice
    to use everything available for it.
    """
    worst_rank = len(all_feature_cols)
    pivot = selections.pivot_table(index="feature", columns=["season", "week"], values="shap_rank")
    pivot = pivot.reindex(all_feature_cols)
    avg_rank = pivot.fillna(worst_rank).mean(axis=1)
    return avg_rank.sort_values()


# ==========================================================================
# FINAL (NO-HOLDOUT) MODELS -- Phase 8: persisted across weekly inference runs
# ==========================================================================
def train_final_models(
    df: pd.DataFrame,
    feature_cols: dict[str, list[str]] = FEATURE_COLUMNS_BY_POSITION,
    target_col: str = "custom_points",
    positions: tuple[str, ...] = POSITIONS,
    quantile_alphas: tuple[float, ...] = (0.10, 0.90),
) -> dict:
    """
    One point-regression model and one quantile model PER alpha in
    quantile_alphas (default q10/q90, matching src/export.py's floor/
    ceiling; scripts/retrain.py passes the full 5-point
    0.10/0.25/0.50/0.75/0.90 set too, for src/simulate.py's per-player
    distributions) per position, each trained on EVERY row of df for that
    position -- no fold held out, the same "nothing to hold out against a
    week that hasn't been played yet" reasoning
    src/export.py::predict_target_week already uses, generalized so the
    resulting model objects can be PERSISTED (see src/artifacts.py) and
    reused across many weekly inference runs instead of being retrained
    inline on every single export.

    `feature_cols` is PER-POSITION ({position: [...]}, defaulting to
    FEATURE_COLUMNS_BY_POSITION) rather than one shared list -- see that
    constant's own comment for why (Team Tendencies helps QB, hurts TE; a
    single list couldn't have it both ways). Categorical dtypes are cast
    ONCE across the union of every position's own columns in THIS CALL's
    `feature_cols` (not the module-level ALL_FEATURE_COLUMNS, so a caller
    passing its own custom feature_cols -- e.g. a test, or a SHAP
    ablation -- doesn't get cast against unrelated real column names it
    never asked for), before the per-position split -- same "cast once
    before splitting" reasoning as walk_forward_predict.

    Returns:
        {position: {"point": LGBMRegressor, "quantiles": {alpha: LGBMRegressor}}}
        -- positions with no rows in df, or missing from `feature_cols`,
        are silently skipped (mirrors predict_target_week's per-position
        `if pos_test.empty: continue`).
    """
    all_cols = sorted({col for cols in feature_cols.values() for col in cols})
    pos_df = _cast_categoricals(df, all_cols)
    models: dict = {}
    for position in positions:
        cols = feature_cols.get(position)
        if not cols:
            continue
        rows = pos_df[pos_df["position"] == position]
        if rows.empty:
            continue
        reg_model = lgb.LGBMRegressor(objective="regression", verbosity=-1, random_state=42)
        reg_model.fit(rows[cols], rows[target_col])

        q_models = {}
        for alpha in quantile_alphas:
            q_model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, verbosity=-1, random_state=42)
            q_model.fit(rows[cols], rows[target_col])
            q_models[alpha] = q_model

        models[position] = {"point": reg_model, "quantiles": q_models}
    return models


def predict_with_models(
    test_df: pd.DataFrame,
    models: dict,
    cqr_widen_by_10_90: dict[str, float],
    feature_cols: dict[str, list[str]] = FEATURE_COLUMNS_BY_POSITION,
    lower_alpha: float = 0.10,
    upper_alpha: float = 0.90,
) -> pd.DataFrame:
    """
    Predict point/floor/ceiling for test_df's rows using ALREADY-TRAINED
    per-position models (from train_final_models, fresh or loaded from a
    saved artifact) and CQR widening constants -- the shared prediction
    step behind both a fresh train-then-predict call and a
    load-artifact-then-predict call, so a train-and-predict-immediately
    run and a load-artifact-and-predict-later run can never silently
    diverge in how floor/ceiling get built.

    `feature_cols` is PER-POSITION, same shape/reasoning as
    train_final_models -- and MUST be whichever dict actually trained
    `models` (predict_target_week_from_artifact passes the artifact's own
    `feature_columns`, never this module's default, so a weekly run always
    matches what retrain.yml actually trained even if this constant is
    edited later).

    Same floor <= point <= ceiling clipping as
    src/export.py::predict_target_week: point comes from a SEPARATE model
    than floor/ceiling, so nothing mathematically guarantees the two
    agree on their own.

    Returns: player_id, position, point, floor, ceiling -- one row per
    (position, player) present in test_df AND in `models`.
    """
    all_cols = sorted({col for cols in feature_cols.values() for col in cols})
    pos_df = _cast_categoricals(test_df, all_cols)
    results = []
    for position, bundle in models.items():
        rows = pos_df[pos_df["position"] == position]
        if rows.empty:
            continue
        cols = feature_cols[position]

        point_raw = bundle["point"].predict(rows[cols])
        q_lower = bundle["quantiles"][lower_alpha].predict(rows[cols])
        q_upper = bundle["quantiles"][upper_alpha].predict(rows[cols])

        floor_raw = np.minimum(q_lower, q_upper)
        ceiling_raw = np.maximum(q_lower, q_upper)
        widen_by = cqr_widen_by_10_90[position]
        floor = floor_raw - widen_by
        ceiling = ceiling_raw + widen_by
        point = np.clip(point_raw, floor, ceiling)

        results.append(pd.DataFrame({
            "player_id": rows["player_id"].to_numpy(),
            "position": position,
            "point": point,
            "floor": floor,
            "ceiling": ceiling,
        }))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["player_id", "position", "point", "floor", "ceiling"]
    )


SIMULATION_QUANTILE_ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)


def predict_quantiles_with_models(
    test_df: pd.DataFrame,
    models: dict,
    cqr_widen_by_10_90: dict[str, float],
    cqr_widen_by_25_75: dict[str, float],
    feature_cols: dict[str, list[str]] = FEATURE_COLUMNS_BY_POSITION,
) -> pd.DataFrame:
    """
    Predicts the full 5-point CQR-calibrated distribution
    src/simulate.py's sample_player_week() needs (pred_q10_cqr,
    pred_q25_cqr, pred_q50, pred_q75_cqr, pred_q90_cqr) for every row in
    test_df, using per-position models already trained with AT LEAST
    SIMULATION_QUANTILE_ALPHAS (train_final_models with that alpha tuple,
    which scripts/retrain.py passes).

    `feature_cols` is PER-POSITION, same shape/reasoning as
    predict_with_models -- MUST be whichever dict actually trained
    `models` (weekly_update.py passes the artifact's own `feature_columns`).

    Mirrors predict_with_models's shape (same models dict, same
    _cast_categoricals/per-position loop) but for the simulator's input
    contract instead of the dashboard's floor/point/ceiling. The 10-90
    and 25-75 pairs are CQR-widened by their OWN, separately-derived
    constants (see PROJECT_CONTEXT.md's CQR section -- the two pairs were
    calibrated independently and use different widening amounts); q50 is
    returned as-is, untouched by construction (there's no interval around
    a single point to widen). Crossing WITHIN each pair is fixed with a
    min/max swap before widening -- src/simulate.py's own
    _ensure_monotonic_quantiles() still re-sorts all 5 points defensively
    afterward (crossing BETWEEN pairs, e.g. a widened q25 landing below a
    widened q10, can still happen and is that function's job to catch,
    not this one's).

    Raises if a position's model bundle is missing any required alpha --
    a silently-absent quantile would otherwise surface much later as a
    confusing KeyError inside sample_player_week.

    Returns: player_id, position, pred_q10_cqr, pred_q25_cqr, pred_q50,
    pred_q75_cqr, pred_q90_cqr -- one row per (position, player) present
    in test_df AND in `models`.
    """
    output_cols = ["player_id", "position", "pred_q10_cqr", "pred_q25_cqr", "pred_q50", "pred_q75_cqr", "pred_q90_cqr"]
    all_cols = sorted({col for cols in feature_cols.values() for col in cols})
    pos_df = _cast_categoricals(test_df, all_cols)
    results = []
    for position, bundle in models.items():
        rows = pos_df[pos_df["position"] == position]
        if rows.empty:
            continue
        cols = feature_cols[position]

        missing = [a for a in SIMULATION_QUANTILE_ALPHAS if a not in bundle["quantiles"]]
        if missing:
            raise KeyError(f"predict_quantiles_with_models: {position} model is missing quantile alphas {missing}")

        q10 = bundle["quantiles"][0.10].predict(rows[cols])
        q25 = bundle["quantiles"][0.25].predict(rows[cols])
        q50 = bundle["quantiles"][0.50].predict(rows[cols])
        q75 = bundle["quantiles"][0.75].predict(rows[cols])
        q90 = bundle["quantiles"][0.90].predict(rows[cols])

        lo_1090, hi_1090 = np.minimum(q10, q90), np.maximum(q10, q90)
        lo_2575, hi_2575 = np.minimum(q25, q75), np.maximum(q25, q75)

        widen_1090 = cqr_widen_by_10_90[position]
        widen_2575 = cqr_widen_by_25_75[position]

        results.append(pd.DataFrame({
            "player_id": rows["player_id"].to_numpy(),
            "position": position,
            "pred_q10_cqr": lo_1090 - widen_1090,
            "pred_q25_cqr": lo_2575 - widen_2575,
            "pred_q50": q50,
            "pred_q75_cqr": hi_2575 + widen_2575,
            "pred_q90_cqr": hi_1090 + widen_1090,
        }))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=output_cols)


# ==========================================================================
# METRICS
# ==========================================================================
def _metrics(actual: pd.Series, pred: pd.Series) -> dict:
    mask = actual.notna() & pred.notna()
    a, p = actual[mask], pred[mask]
    if len(a) < 2:
        return {"n": len(a), "mae": np.nan, "rmse": np.nan, "spearman": np.nan}
    mae = (a - p).abs().mean()
    rmse = np.sqrt(((a - p) ** 2).mean())
    corr, _ = spearmanr(a, p)
    return {"n": len(a), "mae": mae, "rmse": rmse, "spearman": corr}


def evaluate_position(
    preds: pd.DataFrame, target_col: str, baseline_cols: dict[str, str]
) -> pd.DataFrame:
    """
    For one position's walk-forward predictions (`preds`, from
    walk_forward_predict merged with baseline columns): report the model's
    full-sample metrics, then for each baseline, metrics for BOTH the
    model and that baseline computed on the SAME rows (wherever the
    baseline has a non-null value) -- a fair head-to-head, not a
    full-sample model number quietly compared against a baseline's
    partial-coverage number.

    Args:
        preds: must have target_col, 'pred_model_a', and every column
            named in baseline_cols.values().
        target_col: the actual outcome column (custom_points).
        baseline_cols: {display_name: column_name} for each baseline.

    Returns:
        Tidy frame: rows = ['model (full sample)'] + one row per
        (baseline, ['model', baseline_name]), columns = n, mae, rmse,
        spearman.
    """
    rows = []

    full = _metrics(preds[target_col], preds["pred_model_a"])
    rows.append({"comparison": "model (full sample)", "series": "model_a", **full})

    for display_name, col in baseline_cols.items():
        if col not in preds.columns:
            rows.append({"comparison": display_name, "series": "model_a", "n": 0,
                         "mae": np.nan, "rmse": np.nan, "spearman": np.nan})
            rows.append({"comparison": display_name, "series": display_name, "n": 0,
                         "mae": np.nan, "rmse": np.nan, "spearman": np.nan})
            continue
        common = preds[preds[col].notna()]
        model_on_common = _metrics(common[target_col], common["pred_model_a"])
        baseline_metrics = _metrics(common[target_col], common[col])
        rows.append({"comparison": display_name, "series": "model_a", **model_on_common})
        rows.append({"comparison": display_name, "series": display_name, **baseline_metrics})

    return pd.DataFrame(rows)
