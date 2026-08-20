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
        (not lagged) is correct, not a leak.
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

from src.usage import CONTEXT_OUTPUT_COLUMNS, ROLLING_OUTPUT_COLUMNS

logger = logging.getLogger(__name__)

POSITIONS = ("QB", "RB", "WR", "TE")

FEATURE_COLUMNS = list(ROLLING_OUTPUT_COLUMNS) + list(CONTEXT_OUTPUT_COLUMNS)

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
