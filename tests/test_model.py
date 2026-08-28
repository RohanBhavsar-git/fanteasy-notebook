"""
Correctness test for src/model.py's walk-forward validation.

Unlike tests/test_no_leakage.py (which verifies individual FEATURE columns
never see week N's own or future data), this checks the WALK-FORWARD SPLIT
ITSELF: for every fold, the rows LightGBM trains on must be strictly earlier
in chronological order than the fold being predicted. If that ever breaks --
a sort that silently doesn't take effect, an off-by-one in the mask -- every
downstream metric becomes meaningless, so this is worth locking down
directly rather than trusting it by inspection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (  # noqa: E402
    apply_conformal_widening,
    chronological_folds,
    conformal_quantile,
    conformity_scores,
    fix_quantile_crossing,
    interval_breach_by_prediction_bucket,
    predict_quantiles_with_models,
    predict_with_models,
    quantile_coverage,
    quantile_crossing_rate,
    quantile_interval_coverage,
    train_final_models,
    walk_forward_predict,
    walk_forward_predict_quantile,
)


def _synthetic_position_df(n_players=8, seasons=(2024, 2025), weeks=range(1, 19)):
    rng = np.random.default_rng(0)
    rows = []
    for season in seasons:
        for week in weeks:
            for p in range(n_players):
                rows.append({
                    "player_id": f"P{p}",
                    "player_display_name": f"Player {p}",
                    "position": "WR",
                    "season": season,
                    "week": week,
                    "feat_a": rng.normal(),
                    "feat_b": rng.normal(),
                    "custom_points": rng.normal(10, 5),
                })
    return pd.DataFrame(rows)


def test_chronological_folds_skips_only_first_season_warmup():
    df = _synthetic_position_df()
    folds = chronological_folds(df, warmup_weeks=4)

    assert (2024, 1) not in folds
    assert (2024, 4) not in folds
    assert (2024, 5) in folds
    assert (2025, 1) in folds  # NOT re-warmed-up for the second season
    assert (2025, 18) in folds


def test_walk_forward_train_set_never_includes_current_or_future_weeks(monkeypatch):
    """
    Patches lgb.LGBMRegressor with a fake that records exactly which
    (season, week) rows it was trained on for each fold, then asserts none
    of them are >= the fold being predicted. This tests the SPLIT, not the
    model -- a fake regressor makes the assertion about training data
    composition, not prediction quality.
    """
    df = _synthetic_position_df()

    # Wrap the module-level lgb.LGBMRegressor reference with a fake that
    # just records which row indices it was trained on for each fold.
    import src.model as model_module

    captured = []

    class _RecordingModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            captured.append(X.index)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(model_module.lgb, "LGBMRegressor", _RecordingModel)

    feature_cols = ["feat_a", "feat_b"]
    preds = walk_forward_predict(df, "WR", feature_cols=feature_cols, warmup_weeks=4)

    pos_df = df[df["position"] == "WR"].sort_values(["season", "week"]).reset_index(drop=True)
    folds = chronological_folds(pos_df, warmup_weeks=4)

    assert len(captured) == len(folds), "expected one fit() call per evaluated fold"

    for (season, week), train_index in zip(folds, captured):
        train_rows = pos_df.loc[train_index]
        too_late = train_rows[
            (train_rows["season"] > season)
            | ((train_rows["season"] == season) & (train_rows["week"] >= week))
        ]
        assert too_late.empty, (
            f"fold ({season}, wk{week}): training set contains "
            f"{len(too_late)} rows from that week or later"
        )

    assert not preds.empty


def test_walk_forward_predict_quantile_train_set_never_includes_current_or_future_weeks(monkeypatch):
    """
    Same no-lookahead check as
    test_walk_forward_train_set_never_includes_current_or_future_weeks, but
    for the quantile walk-forward path, which fits len(quantiles) separate
    models per fold instead of one -- every one of those fits must still
    see only strictly-earlier rows.
    """
    df = _synthetic_position_df()
    import src.model as model_module

    captured = []

    class _RecordingModel:
        def __init__(self, **kwargs):
            self.alpha = kwargs.get("alpha")

        def fit(self, X, y):
            captured.append(X.index)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(model_module.lgb, "LGBMRegressor", _RecordingModel)

    feature_cols = ["feat_a", "feat_b"]
    quantiles = (0.1, 0.5, 0.9)
    preds = walk_forward_predict_quantile(
        df, "WR", feature_cols=feature_cols, quantiles=quantiles, warmup_weeks=4
    )

    pos_df = df[df["position"] == "WR"].sort_values(["season", "week"]).reset_index(drop=True)
    folds = chronological_folds(pos_df, warmup_weeks=4)

    assert len(captured) == len(folds) * len(quantiles), (
        "expected one fit() call per (fold, quantile) pair"
    )

    fold_for_each_call = [folds[i // len(quantiles)] for i in range(len(captured))]
    for (season, week), train_index in zip(fold_for_each_call, captured):
        train_rows = pos_df.loc[train_index]
        too_late = train_rows[
            (train_rows["season"] > season)
            | ((train_rows["season"] == season) & (train_rows["week"] >= week))
        ]
        assert too_late.empty, (
            f"fold ({season}, wk{week}): training set contains "
            f"{len(too_late)} rows from that week or later"
        )

    assert not preds.empty
    for q in quantiles:
        assert f"pred_q{round(q * 100)}" in preds.columns


def test_quantile_crossing_rate_detects_and_fix_resolves_it():
    preds = pd.DataFrame({
        "pred_q10": [1.0, 5.0, 2.0],
        "pred_q25": [2.0, 4.0, 3.0],   # row 1 (index 1): q25 < q10 -- crossed
        "pred_q50": [3.0, 6.0, 4.0],
        "pred_q75": [4.0, 7.0, 5.0],
        "pred_q90": [5.0, 8.0, 6.0],
    })
    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)

    result = quantile_crossing_rate(preds, quantiles=quantiles)
    assert result["n_rows"] == 3
    assert result["n_crossed"] == 1
    assert result["crossing_rate"] == pytest.approx(1 / 3)

    fixed = fix_quantile_crossing(preds, quantiles=quantiles)
    fixed_result = quantile_crossing_rate(fixed, quantiles=quantiles)
    assert fixed_result["n_crossed"] == 0

    # The rearrangement is a per-row sort, not a value change -- the same
    # multiset of predicted values must still be present in each row.
    cols = [f"pred_q{round(q * 100)}" for q in quantiles]
    for i in preds.index:
        assert sorted(preds.loc[i, cols].tolist()) == sorted(fixed.loc[i, cols].tolist())


def test_quantile_coverage_matches_known_calibration():
    """
    Builds actuals as a known standard normal sample and predictions as
    the TRUE normal quantiles (via scipy) -- a perfectly calibrated
    quantile forecast -- and checks quantile_coverage() recovers coverage
    close to the target for each quantile.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(0)
    actual = rng.normal(size=5000)
    quantiles = (0.1, 0.5, 0.9)
    preds = pd.DataFrame({"custom_points": actual})
    for q in quantiles:
        preds[f"pred_q{round(q * 100)}"] = norm.ppf(q)

    coverage = quantile_coverage(preds, target_col="custom_points", quantiles=quantiles)
    for _, row in coverage.iterrows():
        assert row["actual_coverage"] == pytest.approx(row["target_coverage"], abs=0.02)


def test_quantile_interval_coverage_matches_quantile_coverage_at_the_boundary():
    """
    A target distribution with real point-mass at a specific value (e.g.
    TE's custom_points is exactly 0.0 for ~19% of player-weeks) makes
    predicted-vs-actual ties at a quantile common, not a rounding-scale
    effect -- '<=' vs '<' at that boundary can move the reported figure
    by double digits. quantile_interval_coverage()'s below_q{lower} must
    use the same '<=' convention as quantile_coverage(), so the two never
    silently disagree about the same boundary the way an earlier version
    of this module did (it used strict '<', discovered via a ~13-point
    gap between the two on real TE data).
    """
    preds = pd.DataFrame({
        "custom_points": [0.0, 0.0, 0.0, 5.0, 10.0],
        "pred_q10": [0.0, 0.0, 0.0, 0.0, 0.0],  # ties actual==pred at the zero-mass rows
        "pred_q90": [8.0, 8.0, 8.0, 8.0, 8.0],
    })
    quantiles = (0.1, 0.9)

    coverage = quantile_coverage(preds, target_col="custom_points", quantiles=quantiles)
    coverage_at_q10 = coverage.loc[coverage["quantile"] == 0.1, "actual_coverage"].item()
    coverage_at_q90 = coverage.loc[coverage["quantile"] == 0.9, "actual_coverage"].item()

    interval = quantile_interval_coverage(preds, target_col="custom_points", lower_q=0.1, upper_q=0.9)

    assert interval["below_q10"] == pytest.approx(coverage_at_q10)
    assert interval["above_q90"] == pytest.approx(1 - coverage_at_q90)
    assert interval["below_q10"] + interval["within_interval"] + interval["above_q90"] == pytest.approx(1.0)


def test_conformity_scores_formula():
    preds = pd.DataFrame({
        "custom_points": [5.0, 5.0, 5.0],
        "pred_q10": [2.0, 6.0, 2.0],   # row1: y > lower (inside on this side)
        "pred_q90": [8.0, 8.0, 4.0],   # row0: inside both; row1: below lower; row2: above upper
    })
    scores = conformity_scores(preds, target_col="custom_points", lower_q=0.1, upper_q=0.9)
    # row 0: max(2-5, 5-8) = max(-3, -3) = -3 (inside, margin 3)
    # row 1: max(6-5, 5-8) = max(1, -3) = 1 (fell below the lower bound by 1)
    # row 2: max(2-5, 5-4) = max(-3, 1) = 1 (fell above the upper bound by 1)
    np.testing.assert_allclose(scores, [-3.0, 1.0, 1.0])


def test_conformal_quantile_matches_known_order_statistic():
    # n=9, target_coverage=0.8 -> ceil(10*0.8)=8th order statistic (1-indexed)
    scores = np.array([9.0, 1.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0])
    result = conformal_quantile(scores, target_coverage=0.8)
    assert result == 8.0

    # clipped at the max when the requested rank exceeds n
    small = np.array([1.0, 2.0, 3.0])
    assert conformal_quantile(small, target_coverage=0.99) == 3.0


def test_apply_conformal_widening_shifts_bounds_and_coverage_matches_suffix_reader():
    preds = pd.DataFrame({
        "custom_points": [1.0, 5.0, 9.0],
        "pred_q10": [2.0, 4.0, 6.0],
        "pred_q90": [3.0, 6.0, 8.0],
    })
    widened = apply_conformal_widening(preds, lower_q=0.1, upper_q=0.9, widen_by=1.5)

    np.testing.assert_allclose(widened["pred_q10_cqr"], [0.5, 2.5, 4.5])
    np.testing.assert_allclose(widened["pred_q90_cqr"], [4.5, 7.5, 9.5])
    # originals untouched
    np.testing.assert_allclose(widened["pred_q10"], preds["pred_q10"])

    before = quantile_interval_coverage(widened, target_col="custom_points", lower_q=0.1, upper_q=0.9)
    after = quantile_interval_coverage(
        widened, target_col="custom_points", lower_q=0.1, upper_q=0.9, suffix="_cqr"
    )
    # every row's actual now falls inside the widened interval
    assert after["within_interval"] == pytest.approx(1.0)
    assert after["within_interval"] >= before["within_interval"]


def test_conformal_widening_fixes_undercoverage_on_held_out_calibration_split():
    """
    End-to-end check: build quantile predictions that are deliberately
    too narrow (known undercoverage), split into a calibration half and
    a later evaluation half (respecting time order -- calibration is
    weeks 1-10, evaluation is weeks 11-20, never overlapping), calibrate
    on the EARLIER half only, and confirm coverage on the LATER,
    untouched-by-calibration half moves from clearly-undercovered toward
    the ~80% target.
    """
    rng = np.random.default_rng(1)
    n = 4000
    actual = rng.normal(loc=10, scale=5, size=n)
    # Deliberately too-narrow interval: true 10th/90th of N(10,5) are
    # roughly 3.6/16.4, but predict a much tighter band around the mean.
    pred_lower = np.full(n, 8.0)
    pred_upper = np.full(n, 12.0)
    week = np.repeat(np.arange(1, 21), n // 20)

    preds = pd.DataFrame({
        "custom_points": actual, "pred_q10": pred_lower, "pred_q90": pred_upper, "week": week,
    })
    calib = preds[preds["week"] <= 10]
    evaluation = preds[preds["week"] > 10]

    before = quantile_interval_coverage(evaluation, target_col="custom_points", lower_q=0.1, upper_q=0.9)
    assert before["within_interval"] < 0.6  # confirm it's badly undercovered to start

    scores = conformity_scores(calib, target_col="custom_points", lower_q=0.1, upper_q=0.9)
    widen_by = conformal_quantile(scores, target_coverage=0.8)
    widened_eval = apply_conformal_widening(evaluation, lower_q=0.1, upper_q=0.9, widen_by=widen_by)

    after = quantile_interval_coverage(
        widened_eval, target_col="custom_points", lower_q=0.1, upper_q=0.9, suffix="_cqr"
    )
    assert after["within_interval"] == pytest.approx(0.8, abs=0.03)


def test_interval_breach_by_prediction_bucket_shape():
    n = 300
    rng = np.random.default_rng(2)
    calib = pd.DataFrame({
        "custom_points": rng.normal(10, 3, size=n),
        "pred_q50": np.linspace(0, 20, n),
        "pred_q10": np.linspace(0, 20, n) - 5,
        "pred_q90": np.linspace(0, 20, n) + 5,
    })
    result = interval_breach_by_prediction_bucket(
        calib, target_col="custom_points", lower_q=0.1, upper_q=0.9, n_bins=3
    )
    assert len(result) == 3
    assert result["n"].sum() == n
    for _, row in result.iterrows():
        assert row["below"] + row["within"] + row["above"] == pytest.approx(1.0)


# ==========================================================================
# PHASE 8: train_final_models / predict_with_models
# ==========================================================================
def test_train_final_models_fits_one_point_model_and_a_quantile_pair_per_position(monkeypatch):
    """
    No-holdout training (Phase 8's artifact path): every row for a
    position should be used to fit its models -- unlike walk_forward_*,
    there's no train/test split here at all. Checks fit() is called with
    ALL of that position's rows, once for the point model and once per
    quantile alpha, and that positions absent from df are skipped.
    """
    df = _synthetic_position_df(n_players=4, seasons=(2024,), weeks=range(1, 6))
    import src.model as model_module

    fit_calls = []

    class _RecordingModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            fit_calls.append((self.kwargs.get("objective"), self.kwargs.get("alpha"), len(X)))
            return self

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(model_module.lgb, "LGBMRegressor", _RecordingModel)

    feature_cols = {"WR": ["feat_a", "feat_b"], "QB": ["feat_a", "feat_b"]}
    models = train_final_models(df, feature_cols=feature_cols, positions=("WR", "QB"))

    n_wr_rows = (df["position"] == "WR").sum()
    assert "WR" in models
    assert "QB" not in models  # no QB rows in the synthetic df -- silently skipped

    assert set(models["WR"].keys()) == {"point", "quantiles"}
    assert set(models["WR"]["quantiles"].keys()) == {0.10, 0.90}

    # one point fit + one fit per quantile alpha, every one on ALL WR rows
    assert ("regression", None, n_wr_rows) in fit_calls
    assert ("quantile", 0.10, n_wr_rows) in fit_calls
    assert ("quantile", 0.90, n_wr_rows) in fit_calls
    assert len(fit_calls) == 3


def test_predict_with_models_builds_floor_point_ceiling_with_cqr_widening_and_clipping():
    """
    Uses fake pre-trained models (skip real LightGBM entirely) with fixed
    predict() outputs to check the arithmetic predict_with_models is
    responsible for: floor/ceiling from min/max of the two quantile
    predictions, CQR widening applied per-position, and point clipped into
    [floor, ceiling] -- the same invariant src/export.py's
    validate_export() checks on the final exported JSON.
    """
    class _FixedModel:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    models = {
        "WR": {
            "point": _FixedModel(50.0),  # deliberately outside [floor, ceiling] before clipping
            "quantiles": {0.10: _FixedModel(12.0), 0.90: _FixedModel(8.0)},  # crossed on purpose
        },
    }
    cqr_widen_by = {"WR": 1.0}
    test_df = pd.DataFrame({
        "player_id": ["p1", "p2"],
        "position": ["WR", "WR"],
        "feat_a": [0.0, 0.0],
    })

    result = predict_with_models(test_df, models, cqr_widen_by, feature_cols={"WR": ["feat_a"]})

    assert list(result["player_id"]) == ["p1", "p2"]
    # floor/ceiling take min/max of the (crossed) quantile predictions, then widen by 1.0
    assert (result["floor"] == 8.0 - 1.0).all()
    assert (result["ceiling"] == 12.0 + 1.0).all()
    # point (50.0) is clipped down to the widened ceiling, not left out of range
    assert (result["point"] == result["ceiling"]).all()
    assert (result["floor"] <= result["point"]).all()
    assert (result["point"] <= result["ceiling"]).all()


def test_predict_with_models_skips_positions_not_present_in_test_df():
    class _FixedModel:
        def predict(self, X):
            return np.zeros(len(X))

    models = {
        "WR": {"point": _FixedModel(), "quantiles": {0.10: _FixedModel(), 0.90: _FixedModel()}},
        "QB": {"point": _FixedModel(), "quantiles": {0.10: _FixedModel(), 0.90: _FixedModel()}},
    }
    test_df = pd.DataFrame({"player_id": ["p1"], "position": ["WR"], "feat_a": [0.0]})

    result = predict_with_models(
        test_df, models, {"WR": 0.0, "QB": 0.0}, feature_cols={"WR": ["feat_a"], "QB": ["feat_a"]}
    )
    assert set(result["position"]) == {"WR"}


def test_predict_quantiles_with_models_widens_each_pair_by_its_own_constant_and_fixes_crossing():
    """
    src/simulate.py needs 5 CQR-calibrated points per player. Uses fake
    models with fixed, deliberately-crossed predict() outputs (q10 above
    q90, q25 above q75) to check: (1) the 10-90 and 25-75 pairs are each
    widened by their OWN constant, not a shared one, (2) within-pair
    crossing is resolved via min/max before widening, (3) q50 passes
    through untouched.
    """
    class _FixedModel:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    models = {
        "WR": {
            "point": _FixedModel(99.0),  # unused by this function
            "quantiles": {
                0.10: _FixedModel(12.0), 0.25: _FixedModel(14.0), 0.50: _FixedModel(10.0),
                0.75: _FixedModel(6.0), 0.90: _FixedModel(8.0),
            },
        },
    }
    cqr_10_90 = {"WR": 1.0}
    cqr_25_75 = {"WR": 2.0}
    test_df = pd.DataFrame({"player_id": ["p1"], "position": ["WR"], "feat_a": [0.0]})

    result = predict_quantiles_with_models(test_df, models, cqr_10_90, cqr_25_75, feature_cols={"WR": ["feat_a"]})

    row = result.iloc[0]
    # q10=12, q90=8 -> swapped to lo=8, hi=12, widened by 1.0
    assert row["pred_q10_cqr"] == pytest.approx(7.0)
    assert row["pred_q90_cqr"] == pytest.approx(13.0)
    # q25=14, q75=6 -> swapped to lo=6, hi=14, widened by 2.0
    assert row["pred_q25_cqr"] == pytest.approx(4.0)
    assert row["pred_q75_cqr"] == pytest.approx(16.0)
    # q50 untouched
    assert row["pred_q50"] == pytest.approx(10.0)


def test_predict_quantiles_with_models_raises_on_missing_alpha():
    class _FixedModel:
        def predict(self, X):
            return np.zeros(len(X))

    models = {"WR": {"point": _FixedModel(), "quantiles": {0.10: _FixedModel(), 0.90: _FixedModel()}}}  # missing 0.25/0.50/0.75
    test_df = pd.DataFrame({"player_id": ["p1"], "position": ["WR"], "feat_a": [0.0]})

    with pytest.raises(KeyError):
        predict_quantiles_with_models(test_df, models, {"WR": 0.0}, {"WR": 0.0}, feature_cols={"WR": ["feat_a"]})
