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
    chronological_folds,
    fix_quantile_crossing,
    quantile_coverage,
    quantile_crossing_rate,
    quantile_interval_coverage,
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
