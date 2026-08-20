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

from src.model import chronological_folds, walk_forward_predict  # noqa: E402


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
