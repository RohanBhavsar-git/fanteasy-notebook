"""
Tests for src/usage.py's Family 7 (Phase 3' -- usage trend signal).

Point-in-time / leakage / idempotency coverage lives in test_no_leakage.py,
alongside the rest of Families 1-6, since it needs the same real-data
fixtures those tests already share. This file is functional/unit tests
against small synthetic frames: the rz_opportunity_share formula, the
direction-label thresholds, the MIN_GAMES_FOR_TREND gate, and
get_usage_trend_leaders' selection/sorting/filtering behavior.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.usage import (  # noqa: E402
    MIN_GAMES_FOR_TREND,
    TREND_DIRECTION_THRESHOLD,
    TREND_SOURCE_FEATURES,
    add_trend_features,
    get_usage_trend_leaders,
)


def _make_rolled_row(player_id, season, week, games_played, **overrides):
    """One synthetic player-week with every column add_trend_features
    requires, all defaulted to a flat/inert value the caller overrides."""
    row = {
        "player_id": player_id, "season": season, "week": week,
        "games_played": games_played,
        "rz_targets": 0.0, "team_rz_targets": 4.0,
        "rz_carries": 0.0, "team_rz_carries": 4.0,
    }
    for feat in ("target_share", "carry_share", "offense_pct"):
        row[feat] = 0.2
        row[f"{feat}_ewm3"] = 0.2
        row[f"{feat}_s2d"] = 0.2
        row[f"{feat}_vol"] = 0.05
    row.update(overrides)
    return row


def test_rz_opportunity_share_formula_and_zero_over_zero_is_null():
    df = pd.DataFrame([
        _make_rolled_row("p1", 2025, 5, games_played=5, rz_targets=2.0, team_rz_targets=6.0,
                          rz_carries=1.0, team_rz_carries=3.0),
        _make_rolled_row("p2", 2025, 5, games_played=5, rz_targets=0.0, team_rz_targets=0.0,
                          rz_carries=0.0, team_rz_carries=0.0),
    ])
    out = add_trend_features(df)

    p1 = out[out["player_id"] == "p1"].iloc[0]
    assert p1["rz_opportunity_share"] == pytest.approx((2.0 + 1.0) / (6.0 + 3.0))

    p2 = out[out["player_id"] == "p2"].iloc[0]
    assert pd.isna(p2["rz_opportunity_share"])  # team had zero RZ plays -- 0/0, not 0


def test_direction_labels_match_the_documented_threshold():
    """Clearly above/below the z-threshold flips to rising/falling; a
    smaller in-between gap and an exactly-flat gap both read as 'stable'."""
    vol = 0.04
    thresh_gap = TREND_DIRECTION_THRESHOLD * vol
    rows = [
        _make_rolled_row("just_inside", 2025, 10, games_played=8,
                          target_share_ewm3=0.20 + thresh_gap * 0.5, target_share_s2d=0.20, target_share_vol=vol),
        _make_rolled_row("above_thresh", 2025, 10, games_played=8,
                          target_share_ewm3=0.20 + thresh_gap * 2, target_share_s2d=0.20, target_share_vol=vol),
        _make_rolled_row("below_thresh", 2025, 10, games_played=8,
                          target_share_ewm3=0.20 - thresh_gap * 2, target_share_s2d=0.20, target_share_vol=vol),
        _make_rolled_row("flat", 2025, 10, games_played=8,
                          target_share_ewm3=0.20, target_share_s2d=0.20, target_share_vol=vol),
    ]
    out = add_trend_features(pd.DataFrame(rows)).set_index("player_id")

    assert out.loc["just_inside", "target_share_trend_direction"] == "stable"
    assert out.loc["above_thresh", "target_share_trend_direction"] == "rising"
    assert out.loc["below_thresh", "target_share_trend_direction"] == "falling"
    assert out.loc["flat", "target_share_trend_direction"] == "stable"


def test_min_games_threshold_boundary():
    rows = [
        _make_rolled_row("thin", 2025, 6, games_played=MIN_GAMES_FOR_TREND - 1,
                          target_share_ewm3=0.5, target_share_s2d=0.2, target_share_vol=0.05),
        _make_rolled_row("eligible", 2025, 6, games_played=MIN_GAMES_FOR_TREND,
                          target_share_ewm3=0.5, target_share_s2d=0.2, target_share_vol=0.05),
    ]
    out = add_trend_features(pd.DataFrame(rows)).set_index("player_id")

    assert pd.isna(out.loc["thin", "target_share_trend_signal"])
    assert pd.isna(out.loc["thin", "target_share_trend_direction"])
    assert pd.notna(out.loc["eligible", "target_share_trend_signal"])
    assert out.loc["eligible", "target_share_trend_direction"] == "rising"


def test_add_trend_features_rejects_unknown_feature_in_leaders():
    df = pd.DataFrame([_make_rolled_row("p1", 2025, 6, games_played=6)])
    out = add_trend_features(df)
    with pytest.raises(ValueError, match="unknown feature"):
        get_usage_trend_leaders(out, 2025, 6, feature="not_a_real_feature")


def test_get_usage_trend_leaders_sorts_filters_position_and_excludes_null_signal():
    rows = []
    # Three WRs with distinct signals, one RB (filtered out by position),
    # one WR with games_played below the eligibility floor (null signal --
    # must never appear in either list regardless of how extreme its raw
    # ewm3/s2d gap looks).
    rows.append(_make_rolled_row(
        "wr_up", 2025, 8, games_played=8, position="WR",
        target_share_ewm3=0.40, target_share_s2d=0.20, target_share_vol=0.05,
    ))
    rows.append(_make_rolled_row(
        "wr_down", 2025, 8, games_played=8, position="WR",
        target_share_ewm3=0.05, target_share_s2d=0.20, target_share_vol=0.05,
    ))
    rows.append(_make_rolled_row(
        "wr_flat", 2025, 8, games_played=8, position="WR",
        target_share_ewm3=0.20, target_share_s2d=0.20, target_share_vol=0.05,
    ))
    rows.append(_make_rolled_row(
        "rb_up", 2025, 8, games_played=8, position="RB",
        target_share_ewm3=0.90, target_share_s2d=0.10, target_share_vol=0.05,
    ))
    rows.append(_make_rolled_row(
        "wr_noisy_thin", 2025, 8, games_played=MIN_GAMES_FOR_TREND - 1, position="WR",
        target_share_ewm3=0.99, target_share_s2d=0.01, target_share_vol=0.01,
    ))
    df = pd.DataFrame(rows)
    for col in ("player_display_name", "team"):
        df[col] = "x"

    out = add_trend_features(df)

    risers, fallers = get_usage_trend_leaders(out, 2025, 8, feature="target_share", position="WR", top_n=5)

    assert list(risers["player_id"]) == ["wr_up", "wr_flat", "wr_down"]
    assert list(fallers["player_id"]) == ["wr_down", "wr_flat", "wr_up"]
    assert "rb_up" not in set(risers["player_id"]) | set(fallers["player_id"])
    assert "wr_noisy_thin" not in set(risers["player_id"]) | set(fallers["player_id"])


def test_get_usage_trend_leaders_respects_top_n():
    rows = [
        _make_rolled_row(
            f"wr{i}", 2025, 4, games_played=6, position="WR",
            target_share_ewm3=0.20 + i * 0.05, target_share_s2d=0.20, target_share_vol=0.05,
        )
        for i in range(8)
    ]
    df = pd.DataFrame(rows)
    for col in ("player_display_name", "team"):
        df[col] = "x"
    out = add_trend_features(df)

    risers, fallers = get_usage_trend_leaders(out, 2025, 4, feature="target_share", top_n=3)
    assert len(risers) == 3
    assert len(fallers) == 3
