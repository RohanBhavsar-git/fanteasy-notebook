"""
Tests for src/usage.py's field heatmap zone-bucketing functions (Phase 5):
receiving_zone_plays, passing_zone_plays, rushing_zone_plays.

Aggregation into the exported eligible/groups/zones shape
(build_heatmap_snapshot) is tested in test_export.py, alongside
build_radar_snapshot -- this file is just the play-to-zone bucketing
itself, against small synthetic pbp frames.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.usage import (  # noqa: E402
    HEATMAP_DEPTH_LABELS,
    HEATMAP_FIELD_POS_LABELS,
    passing_zone_plays,
    receiving_zone_plays,
    rushing_zone_plays,
)


def _target_row(**overrides):
    """One synthetic real-target pbp row, defaults overridden per test."""
    row = {
        "season_type": "REG", "season": 2025, "week": 5,
        "pass_attempt": 1, "two_point_attempt": 0,
        "receiver_player_id": "00-rec1", "passer_player_id": "00-qb1",
        "air_yards": 8.0, "yardline_100": 60.0, "pass_location": "right",
        "sack": 0.0,
    }
    row.update(overrides)
    return row


def _carry_row(**overrides):
    row = {
        "season_type": "REG", "season": 2025, "week": 5,
        "rush_attempt": 1, "two_point_attempt": 0, "qb_kneel": 0,
        "rusher_player_id": "00-rb1", "yardline_100": 60.0, "run_location": "left",
    }
    row.update(overrides)
    return row


def test_receiving_zone_plays_buckets_depth_and_field_position_correctly():
    pbp = pd.DataFrame([
        _target_row(receiver_player_id="p1", air_yards=8.0, yardline_100=15.0),   # short|red_zone
        _target_row(receiver_player_id="p1", air_yards=25.0, yardline_100=70.0),  # deep|backfield
        _target_row(receiver_player_id="p1", air_yards=-2.0, yardline_100=30.0),  # behind_los|midfield
        _target_row(receiver_player_id="p2", air_yards=12.0, yardline_100=15.0),  # different player, excluded from p1's rows
        _target_row(receiver_player_id="p1", pass_attempt=0),                      # not a real pass attempt -- excluded
        _target_row(receiver_player_id=np.nan),                                    # no receiver -- excluded (e.g. a sack)
        _target_row(receiver_player_id="p1", two_point_attempt=1),                 # 2pt try -- excluded
        _target_row(receiver_player_id="p1", season_type="POST"),                  # playoffs -- excluded
    ])
    out = receiving_zone_plays(pbp)

    p1 = out[out["player_id"] == "p1"]
    assert len(p1) == 3
    assert set(zip(p1["zone_a"], p1["zone_b"])) == {
        ("short", "red_zone"), ("deep", "backfield"), ("behind_los", "midfield"),
    }
    assert len(out[out["player_id"] == "p2"]) == 1
    assert set(out["zone_a"]).issubset(set(HEATMAP_DEPTH_LABELS))
    assert set(out["zone_b"]).issubset(set(HEATMAP_FIELD_POS_LABELS))


def test_passing_zone_plays_groups_by_passer_and_excludes_sacks():
    """
    A sack sets pass_attempt==1 in this pbp snapshot but has no receiver
    and no pass_location -- _real_target_plays already drops it via the
    receiver_player_id.notna() requirement, so no separate sack==1 filter
    is needed in passing_zone_plays itself. Confirmed here rather than
    assumed: a synthetic sack row with pass_attempt==1 must NOT appear in
    the output.
    """
    pbp = pd.DataFrame([
        _target_row(passer_player_id="qb1", pass_location="left", air_yards=15.0, receiver_player_id="wr1"),
        _target_row(passer_player_id="qb1", pass_location="right", air_yards=3.0, receiver_player_id="wr2"),
        # A sack: pass_attempt==1, sack==1, no real receiver/location -- must be excluded.
        _target_row(passer_player_id="qb1", receiver_player_id=np.nan, pass_location=np.nan, sack=1.0),
    ])
    out = passing_zone_plays(pbp)

    assert len(out) == 2
    assert set(out["player_id"]) == {"qb1"}
    # zone_a=location, zone_b=depth (reversed order from receiving_zone_plays).
    assert set(zip(out["zone_a"], out["zone_b"])) == {("left", "intermediate"), ("right", "short")}


def test_passing_zone_plays_drops_rows_with_unknown_location():
    pbp = pd.DataFrame([
        _target_row(passer_player_id="qb1", pass_location=np.nan),
    ])
    out = passing_zone_plays(pbp)
    assert out.empty


def test_rushing_zone_plays_buckets_direction_and_field_position_and_excludes_kneels():
    pbp = pd.DataFrame([
        _carry_row(rusher_player_id="rb1", run_location="left", yardline_100=8.0),    # left|red_zone
        _carry_row(rusher_player_id="rb1", run_location="middle", yardline_100=55.0), # middle|backfield
        _carry_row(rusher_player_id="rb1", qb_kneel=1),                                # kneel -- excluded
        _carry_row(rusher_player_id="rb1", run_location=np.nan),                       # unknown direction -- excluded
        _carry_row(rusher_player_id="rb1", two_point_attempt=1),                       # 2pt try -- excluded
    ])
    out = rushing_zone_plays(pbp)

    assert len(out) == 2
    assert set(zip(out["zone_a"], out["zone_b"])) == {("left", "red_zone"), ("middle", "backfield")}
    assert set(out["zone_a"]).issubset({"left", "middle", "right"})
