"""
Tests for src/export.py -- Phase 7 JSON export.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.export import (  # noqa: E402
    assemble_player_advanced_stats,
    build_target_week_features,
    build_xfp_summary,
    get_export_candidates,
    get_export_scope,
    normalize_team_code,
    validate_export,
)


def test_normalize_team_code_maps_lar_to_la_and_passes_others_through():
    assert normalize_team_code("LAR") == "LA"
    assert normalize_team_code("KC") == "KC"


def test_get_export_candidates_starts_from_history_and_reports_real_match_rate():
    historical_features = pd.DataFrame([
        {"player_id": "00-0001", "position": "QB"},    # matched, has a current team
        {"player_id": "00-0002", "position": "RB"},    # matched, team needs LAR->LA normalizing
        {"player_id": "00-0003", "position": "K"},      # wrong position -- never a candidate
        {"player_id": "00-0004", "position": "WR"},    # sleeper match but no current team (unsigned FA)
        {"player_id": "00-0005", "position": "TE"},    # no crosswalk entry at all -- a real gap
    ])
    sleeper_players = pd.DataFrame([
        {"sleeper_id": "1", "team": "KC"},
        {"sleeper_id": "2", "team": "LAR"},
        {"sleeper_id": "4", "team": None},
    ])
    crosswalk = pd.DataFrame([
        {"sleeper_id": "1", "gsis_id": "00-0001"},
        {"sleeper_id": "2", "gsis_id": "00-0002"},
        {"sleeper_id": "4", "gsis_id": "00-0004"},
    ])

    candidates, report = get_export_candidates(historical_features, sleeper_players, crosswalk)

    assert set(candidates["player_id"]) == {"00-0001", "00-0002"}
    assert candidates.loc[candidates["player_id"] == "00-0002", "team"].iloc[0] == "LA"
    assert report == {
        "n_position_eligible": 4,       # QB/RB/WR/TE only, K excluded
        "n_crosswalk_matched": 3,       # 0001, 0002, 0004 -- 0005 has no crosswalk entry
        "n_with_current_team": 2,       # 0004 matched but has no team on file
        "crosswalk_match_rate": 0.75,
    }


def test_get_export_scope_never_drops_rostered_players_below_top_n_cutoff():
    predictions = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(5)],
        "point": [30.0, 20.0, 10.0, 5.0, 1.0],
    })
    rostered_gsis_ids = {"p3"}  # a low-projection rostered player

    scoped, report = get_export_scope(rostered_gsis_ids, predictions, top_n=2)

    assert "p3" in scoped["player_id"].values  # rostered, kept despite low rank
    assert set(scoped["player_id"]) == {"p3", "p0", "p1"}  # rostered + top 2 of the rest
    assert report == {"n_rostered": 1, "n_top_n": 2, "n_total": 3}


def test_build_xfp_summary_sums_the_completed_season_and_computes_the_gap():
    historical = pd.DataFrame([
        {"player_id": "p1", "season": 2025, "xfp": 10.0, "custom_points": 12.0},
        {"player_id": "p1", "season": 2025, "xfp": 8.0, "custom_points": 6.0},
        {"player_id": "p1", "season": 2024, "xfp": 100.0, "custom_points": 100.0},  # wrong season, excluded
        {"player_id": "p2", "season": 2025, "xfp": np.nan, "custom_points": 5.0},  # QB row: xfp null
    ])
    result = build_xfp_summary(historical, xfp_season=2025).set_index("player_id")

    assert result.loc["p1", "season_xfp"] == pytest.approx(18.0)
    assert result.loc["p1", "season_actual"] == pytest.approx(18.0)
    assert result.loc["p1", "fp_over_expected"] == pytest.approx(0.0)
    assert pd.isna(result.loc["p2", "season_xfp"])  # never fabricated from a null source


def test_build_target_week_features_new_season_row_is_null_in_season_and_carries_prev_season():
    """
    The core novel mechanism: a stub row for a brand-new season/week must
    come out with null in-season rolling stats (nothing has happened yet
    this season) but a real prev_season_* value carried from the player's
    actual prior-season data -- exactly how every OTHER season's week 1
    already looks in real historical data, achieved with zero special-casing
    by reusing add_context_features/add_rolling_features unmodified.
    """
    from src.usage import ROLLING_SOURCE_COLUMNS

    rows = []
    for week in range(1, 4):
        row = {
            "player_id": "p1", "position": "WR", "team": "KC",
            "season": 2025, "week": week, "offense_pct": 0.8,
        }
        for col in ROLLING_SOURCE_COLUMNS:
            row[col] = 5.0
        rows.append(row)
    historical = pd.DataFrame(rows)

    candidates = pd.DataFrame([{"player_id": "p1", "position": "WR", "team": "KC"}])

    schedule = pd.DataFrame([{
        "game_type": "REG", "season": 2026, "week": 1,
        "home_team": "KC", "away_team": "DEN", "home_rest": 7, "away_rest": 7,
        "total_line": 45.0, "spread_line": 3.0, "roof": "outdoors", "surface": "grass",
        "temp": np.nan, "wind": np.nan,
    }])

    combined = build_target_week_features(historical, candidates, schedule, target_season=2026, target_week=1)
    stub_row = combined[(combined["season"] == 2026) & (combined["week"] == 1)].iloc[0]

    assert pd.isna(stub_row["target_share_ewm3"])  # nothing played yet this season
    assert stub_row["games_played"] == 0
    assert stub_row["prev_season_target_share"] == pytest.approx(5.0)  # carried from 2025
    assert stub_row["is_home"] == True  # noqa: E712 -- real Family 5 context from the real schedule
    assert stub_row["spread"] == pytest.approx(3.0)  # KC's own signed spread (favored, home spread_line=-3)


def test_assemble_and_validate_export_round_trip():
    scoped_predictions = pd.DataFrame({
        "player_id": ["00-0001", "00-0002"],
        "position": ["QB", "RB"],
        "point": [20.0, 12.0],
        "floor": [10.0, 5.0],
        "ceiling": [30.0, 20.0],
    })
    usage = pd.DataFrame({
        "player_id": ["00-0001", "00-0002"],
        "target_share_ewm3": [np.nan, 0.2],
        "touch_share_ewm3": [np.nan, 0.5],
        "offense_pct_ewm3": [0.95, 0.7],
        "snap_share_delta_3wk": [0.01, 0.02],
        "rz_target_share_ewm3": [np.nan, 0.1],
        "rz_carry_share_ewm3": [np.nan, 0.3],
        "prev_season_target_share": [np.nan, 0.18],
        "prev_season_touch_share": [np.nan, 0.45],
        "prev_season_offense_pct": [0.9, 0.65],
    })
    xfp_summary = pd.DataFrame({
        "player_id": ["00-0002"],
        "season_xfp": [180.0], "season_actual": [200.0], "fp_over_expected": [20.0],
    })
    crosswalk = pd.DataFrame({
        "gsis_id": ["00-0001", "00-0002"],
        "sleeper_id": ["4984", "5001"],
    })

    payload, report = assemble_player_advanced_stats(
        scoped_predictions, usage, xfp_summary, crosswalk,
        target_season=2026, target_week=1, seasons_trained=list(range(2018, 2026)),
        model_version="test-version",
    )

    assert report == {"n_scoped": 2, "n_matched": 2, "match_rate": 1.0}
    assert set(payload["players"].keys()) == {"4984", "5001"}
    assert payload["players"]["4984"]["xfp"]["season_xfp"] is None  # no xfp row for this player
    assert payload["players"]["5001"]["xfp"]["season_xfp"] == pytest.approx(180.0)
    assert payload["meta"]["season"] == 2026
    assert payload["meta"]["week"] == 1

    validation = validate_export(payload, crosswalk)
    assert validation["n_players"] == 2


def test_validate_export_catches_floor_point_ceiling_violation():
    payload = {"players": {"4984": {
        "projection": {"point": 50.0, "floor": 10.0, "ceiling": 30.0},  # point above ceiling
    }}}
    crosswalk = pd.DataFrame({"sleeper_id": ["4984"]})
    with pytest.raises(AssertionError, match="floor <= point <= ceiling"):
        validate_export(payload, crosswalk)


def test_validate_export_catches_unknown_sleeper_id():
    payload = {"players": {"not_a_real_id": {
        "projection": {"point": 10.0, "floor": 5.0, "ceiling": 15.0},
    }}}
    crosswalk = pd.DataFrame({"sleeper_id": ["4984"]})
    with pytest.raises(AssertionError, match="not real Sleeper IDs"):
        validate_export(payload, crosswalk)
