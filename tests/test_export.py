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

import src.export as export_module  # noqa: E402
from src.export import (  # noqa: E402
    assemble_player_advanced_stats,
    assemble_simulation_block,
    build_heatmap_snapshot,
    build_matchup_simulation,
    build_playoff_odds,
    build_radar_snapshot,
    build_starter_quantile_rows,
    build_target_week_features,
    build_team_game_id_lookup,
    build_trend_snapshot,
    build_weekly_xfp,
    build_xfp_summary,
    get_archive_candidates,
    get_export_candidates,
    get_export_scope,
    normalize_team_code,
    position_starter_counts,
    validate_export,
    validate_simulation,
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


def test_get_archive_candidates_uses_last_real_team_not_current_sleeper_snapshot():
    """
    The one real difference from get_export_candidates: a player with a
    real crosswalk match but NO current team on file (retired/moved on by
    today) is still a candidate here, resolved from their own last real
    row -- this is the whole point of get_archive_candidates existing
    separately, not a live-path detail carried over by accident.
    """
    historical_features = pd.DataFrame([
        {"player_id": "00-0001", "position": "QB", "season": 2024, "week": 1, "team": "KC"},
        {"player_id": "00-0001", "position": "QB", "season": 2025, "week": 3, "team": "KC"},  # latest -- should win
        {"player_id": "00-0002", "position": "RB", "season": 2021, "week": 5, "team": "LAR"},  # retired since; needs LAR->LA
        {"player_id": "00-0003", "position": "K", "season": 2025, "week": 1, "team": "SF"},    # wrong position
        {"player_id": "00-0004", "position": "WR", "season": 2025, "week": 2, "team": "DAL"},  # no crosswalk entry
    ])
    crosswalk = pd.DataFrame([
        {"sleeper_id": "1", "gsis_id": "00-0001"},
        {"sleeper_id": "2", "gsis_id": "00-0002"},
    ])

    candidates, report = get_archive_candidates(historical_features, crosswalk)

    assert set(candidates["player_id"]) == {"00-0001", "00-0002"}
    assert candidates.loc[candidates["player_id"] == "00-0002", "team"].iloc[0] == "LA"
    assert report == {
        "n_position_eligible": 3,   # QB/RB/WR, K excluded -- 0001/0002/0004
        "n_crosswalk_matched": 2,   # 0004 has no crosswalk entry
        "n_with_current_team": 2,   # always == n_crosswalk_matched here, by construction
        "crosswalk_match_rate": pytest.approx(2 / 3),
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


def test_build_weekly_xfp_scopes_to_target_season_and_drops_null_weeks():
    historical = pd.DataFrame([
        {"player_id": "p1", "season": 2025, "week": 1, "xfp": 9.5},
        {"player_id": "p1", "season": 2025, "week": 2, "xfp": np.nan},  # dropped, not zero-filled
        {"player_id": "p1", "season": 2024, "week": 1, "xfp": 100.0},  # wrong season, excluded
        {"player_id": "p2", "season": 2025, "week": 1, "xfp": 12.25},
    ])
    result = build_weekly_xfp(historical, target_season=2025)

    assert set(zip(result["player_id"], result["week"])) == {("p1", 1), ("p2", 1)}
    assert result.set_index("player_id").loc["p1", "xfp"] == pytest.approx(9.5)


def test_build_weekly_xfp_empty_when_target_season_has_no_rows():
    historical = pd.DataFrame([{"player_id": "p1", "season": 2025, "week": 1, "xfp": 9.5}])
    result = build_weekly_xfp(historical, target_season=2026)
    assert result.empty


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


def test_build_trend_snapshot_renames_ewm3_columns_to_avoid_usage_collision():
    """
    target_share_ewm3/offense_pct_ewm3 are ALSO columns on the separate
    `usage` frame (USAGE_EXPORT_COLUMNS) -- if build_trend_snapshot
    returned them under that same name, merging both onto the same row in
    assemble_player_advanced_stats would silently suffix them to _x/_y
    instead of raising. This checks the rename that prevents it, and that
    only the target week's rows come back.
    """
    combined = pd.DataFrame({
        "player_id": ["p1", "p1", "p2"],
        "season": [2025, 2026, 2026],
        "week": [17, 1, 1],
        "target_share_ewm3": [0.30, np.nan, 0.25],
        "carry_share_ewm3": [np.nan, np.nan, 0.10],
        "offense_pct_ewm3": [0.80, np.nan, 0.60],
        "rz_opportunity_share_ewm3": [0.05, np.nan, np.nan],
        "target_share_trend_signal": [0.2, np.nan, -0.1],
        "carry_share_trend_signal": [np.nan, np.nan, np.nan],
        "offense_pct_trend_signal": [0.1, np.nan, 0.3],
        "rz_opportunity_share_trend_signal": [np.nan, np.nan, np.nan],
        "target_share_trend_direction": ["stable", None, "falling"],
        "carry_share_trend_direction": [None, None, None],
        "offense_pct_trend_direction": ["stable", None, "rising"],
        "rz_opportunity_share_trend_direction": [None, None, None],
    })

    out = build_trend_snapshot(combined, target_season=2026, target_week=1)

    assert set(out["player_id"]) == {"p1", "p2"}  # only the target week
    assert "target_share_ewm3" not in out.columns  # renamed, not passed through verbatim
    assert "offense_pct_ewm3" not in out.columns
    assert {"trend_target_share_current", "trend_carry_share_current",
            "trend_offense_pct_current", "trend_rz_opportunity_share_current"}.issubset(out.columns)

    p2 = out[out["player_id"] == "p2"].iloc[0]
    assert p2["trend_target_share_current"] == pytest.approx(0.25)
    assert p2["target_share_trend_direction"] == "falling"


# ==========================================================================
# PHASE 4: radar percentiles
# ==========================================================================
def test_position_starter_counts_matches_this_leagues_real_roster_positions():
    """
    Pinned against this league's REAL roster_positions (14 teams: QB, 2x RB,
    2x WR, TE, FLEX, K, DEF, 5x BN) -- if either this function or
    index.html's positionStarterCount() drifts from the other, this breaks
    instead of the two silently disagreeing. WR's 35 matches the number
    already published in the Weekly Production chart's own footnote
    ("(2 WR + 0.5 flex share) x 14 teams ~= 35" -- see PROJECT_CONTEXT.md).
    """
    roster_positions = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN", "BN", "BN", "BN"]
    counts = position_starter_counts(roster_positions, n_teams=14)
    assert counts == {"QB": 14, "RB": 33, "WR": 35, "TE": 16}


def test_position_starter_counts_falls_back_to_defaults_when_position_absent():
    """A hypothetical league with no dedicated slot AND no FLEX for a
    position falls back to _DEFAULT_STARTER_SHARE, same as the JS
    fallback branch (raw === 0)."""
    roster_positions = ["QB", "WR", "WR", "TE", "K", "DEF", "BN"]  # no RB slot, no FLEX
    counts = position_starter_counts(roster_positions, n_teams=10)
    assert counts["RB"] == round(2.5 * 10)  # default RB share (2.5) x 10 teams
    assert counts["QB"] == 10  # real direct slot, unaffected by the fallback


def _radar_test_frame(rows: list[dict]) -> pd.DataFrame:
    """Fills in every RADAR_METRICS axis column not explicitly set on a
    row with NaN -- derived from the module's own RADAR_METRICS constant
    rather than a hardcoded column list, so this helper can't quietly
    drift from what build_radar_snapshot actually reads."""
    all_cols = sorted({axis["column"] for axes in export_module.RADAR_METRICS.values() for axis in axes})
    df = pd.DataFrame(rows)
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def test_build_radar_snapshot_percentiles_against_pool_only_not_every_candidate():
    """
    3 WR candidates eligible on games_played, 1 short of the floor.
    roster_positions gives a WR pool of exactly 2 -- w3 (3rd by season
    points) is excluded from the pool but still gets percentiled AGAINST
    it, which is the whole point: a bench-caliber target_share should read
    as near-0th percentile relative to real starters, not compressed
    upward by including every WR in the denominator.
    """
    combined = _radar_test_frame([
        # target-week (2025, wk10) stub rows
        {"player_id": "w1", "position": "WR", "season": 2025, "week": 10, "games_played": 8,
         "target_share_s2d": 0.30, "custom_points": np.nan},
        {"player_id": "w2", "position": "WR", "season": 2025, "week": 10, "games_played": 8,
         "target_share_s2d": 0.20, "custom_points": np.nan},
        {"player_id": "w3", "position": "WR", "season": 2025, "week": 10, "games_played": 8,
         "target_share_s2d": 0.05, "custom_points": np.nan},
        {"player_id": "w4", "position": "WR", "season": 2025, "week": 10, "games_played": 2,
         "target_share_s2d": 0.50, "custom_points": np.nan},  # below games floor -- excluded outright
        # historical rows behind the target week -- season-to-date points ranking
        {"player_id": "w1", "position": "WR", "season": 2025, "week": 9, "games_played": 7, "custom_points": 100.0},
        {"player_id": "w2", "position": "WR", "season": 2025, "week": 9, "games_played": 7, "custom_points": 80.0},
        {"player_id": "w3", "position": "WR", "season": 2025, "week": 9, "games_played": 7, "custom_points": 20.0},
    ])
    roster_positions = ["WR"]  # 1 direct slot, no FLEX -- pool = round(1 * 2) = 2 teams

    radar = build_radar_snapshot(combined, target_season=2025, target_week=10,
                                  roster_positions=roster_positions, n_teams=2)

    assert radar["w1"]["eligible"] and radar["w2"]["eligible"] and radar["w3"]["eligible"]
    assert radar["w1"]["pool_size"] == 2  # w1, w2 -- w3 ranked 3rd by season points, excluded
    target_share_axis = lambda r: next(a for a in r["axes"] if a["label"] == "Target Share")
    assert target_share_axis(radar["w1"])["percentile"] == 75   # above w2, below no one else in pool
    assert target_share_axis(radar["w1"])["raw"] == pytest.approx(30.0)  # 0.30 -> 30.0, pct=True axis
    assert target_share_axis(radar["w2"])["percentile"] == 25
    assert target_share_axis(radar["w3"])["percentile"] == 0    # not in the pool, ranks below both who are

    # w4 never got a chance to compress w3's ranking -- excluded before pool selection.
    assert radar["w4"] == {"eligible": False, "games_played": 2, "min_games": 5}


def test_build_radar_snapshot_all_ineligible_when_nobody_clears_games_floor():
    """Early season: everyone at a position is short on games -- pool_size
    is 0, so every candidate at that position (even one with real target
    share data) gets an honest ineligible object, never a percentile
    against an empty/meaningless pool."""
    combined = _radar_test_frame([
        {"player_id": "w1", "position": "WR", "season": 2025, "week": 2, "games_played": 1,
         "target_share_s2d": 0.30, "custom_points": np.nan},
        {"player_id": "w1", "position": "WR", "season": 2025, "week": 1, "games_played": 0, "custom_points": 5.0},
    ])
    radar = build_radar_snapshot(combined, target_season=2025, target_week=2,
                                  roster_positions=["WR"], n_teams=2)
    assert radar["w1"] == {"eligible": False, "games_played": 1, "min_games": 5}


# ==========================================================================
# PHASE 5: field heatmap zones
# ==========================================================================
def _pbp_row(**overrides):
    """
    build_heatmap_snapshot runs receiving_zone_plays/passing_zone_plays/
    rushing_zone_plays over the SAME full pbp frame regardless of which
    kinds a given test cares about (real nflverse pbp always has every
    column, just null where a play type doesn't apply) -- so every
    synthetic row here carries the full neutral schema all three
    functions read, with only the fields relevant to one play type
    actually set to something real. _pbp_target_row/_pbp_carry_row below
    just override the ones that matter for their play type.
    """
    row = {
        "season_type": "REG", "season": 2025, "week": 5, "two_point_attempt": 0,
        "pass_attempt": 0, "receiver_player_id": np.nan, "passer_player_id": np.nan,
        "air_yards": np.nan, "pass_location": np.nan, "sack": 0.0,
        "rush_attempt": 0, "qb_kneel": 0, "rusher_player_id": np.nan, "run_location": np.nan,
        "yardline_100": 60.0,
    }
    row.update(overrides)
    return row


def _pbp_target_row(**overrides):
    return _pbp_row(**{
        "pass_attempt": 1, "receiver_player_id": "w1", "passer_player_id": "qb1",
        "air_yards": 8.0, "pass_location": "right",
        **overrides,
    })


def _pbp_carry_row(**overrides):
    return _pbp_row(**{
        "rush_attempt": 1, "rusher_player_id": "rb1", "run_location": "left",
        **overrides,
    })


def test_build_heatmap_snapshot_handles_completely_empty_pbp_without_crashing():
    """
    Real scenario: a not-yet-started season where get_pbp() itself can't
    even be called (nflreadpy's own season-range validation), so
    scripts/weekly_update.py passes pd.DataFrame() -- zero rows AND zero
    columns, not the real nflverse schema. games_played is necessarily 0
    for every candidate in this scenario (5+ games played is impossible
    without pbp for those games existing), so this must come back
    ineligible for everyone rather than a KeyError on a missing column.
    """
    combined = pd.DataFrame([
        {"player_id": "w1", "position": "WR", "season": 2026, "week": 1, "games_played": 0},
    ])
    heatmap = build_heatmap_snapshot(combined, pd.DataFrame(), target_season=2026, target_week=1)
    assert heatmap["w1"] == {"eligible": False, "games_played": 0, "min_games": 5}


def test_build_heatmap_snapshot_ineligible_below_games_floor():
    combined = pd.DataFrame([
        {"player_id": "w1", "position": "WR", "season": 2025, "week": 10, "games_played": 3},
    ])
    pbp = pd.DataFrame([_pbp_target_row(receiver_player_id="w1", week=9)])
    heatmap = build_heatmap_snapshot(combined, pbp, target_season=2025, target_week=10)
    assert heatmap["w1"] == {"eligible": False, "games_played": 3, "min_games": 5}


def test_build_heatmap_snapshot_receiving_shares_sum_to_one_and_flags_sparse():
    combined = pd.DataFrame([
        {"player_id": "w1", "position": "WR", "season": 2025, "week": 10, "games_played": 9},
    ])
    pbp = pd.DataFrame(
        [_pbp_target_row(receiver_player_id="w1", air_yards=8.0, yardline_100=15.0, week=w) for w in range(1, 6)]
        + [_pbp_target_row(receiver_player_id="w1", air_yards=25.0, yardline_100=70.0, week=1)]  # 1 play -- sparse
        + [_pbp_target_row(receiver_player_id="w1", week=10)]  # target week itself -- not yet played, excluded
    )
    heatmap = build_heatmap_snapshot(combined, pbp, target_season=2025, target_week=10)

    h = heatmap["w1"]
    assert h["eligible"] is True
    assert len(h["groups"]) == 1
    group = h["groups"][0]
    assert group["kind"] == "receiving"
    assert group["total_plays"] == 6  # 5 short|red_zone + 1 deep|backfield -- NOT the week-10 row
    assert abs(sum(z["share"] for z in group["zones"]) - 1.0) < 1e-9

    solid = next(z for z in group["zones"] if z["id"] == "short|red_zone")
    assert solid["count"] == 5 and solid["sparse"] is False
    thin = next(z for z in group["zones"] if z["id"] == "deep|backfield")
    assert thin["count"] == 1 and thin["sparse"] is True
    assert group["zones"][0]["id"] == "short|red_zone"  # sorted by count descending


def test_build_heatmap_snapshot_rb_gets_independent_rushing_and_receiving_groups():
    combined = pd.DataFrame([
        {"player_id": "rb1", "position": "RB", "season": 2025, "week": 10, "games_played": 9},
    ])
    pbp = pd.DataFrame(
        [_pbp_carry_row(rusher_player_id="rb1", run_location="left", yardline_100=15.0) for _ in range(3)]
        + [_pbp_carry_row(rusher_player_id="rb1", run_location="right", yardline_100=70.0)]
        + [_pbp_target_row(receiver_player_id="rb1", air_yards=-2.0, yardline_100=40.0)]
    )
    heatmap = build_heatmap_snapshot(combined, pbp, target_season=2025, target_week=10)

    h = heatmap["rb1"]
    kinds = {g["kind"]: g for g in h["groups"]}
    assert set(kinds) == {"rushing", "receiving"}
    assert kinds["rushing"]["total_plays"] == 4
    assert kinds["receiving"]["total_plays"] == 1
    # Each kind's shares sum to 1.0 independently -- a target and a carry
    # don't share a denominator, so they're never pooled into one 1.0.
    assert abs(sum(z["share"] for z in kinds["rushing"]["zones"]) - 1.0) < 1e-9
    assert abs(sum(z["share"] for z in kinds["receiving"]["zones"]) - 1.0) < 1e-9


def test_build_heatmap_snapshot_qb_gets_passing_only_never_rushing():
    combined = pd.DataFrame([
        {"player_id": "qb1", "position": "QB", "season": 2025, "week": 10, "games_played": 9},
    ])
    pbp = pd.DataFrame([
        _pbp_target_row(passer_player_id="qb1", receiver_player_id="w9", pass_location="middle", air_yards=15.0),
        _pbp_carry_row(rusher_player_id="qb1"),  # a real QB scramble -- must not appear (QB is passing-only)
    ])
    heatmap = build_heatmap_snapshot(combined, pbp, target_season=2025, target_week=10)
    assert [g["kind"] for g in heatmap["qb1"]["groups"]] == ["passing"]


def test_build_heatmap_snapshot_omits_a_kind_with_zero_real_plays():
    """A rostered RB who genuinely never touched the ball as a receiver
    (real, if rare) gets a rushing-only groups list, not a receiving
    group padded with zero zones."""
    combined = pd.DataFrame([
        {"player_id": "rb1", "position": "RB", "season": 2025, "week": 10, "games_played": 9},
    ])
    pbp = pd.DataFrame([_pbp_carry_row(rusher_player_id="rb1")])
    heatmap = build_heatmap_snapshot(combined, pbp, target_season=2025, target_week=10)
    assert [g["kind"] for g in heatmap["rb1"]["groups"]] == ["rushing"]


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
        "xfp_vol": [np.nan, 3.2],
    })
    trend = pd.DataFrame({
        "player_id": ["00-0001", "00-0002"],
        "trend_target_share_current": [np.nan, 0.2],
        "trend_carry_share_current": [np.nan, np.nan],
        "trend_offense_pct_current": [0.95, 0.7],
        "trend_rz_opportunity_share_current": [np.nan, 0.15],
        "target_share_trend_signal": [np.nan, 0.4],
        "carry_share_trend_signal": [np.nan, np.nan],
        "offense_pct_trend_signal": [np.nan, -0.6],
        "rz_opportunity_share_trend_signal": [np.nan, 0.1],
        "target_share_trend_direction": [None, "rising"],
        "carry_share_trend_direction": [None, None],
        "offense_pct_trend_direction": [None, "falling"],
        "rz_opportunity_share_trend_direction": [None, "stable"],
    })
    xfp_summary = pd.DataFrame({
        "player_id": ["00-0002"],
        "season_xfp": [180.0], "season_actual": [200.0], "fp_over_expected": [20.0],
    })
    weekly_xfp = pd.DataFrame({
        "player_id": ["00-0002", "00-0002", "00-0002"],
        "week": [1, 2, 3],
        "xfp": [9.5, 11.25, np.nan],  # week 3 null on purpose -- should be dropped, not zero-filled
    })
    crosswalk = pd.DataFrame({
        "gsis_id": ["00-0001", "00-0002"],
        "sleeper_id": ["4984", "5001"],
    })
    # 00-0001 has no entry at all (falls back to the honest default);
    # 00-0002 has a real eligible radar from build_radar_snapshot's shape.
    radar = {
        "00-0002": {
            "eligible": True, "games_played": 8, "min_games": 5, "pool_size": 33,
            "axes": [{"label": "Touch Volume", "unit": " touches/gm", "percentile": 62, "raw": 14.3}],
        },
    }
    heatmap = {
        "00-0002": {
            "eligible": True, "games_played": 8, "min_games": 5,
            "groups": [{"kind": "rushing", "total_plays": 40,
                        "zones": [{"id": "left|midfield", "label": "Left · Midfield", "count": 40, "share": 1.0, "sparse": False}]}],
        },
    }

    payload, report = assemble_player_advanced_stats(
        scoped_predictions, usage, trend, xfp_summary, weekly_xfp, radar, heatmap, crosswalk,
        target_season=2026, target_week=1, seasons_trained=list(range(2018, 2026)),
        model_version="test-version",
    )

    assert report == {"n_scoped": 2, "n_matched": 2, "match_rate": 1.0}
    assert set(payload["players"].keys()) == {"4984", "5001"}
    assert payload["players"]["4984"]["xfp"]["season_xfp"] is None  # no xfp row for this player
    assert payload["players"]["5001"]["xfp"]["season_xfp"] == pytest.approx(180.0)
    assert payload["meta"]["season"] == 2026
    assert payload["meta"]["week"] == 1

    # weekly_xfp: grouped into {week_str: xfp}, null week dropped (not
    # zero-filled), and a player with no weekly_xfp rows at all gets {}
    # rather than a missing key.
    assert payload["players"]["4984"]["weekly_xfp"] == {}
    assert payload["players"]["5001"]["weekly_xfp"] == {"1": 9.5, "2": 11.25}

    # xfp_vol: rides the usage block like every other USAGE_EXPORT_COLUMNS entry.
    assert payload["players"]["4984"]["usage"]["xfp_vol"] is None
    assert payload["players"]["5001"]["usage"]["xfp_vol"] == pytest.approx(3.2)

    # Phase 3' trend block: renamed to human-readable keys, null-safe,
    # direction strings pass through untouched.
    assert payload["players"]["4984"]["trend"]["snap_share"] == {
        "current": 0.95, "signal": None, "direction": None,
    }
    assert payload["players"]["5001"]["trend"] == {
        "snap_share": {"current": 0.7, "signal": -0.6, "direction": "falling"},
        "target_share": {"current": 0.2, "signal": 0.4, "direction": "rising"},
        "carry_share": {"current": None, "signal": None, "direction": None},
        "red_zone_share": {"current": 0.15, "signal": 0.1, "direction": "stable"},
    }

    # radar: a candidate absent from the radar dict falls back to an honest
    # ineligible object rather than a missing key or a KeyError; a real
    # eligible entry passes through untouched.
    assert payload["players"]["4984"]["radar"] == {"eligible": False, "games_played": 0, "min_games": 5}
    assert payload["players"]["5001"]["radar"]["eligible"] is True
    assert payload["players"]["5001"]["radar"]["pool_size"] == 33
    assert payload["players"]["5001"]["radar"]["axes"][0]["label"] == "Touch Volume"

    # heatmap: same absent-entry fallback and pass-through shape as radar.
    assert payload["players"]["4984"]["heatmap"] == {"eligible": False, "games_played": 0, "min_games": 5}
    assert payload["players"]["5001"]["heatmap"]["eligible"] is True
    assert payload["players"]["5001"]["heatmap"]["groups"][0]["kind"] == "rushing"

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


# ==========================================================================
# STEP 7: simulation wiring (Phase 8 round 2)
# ==========================================================================
def test_build_team_game_id_lookup_stacks_home_and_away():
    schedule = pd.DataFrame([
        {"season": 2025, "week": 10, "game_type": "REG", "home_team": "KC", "away_team": "BUF", "game_id": "g1"},
        {"season": 2025, "week": 10, "game_type": "POST", "home_team": "SF", "away_team": "DAL", "game_id": "g2"},
    ])
    lookup = build_team_game_id_lookup(schedule)
    pairs = set(zip(lookup["team"], lookup["game_id"]))
    assert ("KC", "g1") in pairs
    assert ("BUF", "g1") in pairs
    assert not any(t in ("SF", "DAL") for t, _ in pairs)  # POST excluded


def test_build_starter_quantile_rows_covers_model_player_kdst_fallback_and_bye_drop():
    quantiles_by_gsis = pd.DataFrame([
        {"player_id": "00-001", "position": "RB", "pred_q10_cqr": 5.0, "pred_q25_cqr": 8.0,
         "pred_q50": 12.0, "pred_q75_cqr": 16.0, "pred_q90_cqr": 20.0},
    ])
    sleeper_proj_points = {"kdst_1": 7.5}  # no model coverage for this one at all
    sleeper_to_gsis = {"sleeper_1": "00-001"}  # model-covered starter
    # sleeper_2 has no crosswalk entry and no sleeper projection -> falls to 0.0
    team_by_sleeper = {"sleeper_1": "KC", "kdst_1": "SF", "sleeper_2": "BUF", "bye_guy": "MIA"}
    game_id_by_team_week = {(2025, 10, "KC"): "g1", (2025, 10, "SF"): "g2", (2025, 10, "BUF"): "g1"}
    # "MIA" deliberately has no game_id entry -- simulates a bye week

    rows = build_starter_quantile_rows(
        ["sleeper_1", "kdst_1", "sleeper_2", "bye_guy", "0", None],
        2025, 10, quantiles_by_gsis, sleeper_proj_points, sleeper_to_gsis, team_by_sleeper, game_id_by_team_week,
    )

    assert len(rows) == 3  # bye_guy dropped (no game_id), "0"/None skipped as empty slots
    model_row = rows[rows["game_id"] == "g1"].iloc[0]
    assert model_row["pred_q50"] == 12.0 and model_row["pred_q10_cqr"] == 5.0

    kdst_row = rows[rows["game_id"] == "g2"].iloc[0]
    assert (kdst_row[["pred_q10_cqr", "pred_q25_cqr", "pred_q50", "pred_q75_cqr", "pred_q90_cqr"]] == 7.5).all()

    zero_fallback_row = rows[(rows["game_id"] == "g1") & (rows["pred_q50"] == 0.0)]
    assert len(zero_fallback_row) == 1  # sleeper_2: no crosswalk, no sleeper projection -> 0.0 constant


def test_build_matchup_simulation_rounds_to_whole_percent_and_skips_empty_lineups(monkeypatch):
    matchups = pd.DataFrame([
        {"roster_id": 1, "matchup_id": 100},
        {"roster_id": 2, "matchup_id": 100},
        {"roster_id": 3, "matchup_id": 101},  # no partner -- odd group, skipped
        {"roster_id": 4, "matchup_id": 102},
        {"roster_id": 5, "matchup_id": 102},
    ])

    def fake_simulate_matchup(lineup_a, lineup_b, n_sims):
        return {"team_a_win_prob": 0.347, "team_b_win_prob": 0.653}

    monkeypatch.setattr(export_module, "simulate_matchup", fake_simulate_matchup)

    def lineup_fn(roster_id):
        if roster_id == 4:
            return pd.DataFrame(columns=["game_id"])  # roster 4's lineup unresolvable
        return pd.DataFrame({"game_id": ["g1"], "pred_q50": [10.0]})

    results = build_matchup_simulation(matchups, lineup_fn, n_sims=10000)

    assert len(results) == 1  # only matchup 100 has both a real pair AND resolvable lineups
    r = results[0]
    assert r["win_prob_a"] == 35 and r["win_prob_b"] == 65  # whole-percent rounding, not 34.7/65.3
    assert isinstance(r["win_prob_a"], int)


def test_build_playoff_odds_rounds_and_reshapes_to_string_keyed_dict(monkeypatch):
    def fake_simulate_season(remaining_weeks, starting_standings, lineup_builder, playoff_teams, n_sims):
        return pd.DataFrame({"roster_id": [1, 2], "playoff_prob": [0.667, 0.128], "mean_final_wins": [9.0, 5.0], "mean_final_points_for": [1400.0, 1200.0]})

    monkeypatch.setattr(export_module, "simulate_season", fake_simulate_season)

    standings = pd.DataFrame({"roster_id": [1, 2], "wins": [6, 3], "points_for": [800.0, 700.0]})
    odds = build_playoff_odds([], standings, lambda r, w: pd.DataFrame(), playoff_teams=6, n_sims=3000)

    assert odds == {"1": 67, "2": 13}  # rounded to whole percent, string keys
    for pct in odds.values():
        assert isinstance(pct, int)


def test_build_playoff_odds_returns_empty_dict_for_empty_standings():
    assert build_playoff_odds([], pd.DataFrame(), lambda r, w: pd.DataFrame(), playoff_teams=6) == {}


def test_assemble_simulation_block_returns_none_when_nothing_to_show():
    assert assemble_simulation_block([], {}, week=1) is None


def test_assemble_simulation_block_populates_when_either_side_has_data():
    block = assemble_simulation_block([{"matchup_id": 1, "win_prob_a": 60, "win_prob_b": 40}], {}, week=5)
    assert block["week"] == 5
    assert block["matchups"][0]["matchup_id"] == 1
    assert "calibration_caveat" in block and "accuracy_caveat" in block
    assert "93.6%" in block["accuracy_caveat"]


def test_validate_simulation_none_is_fine():
    assert validate_simulation(None) == {"present": False}


def test_validate_simulation_passes_on_sane_data():
    simulation = {
        "matchups": [{"matchup_id": 1, "win_prob_a": 60, "win_prob_b": 40}],
        "playoff_odds": {"1": 75, "2": 10},
    }
    report = validate_simulation(simulation)
    assert report == {"present": True, "n_matchups": 1, "n_rosters_with_playoff_odds": 2}


def test_validate_simulation_catches_probabilities_not_summing_to_100():
    simulation = {"matchups": [{"matchup_id": 1, "win_prob_a": 60, "win_prob_b": 60}], "playoff_odds": {}}
    with pytest.raises(AssertionError, match="not ~100"):
        validate_simulation(simulation)


def test_validate_simulation_catches_out_of_range_playoff_prob():
    simulation = {"matchups": [], "playoff_odds": {"1": 150}}
    with pytest.raises(AssertionError, match="out of \\[0, 100\\]"):
        validate_simulation(simulation)
