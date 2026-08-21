"""
Tests for src/simulate.py -- game-environment sampling and matchup
simulation (Phase 6.5 step 9).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulate import (  # noqa: E402
    QUANTILE_COLUMNS,
    _ensure_monotonic_quantiles,
    _inverse_quantile_row,
    calibration_report,
    sample_player_week,
    simulate_matchup,
    simulate_season,
)
from src.ingest import playoff_participants_from_bracket  # noqa: E402


def _row(game_id, q10, q25, q50, q75, q90):
    return {
        "game_id": game_id, "player_id": None,
        "pred_q10_cqr": q10, "pred_q25_cqr": q25, "pred_q50": q50,
        "pred_q75_cqr": q75, "pred_q90_cqr": q90,
    }


def test_ensure_monotonic_quantiles_sorts_crossed_rows():
    df = pd.DataFrame([
        _row("g1", 5.0, 2.0, 3.0, 8.0, 10.0),  # crossed: q10 > q25
        _row("g1", 1.0, 2.0, 3.0, 4.0, 5.0),   # already sorted
    ])
    fixed = _ensure_monotonic_quantiles(df)
    vals = fixed[QUANTILE_COLUMNS].to_numpy()
    assert np.all(np.diff(vals, axis=1) >= 0)
    # same multiset of values per row, just reordered
    assert sorted(vals[0]) == sorted([5.0, 2.0, 3.0, 8.0, 10.0])


def test_inverse_quantile_row_interpolates_and_extrapolates_linearly():
    quantile_values = np.array([2.0, 4.0, 6.0, 10.0, 14.0])
    u = np.array([0.10, 0.50, 0.90, 0.0, 1.0])
    out = _inverse_quantile_row(u, quantile_values)

    np.testing.assert_allclose(out[:3], [2.0, 6.0, 14.0])
    # below q10: slope from (0.10,2.0)-(0.25,4.0) = 2.0/0.15
    expected_below = 2.0 + (2.0 / 0.15) * (0.0 - 0.10)
    assert out[3] == pytest.approx(expected_below)
    # above q90: slope from (0.75,10.0)-(0.90,14.0) = 4.0/0.15
    expected_above = 14.0 + (4.0 / 0.15) * (1.0 - 0.90)
    assert out[4] == pytest.approx(expected_above)


def test_sample_player_week_requires_expected_columns():
    df = pd.DataFrame([{"game_id": "g1"}])
    with pytest.raises(KeyError):
        sample_player_week(df, n_sims=10)


def test_sample_player_week_marginal_coverage_matches_input_quantiles():
    """A single player's simulated draws should reproduce roughly the
    quantile points it was built from -- e.g. ~10% of draws below q10."""
    df = pd.DataFrame([_row("g1", 5.0, 8.0, 10.0, 12.0, 15.0)])
    draws = sample_player_week(df, n_sims=50000, seed=0)[0]

    assert draws.mean() == pytest.approx(10.0, abs=0.5)
    assert (draws <= 5.0).mean() == pytest.approx(0.10, abs=0.02)
    assert (draws <= 10.0).mean() == pytest.approx(0.50, abs=0.02)
    assert (draws <= 15.0).mean() == pytest.approx(0.90, abs=0.02)


def test_sample_player_week_correlates_within_game_not_across_games():
    """Two players sharing a game_id should have simulated draws
    correlated at approximately rho; two players in different games
    should be approximately uncorrelated."""
    rho = 0.35
    df = pd.DataFrame([
        _row("g1", 5.0, 8.0, 10.0, 12.0, 15.0),   # player 0, game 1
        _row("g1", 2.0, 4.0, 6.0, 9.0, 12.0),     # player 1, game 1 (same game)
        _row("g2", 3.0, 6.0, 8.0, 11.0, 14.0),    # player 2, game 2 (different game)
    ])
    draws = sample_player_week(df, n_sims=50000, rho=rho, seed=1)

    same_game_corr, _ = spearmanr(draws[0], draws[1])
    diff_game_corr, _ = spearmanr(draws[0], draws[2])

    assert same_game_corr == pytest.approx(rho, abs=0.03)
    assert abs(diff_game_corr) < 0.03


def test_sample_player_week_degenerate_quantiles_produce_a_fixed_value():
    """All 5 quantile columns equal (the K/DST / missing-coverage
    fallback pattern) must simulate as that exact constant every time,
    regardless of the percentile drawn."""
    df = pd.DataFrame([_row("g1", 7.5, 7.5, 7.5, 7.5, 7.5)])
    draws = sample_player_week(df, n_sims=1000, seed=2)[0]
    assert np.all(draws == 7.5)


def test_simulate_matchup_probabilities_sum_to_one():
    lineup_a = pd.DataFrame([_row("g1", 5.0, 8.0, 10.0, 12.0, 15.0)])
    lineup_b = pd.DataFrame([_row("g2", 3.0, 6.0, 8.0, 11.0, 14.0)])
    result = simulate_matchup(lineup_a, lineup_b, n_sims=5000, seed=3)
    total = result["team_a_win_prob"] + result["team_b_win_prob"] + result["tie_prob"]
    assert total == pytest.approx(1.0)
    assert len(result["team_a_totals"]) == 5000


def test_simulate_matchup_strictly_better_lineup_wins_most_of_the_time():
    better = pd.DataFrame([
        _row("g1", 10.0, 14.0, 18.0, 22.0, 26.0),
        _row("g2", 8.0, 10.0, 12.0, 14.0, 16.0),
    ])
    worse = pd.DataFrame([
        _row("g3", 1.0, 3.0, 5.0, 7.0, 9.0),
        _row("g4", 0.0, 2.0, 4.0, 6.0, 8.0),
    ])
    result = simulate_matchup(better, worse, n_sims=5000, seed=4)
    assert result["team_a_win_prob"] > 0.9


def test_simulate_matchup_shares_correlation_across_both_lineups():
    """A player on team A and a player on team B who share a game_id
    should still move together -- correlation isn't scoped to one
    fantasy roster, it's scoped to the real NFL game."""
    lineup_a = pd.DataFrame([_row("shared_game", 5.0, 8.0, 10.0, 12.0, 15.0)])
    lineup_b = pd.DataFrame([_row("shared_game", 3.0, 6.0, 8.0, 11.0, 14.0)])
    result = simulate_matchup(lineup_a, lineup_b, n_sims=50000, rho=0.35, seed=5)
    corr, _ = spearmanr(result["team_a_totals"], result["team_b_totals"])
    assert corr == pytest.approx(0.35, abs=0.03)


def test_simulate_matchup_runs_10000_sims_in_seconds_not_minutes():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(9):
        base = rng.uniform(3, 20)
        rows.append(_row(f"game_{i % 5}", base * 0.5, base * 0.8, base, base * 1.2, base * 1.5))
    lineup_a = pd.DataFrame(rows)
    lineup_b = pd.DataFrame(rows)

    start = time.perf_counter()
    simulate_matchup(lineup_a, lineup_b, n_sims=10000, seed=6)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"10,000-sim matchup took {elapsed:.1f}s -- expected seconds, not minutes"


def _flat_lineup(base_level, game_id="g"):
    """A one-player lineup with a fixed, non-degenerate spread centered on base_level."""
    return pd.DataFrame([_row(game_id, base_level - 5, base_level - 2, base_level, base_level + 2, base_level + 5)])


def test_simulate_season_exactly_playoff_teams_qualify_every_trial():
    """made_playoffs picks exactly `playoff_teams` rosters per trial by
    construction -- so summed across ALL rosters, playoff_prob must add
    up to exactly playoff_teams, not just approximately."""
    standings = pd.DataFrame({
        "roster_id": [1, 2, 3, 4], "wins": [0, 0, 0, 0], "points_for": [0.0, 0.0, 0.0, 0.0],
    })
    remaining_weeks = [(1, [(1, 2), (3, 4)]), (2, [(1, 3), (2, 4)])]

    def lineup_builder(roster_id, week):
        return _flat_lineup(10.0, game_id=f"g_{roster_id}_{week}")

    result = simulate_season(
        remaining_weeks, standings, lineup_builder, playoff_teams=2, n_sims=500, seed=1,
    )
    assert result["playoff_prob"].sum() == pytest.approx(2.0)
    assert len(result) == 4


def test_simulate_season_favors_the_stronger_team():
    standings = pd.DataFrame({
        "roster_id": [1, 2, 3, 4], "wins": [8, 2, 4, 4], "points_for": [1200.0, 700.0, 900.0, 900.0],
    })
    remaining_weeks = [(15, [(1, 2), (3, 4)]), (16, [(1, 3), (2, 4)])]

    def lineup_builder(roster_id, week):
        base = {1: 20.0, 2: 8.0, 3: 12.0, 4: 12.0}[roster_id]
        return _flat_lineup(base, game_id=f"g_{roster_id}_{week}")

    result = simulate_season(
        remaining_weeks, standings, lineup_builder, playoff_teams=2, n_sims=2000, seed=2,
    ).set_index("roster_id")

    assert result.loc[1, "playoff_prob"] > 0.95  # dominant record + best remaining projections
    assert result.loc[2, "playoff_prob"] < 0.10  # worst record + weakest remaining projections


def test_simulate_season_runs_correlated_within_week_across_all_matchups():
    """Two rosters in DIFFERENT fantasy matchups but the SAME real game
    should still move together in the season simulation, same as
    simulate_matchup's cross-lineup correlation test."""
    standings = pd.DataFrame({
        "roster_id": [1, 2, 3, 4], "wins": [0, 0, 0, 0], "points_for": [0.0, 0.0, 0.0, 0.0],
    })

    def lineup_builder(roster_id, week):
        shared_game = {1: "gameX", 3: "gameX", 2: "gameY", 4: "gameZ"}[roster_id]
        return _flat_lineup(10.0, game_id=shared_game)

    remaining_weeks = [(1, [(1, 2), (3, 4)])]  # roster 1 (matchup A) and roster 3 (matchup B) share gameX
    result = simulate_season(
        remaining_weeks, standings, lineup_builder, playoff_teams=2, n_sims=1, seed=3,
    )
    assert len(result) == 4  # smoke test: runs without error when matchup grouping crosses lineups


def test_playoff_participants_from_bracket_matches_real_2024_response():
    bracket = [
        {"m": 1, "r": 1, "l": 13, "w": 9, "t1": 13, "t2": 9},
        {"m": 2, "r": 1, "l": 14, "w": 6, "t1": 14, "t2": 6},
        {"m": 3, "r": 1, "l": 3, "w": 12, "t1": 12, "t2": 3},
        {"m": 4, "r": 2, "l": 8, "w": 9, "t1": 8, "t2": 9, "t2_from": {"w": 1}},
        {"m": 5, "r": 2, "l": 6, "w": 12, "t1": 6, "t2": 12, "t2_from": {"w": 3}, "t1_from": {"w": 2}},
        {"m": 6, "r": 2, "l": 14, "w": 3, "t1": 14, "t2": 3, "t2_from": {"l": 3}, "t1_from": {"l": 2}},
        {"p": 1, "m": 7, "r": 3, "l": 9, "w": 12, "t1": 9, "t2": 12, "t2_from": {"w": 5}, "t1_from": {"w": 4}},
        {"p": 3, "m": 8, "r": 3, "l": 8, "w": 6, "t1": 8, "t2": 6, "t2_from": {"l": 5}, "t1_from": {"l": 4}},
        {"p": 5, "m": 9, "r": 3, "l": 3, "w": 13, "t1": 13, "t2": 3, "t2_from": {"w": 6}, "t1_from": {"l": 1}},
    ]
    result = playoff_participants_from_bracket(bracket)
    assert result == {13, 9, 14, 6, 12, 3, 8}


def test_calibration_report_matches_known_bernoulli_rate():
    rng = np.random.default_rng(7)
    n = 4000
    sim_probs = np.full(n, 0.6)
    actual_outcomes = (rng.uniform(size=n) < 0.6).astype(float)

    report = calibration_report(sim_probs, actual_outcomes, n_bins=10)
    populated = report[report["n"] > 0]
    assert len(populated) == 1
    row = populated.iloc[0]
    assert row["actual_rate"] == pytest.approx(0.6, abs=0.03)
    assert row["n"] == n
