"""
FanTeasy Stats -- Phase 6.5 step 9: game-environment sampling and matchup
simulation.

Per PHASE_2B_6_SPEC.md, the part that matters is correlating players who
share a real NFL game -- ignoring it makes simulated totals cluster too
tightly around the mean, which overstates confidence in every win
probability the simulator produces. The approach here is the spec's
"option 1": simulate the game environment first, then each player's share
of it, so two players in the same real game move together automatically
because they share a draw.

Mechanism (a one-factor Gaussian copula): every player-week row draws a
percentile
    u = Phi(sqrt(rho)*z_game + sqrt(1-rho)*z_player)
where z_game is ONE standard-normal draw shared by every row with the
same `game_id`, and z_player is that row's own idiosyncratic draw. Any
two rows sharing a game_id have percentile correlation exactly `rho`;
rows in different games are independent. u is then mapped through THAT
row's own calibrated quantiles to a simulated point total -- so a
shootout lifts everyone in it, each according to their own range of
outcomes, exactly as the spec describes.

`rho` is a fixed constant (GAME_ENVIRONMENT_RHO), not fit to data --
Phase 6.5's own spec defers "measure the correlations directly" (its
option 2) to later work and says to start with the simpler option 1.

Player distributions come from src/model.py's CQR-calibrated quantiles
(pred_q10_cqr, pred_q25_cqr, pred_q50, pred_q75_cqr, pred_q90_cqr -- q50
is untouched by CQR by construction). Two documented limitations from
that calibration step carry through into every simulation run here (see
PROJECT_CONTEXT.md's Phase 6 findings for the numbers):
  - The correction overshoots asymmetrically: ceilings ended up MORE
    conservative than floors at every position, since the floor was the
    worse-calibrated side before correction. Simulated booms are
    therefore a bit more muted than simulated busts, relative to what
    each side's own miscoverage alone would call for.
  - TE's lowest-predicted-usage tercile (where the position's 19.4%
    zero-inflation concentrates) remains under-corrected relative to the
    rest of TE's range -- a backup/committee TE's simulated floor is
    still a bit optimistic compared to reality.

Neither limitation is fixed here; both are inherited, not introduced, by
this module.

K/DST and any player missing model coverage: src/model.py's projection
model is QB/RB/WR/TE only (see its module docstring and CLAUDE.md's scope
boundaries). To include a K/DST or missing-coverage starter in a
simulated lineup, set all 5 quantile columns to the SAME constant (e.g.
Sleeper's own point projection, scored via
model.sleeper_projected_points()) -- interpolating between 5 identical
points always returns that constant regardless of the percentile drawn,
so it contributes a fixed, non-modeled amount without any special-casing
in this module. This mirrors the dashboard's own existing convention of
showing Sleeper's K/DST projection labeled as Sleeper's, not the model's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

QUANTILE_COLUMNS = ["pred_q10_cqr", "pred_q25_cqr", "pred_q50", "pred_q75_cqr", "pred_q90_cqr"]
QUANTILE_POINTS = np.array([0.10, 0.25, 0.50, 0.75, 0.90])

GAME_ENVIRONMENT_RHO = 0.35


def _ensure_monotonic_quantiles(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts each row's 5 quantile columns into non-decreasing order.

    Needed again here even though src/model.py's fix_quantile_crossing()
    already fixed crossing WITHIN each CQR interval pair (q10 <= q90,
    q25 <= q75): the 10-90 and 25-75 pairs are widened by DIFFERENT
    constants (see PROJECT_CONTEXT.md), so q25_cqr can end up below
    q10_cqr, or q75_cqr above q90_cqr, even though each pair is
    individually monotonic after its own widening. An inverse-CDF built
    from non-monotonic points isn't a real inverse-CDF.
    """
    out = preds.copy()
    sorted_vals = np.sort(out[QUANTILE_COLUMNS].to_numpy(dtype=float), axis=1)
    for i, col in enumerate(QUANTILE_COLUMNS):
        out[col] = sorted_vals[:, i]
    return out


def _inverse_quantile_row(u: np.ndarray, quantile_values: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear inverse-CDF for one player-week, built from its 5
    known (percentile, value) points. Interpolates within [0.10, 0.90];
    linearly EXTRAPOLATES beyond it using the outermost known slope --
    np.interp's default silently flattens outside the given range, which
    would understate tail variance exactly where a Monte Carlo
    simulation needs it most (the 1st or 99th percentile draws).
    """
    out = np.interp(u, QUANTILE_POINTS, quantile_values)
    below = u < QUANTILE_POINTS[0]
    if below.any():
        slope = (quantile_values[1] - quantile_values[0]) / (QUANTILE_POINTS[1] - QUANTILE_POINTS[0])
        out[below] = quantile_values[0] + slope * (u[below] - QUANTILE_POINTS[0])
    above = u > QUANTILE_POINTS[-1]
    if above.any():
        slope = (quantile_values[-1] - quantile_values[-2]) / (QUANTILE_POINTS[-1] - QUANTILE_POINTS[-2])
        out[above] = quantile_values[-1] + slope * (u[above] - QUANTILE_POINTS[-1])
    return out


def sample_player_week(
    preds: pd.DataFrame,
    n_sims: int = 10000,
    rho: float = GAME_ENVIRONMENT_RHO,
    seed: int | None = None,
) -> np.ndarray:
    """
    Monte Carlo samples every player-week's fantasy points, n_sims times,
    correlated within the same real NFL game (see module docstring for
    the game-environment mechanism).

    Args:
        preds: one row per player-week. Must have `game_id` and every
            column in QUANTILE_COLUMNS.
        n_sims: number of Monte Carlo draws.
        rho: game-environment correlation. Fixed, not tuned.
        seed: for reproducible draws (tests; pass None for real use).

    Returns:
        (len(preds), n_sims) array; row i is preds row i's simulated
        points across all n_sims draws (input row order preserved).
    """
    missing = [c for c in ["game_id"] + QUANTILE_COLUMNS if c not in preds.columns]
    if missing:
        raise KeyError(f"sample_player_week: preds is missing columns {missing}")

    preds = _ensure_monotonic_quantiles(preds)
    rng = np.random.default_rng(seed)
    n_rows = len(preds)

    game_codes, game_index = pd.factorize(preds["game_id"])
    n_games = len(game_index)

    z_game = rng.standard_normal(size=(n_games, n_sims))
    z_idio = rng.standard_normal(size=(n_rows, n_sims))

    a, b = np.sqrt(rho), np.sqrt(1 - rho)
    z_combined = a * z_game[game_codes] + b * z_idio
    u = norm.cdf(z_combined)

    quantile_values = preds[QUANTILE_COLUMNS].to_numpy(dtype=float)
    out = np.empty((n_rows, n_sims), dtype=np.float64)
    for i in range(n_rows):
        out[i] = _inverse_quantile_row(u[i], quantile_values[i])
    return out


def simulate_matchup(
    lineup_a: pd.DataFrame,
    lineup_b: pd.DataFrame,
    n_sims: int = 10000,
    rho: float = GAME_ENVIRONMENT_RHO,
    seed: int | None = None,
) -> dict:
    """
    Simulates one fantasy matchup n_sims times. lineup_a's and
    lineup_b's starters are sampled TOGETHER in a single
    sample_player_week() call, so correlation applies across BOTH
    fantasy rosters whenever their players share a real game_id (e.g.
    two managers who each started a player from the same real NFL game)
    -- a shootout can lift players on both sides of a fantasy matchup at
    once, exactly like the spec describes.

    Args:
        lineup_a, lineup_b: one row per STARTING player (not full
            roster), same required columns as sample_player_week.
        n_sims, rho, seed: passed through to sample_player_week.

    Returns:
        dict with team_a_win_prob, team_b_win_prob, tie_prob,
        team_a_mean, team_b_mean, team_a_totals, team_b_totals (the last
        two are the raw (n_sims,) arrays, for anyone who wants the full
        simulated distribution, not just the summary).
    """
    combined = pd.concat(
        [lineup_a.assign(_sim_team="a"), lineup_b.assign(_sim_team="b")], ignore_index=True
    )
    draws = sample_player_week(combined, n_sims=n_sims, rho=rho, seed=seed)

    is_a = (combined["_sim_team"] == "a").to_numpy()
    team_a_totals = draws[is_a].sum(axis=0)
    team_b_totals = draws[~is_a].sum(axis=0)

    return {
        "team_a_win_prob": float((team_a_totals > team_b_totals).mean()),
        "team_b_win_prob": float((team_a_totals < team_b_totals).mean()),
        "tie_prob": float((team_a_totals == team_b_totals).mean()),
        "team_a_mean": float(team_a_totals.mean()),
        "team_b_mean": float(team_b_totals.mean()),
        "team_a_totals": team_a_totals,
        "team_b_totals": team_b_totals,
    }


def calibration_report(
    sim_probs: np.ndarray, actual_outcomes: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Reliability check for simulated win probabilities: bins sim_probs
    into n_bins equal-width [0, 1] bins and reports the ACTUAL win rate
    within each bin against the bin's predicted mean. A well-calibrated
    simulator's actual rate tracks the predicted rate across bins -- e.g.
    among observations where the simulator said "~60%", about 60% should
    have actually gone that way.

    Args:
        sim_probs: one simulated win probability per observation. Each
            historical matchup should contribute TWO observations here,
            not one -- team_a's probability paired with whether team_a
            won, AND team_b's probability paired with whether team_b won
            -- the standard way to build a reliability diagram over
            paired outcomes without arbitrarily privileging one side.
        actual_outcomes: parallel 0/1 array, 1 if that observation's
            team actually won.
        n_bins: number of equal-width probability bins.

    Returns:
        DataFrame: bin_low, bin_high, predicted_mean, actual_rate, n.
    """
    sim_probs = np.asarray(sim_probs, dtype=float)
    actual_outcomes = np.asarray(actual_outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(sim_probs, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        rows.append({
            "bin_low": bins[b], "bin_high": bins[b + 1],
            "predicted_mean": sim_probs[mask].mean() if n else np.nan,
            "actual_rate": actual_outcomes[mask].mean() if n else np.nan,
            "n": n,
        })
    return pd.DataFrame(rows)
