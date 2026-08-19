# Phase 2b + Phase 6 — Feature and Model Spec

Working spec for the usage/efficiency feature layer and the projection model.
Read alongside `CLAUDE.md` and `PROJECT_CONTEXT.md`. Delete or fold into
`NOTEBOOK_OUTLINE.md` once both phases are complete.

---

## What this is actually for

**Not** "beat Sleeper every week." Weekly fantasy scoring is dominated by
touchdown variance, which is close to a coin flip conditional on opportunity.
No public model predicts it well, and building toward that target means
declaring failure at something that was never achievable.

The goal is **surfacing signal that Sleeper and other platforms don't show**:

1. **Opportunity quality, not just volume.** Sleeper shows targets. It doesn't
   show air-yards share, red-zone target share, or how those trend.
2. **Role trajectory.** A rising snap share three weeks running is actionable
   before it shows up in fantasy points. Platforms show the points.
3. **Regression candidates.** Players scoring far above or below what their
   opportunity implies. This is the highest-value analytical output here.
4. **League-specific reranking.** This league is 0.5 PPR with a stacking fumble
   penalty and return-yardage scoring. Public rankings are generic PPR or
   standard. Where those diverge is an edge available to nobody else, because
   it only exists for this league.

Sleeper's projection is a **reference point and a feature**, not the target to
beat. It encodes beat-writer and depth-chart information not present in
nflverse, so ignoring it discards real signal.

---

## Non-negotiable: point-in-time correctness

Every feature used to predict week N must be computable **before kickoff of
week N**, using only weeks < N.

This is the single most common way a fantasy model looks excellent in backtest
and performs like a coin flip live. `pandas.rolling()` **includes the current
row** — so a naive "trailing 3-week target share" contains week N's own target
share, and the model learns to read the answer.

**Requirements:**

- Every rolling/expanding feature is `.shift(1)`-ed within
  `groupby('player_id')` before the window is applied.
- Sort by `['player_id', 'season', 'week']` before any window operation.
- Season boundaries are **not** crossed by in-season windows. Prior-season
  aggregates go in their own columns with their own missingness.
- Write a leakage test in `tests/test_no_leakage.py`:
  - Build the feature table twice — once from the full dataset, once from a
    dataset truncated at week N-1.
  - Assert the week-N feature rows are **identical** between the two.
  - Run it for several N. If any feature differs, it leaks.
- The test must pass before any model is trained. A model trained on leaked
  features produces numbers that are worse than useless, because they look good.

---

## Phase 2b — Feature layer

New module: `src/usage.py`. Keep `features.py` for scoring; do not mix them.

Input: `data/processed/weekly_scored.parquet`, plus `pbp`, `snaps`, `schedule`,
`ngs_*` from the `data/raw/` cache.
Output: `data/processed/weekly_features.parquet` — one row per player-week,
QB/RB/WR/TE, `season_type == 'REG'` only.

### Function shape

One function per family, each taking and returning a DataFrame, each idempotent
(drop existing output columns before merging — notebook cells get re-run):

```python
def add_volume_features(df, pbp) -> pd.DataFrame: ...
def add_snap_features(df, snaps, crosswalk) -> pd.DataFrame: ...
def add_efficiency_features(df, pbp, ngs) -> pd.DataFrame: ...
def add_situational_features(df, pbp) -> pd.DataFrame: ...
def add_context_features(df, schedule) -> pd.DataFrame: ...
def add_rolling_features(df, windows=(3,)) -> pd.DataFrame: ...
def build_feature_table(...) -> pd.DataFrame: ...   # orchestrates the above
```

### Family 1 — Volume and share (per player-week)

Team denominators come from the player's own team that week.

- `targets`, `target_share` (targets / team pass attempts)
- `air_yards`, `air_yards_share`
- `wopr` = 1.5 × target_share + 0.7 × air_yards_share
- `carries`, `carry_share` (carries / team rush attempts)
- `touches` = carries + receptions, `touch_share`
- QB: `pass_attempts`, `dropbacks`, `designed_rush_attempts`, `scramble_rate`

### Family 2 — Snap share

Requires the `pfr_player_id` → `gsis_id` join (verified at 99.67% for
QB/RB/WR/TE; near-zero for o-line, which is out of scope).

- `offense_snaps`, `offense_pct`
- `snap_share_delta_3wk` — trailing 3-week change. Role trajectory is the
  point; a rising snap share leads fantasy production.

### Family 3 — Efficiency

- `adot` = air_yards / targets
- `yac_per_reception`, `yards_per_route_run` if route data is available
- `catch_rate`, `yards_per_target`, `yards_per_carry`
- QB: `cpoe`, `epa_per_dropback` (already in pbp)
- NGS: `avg_separation`, `avg_cushion`, `time_to_throw` — note NGS is
  2016+ and some fields are sparse

### Family 4 — Situational (the differentiated set)

This is where the edge lives. Platforms surface almost none of it.

- `rz_targets`, `rz_carries` (`yardline_100 <= 20`), plus shares
- `inside_10_touches`, `inside_5_carries` — where touchdowns actually come from
- `goal_line_carry_share`
- `third_down_target_share`, `two_minute_target_share`
- `pass_rate_over_expected` faced (team-level game script proxy)

### Family 5 — Game context

From `load_schedules()`, which carries betting columns
(`spread_line`, `total_line`, moneylines) — verified available.

- `team_implied_total` = total_line/2 − spread_line/2 (sign per home/away —
  get this right, it's easy to invert)
- `game_total`, `spread`, `is_home`, `days_rest`
- `roof`, `surface`, `temp`, `wind` (weather only matters outdoors)
- Opponent defensive strength **by position**, computed on prior weeks only —
  fantasy points allowed to RB/WR/TE, opponent-adjusted if practical

### Family 6 — Rolling aggregates

For every continuous feature above, produce:

- `<feat>_ewm3` — exponentially weighted mean, ~3-week half-life (recency)
- `<feat>_std` — season-to-date expanding mean (stability)
- `games_played` — so the model knows how much to trust each

All `.shift(1)`-ed. Let the model learn the weighting rather than picking one
window by hand.

### Expected fantasy points (xFP) — build this deliberately

The most valuable single output in this phase, and the thing no platform shows.

For each player-week, compute what their opportunity was *worth*, independent
of whether it converted:

1. From pbp across the training seasons, compute league-average fantasy points
   per opportunity, bucketed by opportunity type and field position — e.g.
   expected points per target at each air-yards band, per carry inside the 5,
   per carry between the 20s.
2. Apply those rates to the player's actual opportunities that week.
3. `xfp` = sum. `fp_over_expected` = `custom_points` − `xfp`.

Use **this league's scoring** (`compute_custom_score`) for the rates, not
generic PPR. That's what makes it specific to you.

Two uses:
- **As a model feature.** Trailing xFP predicts future points better than
  trailing actual points, because it strips touchdown noise.
- **As a standalone dashboard panel.** Season-long `fp_over_expected` is a
  regression-candidate list: who's been lucky, who's been unlucky. Genuinely
  actionable for buy-low/sell-high, and not available on Sleeper.

### Acceptance criteria for Phase 2b

- `tests/test_no_leakage.py` passes
- Feature table has no unexpected nulls — every null is explained (week 1 has
  no history, NGS is sparse pre-2016, etc.) and documented
- Distributions sanity-checked in `03_usage_features.ipynb`: target shares sum
  to roughly 1.0 per team-week, snap shares are in [0, 1], aDOT is plausible
  by position
- `02_custom_scoring.ipynb` still reports **0 mismatches** after any change

---

## Phase 6 — Projection model

### Target

`custom_points` from `compute_custom_score()`. **Not** `fantasy_points_ppr` —
that's full PPR and this league is 0.5.

### Two model formulations, in this order

**A. Direct.** Predict `custom_points` for week N from the feature table.
Simple, works, is the honest baseline for your own model.

**B. Residual.** Predict `custom_points − sleeper_projection`. This asks the
easier and more useful question: *where is Sleeper wrong?* Sleeper already
encodes injury and depth-chart information nflverse lacks, so starting from its
number means you begin at the baseline rather than climbing to it.

Build A first, then B, and compare. If B wins, that's the model — and the
residual itself is an analytical output worth surfacing.

### Model

- Start with **LightGBM** (fast, handles missing values natively, strong on
  tabular). XGBoost as a cross-check.
- **Separate models per position.** QB, RB, WR, TE have different feature
  importances and different scoring drivers. One model with position as a
  feature underperforms four models.
- Optuna for hyperparameter search, but only after the feature table is stable.
  Tuning a leaky feature set optimizes the leak.

### Validation — time-series only

- **Expanding-window walk-forward.** Train on weeks 1..N-1, predict week N,
  advance. Never shuffle. Never use `KFold`.
- Hold out an entire season (2025) as a final untouched test if data volume
  permits; otherwise hold out the last 4 weeks.
- Report metrics **per position**, not pooled. Pooled numbers hide that TE
  prediction is much harder than QB.

### Baselines to report alongside the model

Always all four. A model that beats none of them isn't ready.

1. Season-to-date average
2. Trailing 3-week average
3. Sleeper's projection
4. xFP (trailing average of expected points)

### Metrics

- **MAE and RMSE** per position. MAE is more interpretable for fantasy; RMSE
  punishes the blowup weeks that decide matchups.
- **Spearman correlation** — for lineup decisions, ranking matters more than
  absolute points.
- **Calibration plot** — predicted vs actual, decile-binned. A model that's
  right on average but overconfident at the extremes is dangerous for
  start/sit calls.
- **Prediction intervals**, via quantile regression (LightGBM supports
  `objective='quantile'`). Floor/ceiling is more useful than a point estimate
  for lineup decisions, and it's honest about the noise.

### Honesty requirements

- If the model does not beat Sleeper, **say so in the dashboard.** Show both,
  labeled. The project's stated principle is no fake data; an overstated model
  is the same failure in a different costume.
- Report where the model is *systematically* better or worse — by position,
  by projected volume, by role type. "Better on high-target-share WRs, worse
  on committee RBs" is a real finding and more useful than a single number.
- SHAP values for explainability. If the top feature is something that
  shouldn't matter, that's a leak or a bug, not a discovery.

---

---

## Phase 6.5 — Monte Carlo simulation

### What this is, in plain terms

A **Monte Carlo simulation** means: instead of computing one answer, you play
out the situation thousands of times with randomness, then count how the
results came out. If you simulate a week 10,000 times and your team wins 3,800
of them, your win probability is 38%.

This is the piece that answers questions no fantasy platform shows you.

### Why it's worth building

A single projected score hides everything useful. "You're projected to lose by
12" doesn't tell you whether that's hopeless or a coin flip with one big game.
Simulation gives you:

- **Matchup win probability.** Play out this week's matchup 10,000 times, count
  the wins.
- **Playoff odds.** Play out the whole rest of the season, count how often each
  team makes the playoffs.
- **Better start/sit calls.** If you're a big underdog, you don't want the
  player with the highest average — you want the one most likely to score 25+.
  Averages can't tell you that. Simulation can.

### Prerequisite: distributions, not point estimates

A **distribution** here means the full range of outcomes a player could produce
and how likely each is, rather than a single number. Instead of "Player X will
score 14.2," you want "usually 10-18, occasionally 30, sometimes 4."

Phase 6 already specifies **quantile regression** for this — a model that
predicts several points along the range (say the 10th, 25th, 50th, 75th, and
90th percentile) instead of just the middle. LightGBM does this with
`objective='quantile'`. Those predicted percentiles are what the simulation
samples from.

### The hard part: players are not independent

The naive approach samples each of your 9 starters separately, as if their
outcomes had nothing to do with each other. **That is wrong, and it matters.**

- A QB and his own top receiver rise and fall together. A 4-touchdown passing
  game means somebody caught those touchdowns.
- Two players in the same game share a game environment. A shootout lifts
  everyone; a defensive slog suppresses everyone.
- Two running backs on the same team compete for the same carries — one going
  off usually means the other didn't.

**Correlation** is just the term for "these move together." Ignoring it makes
your simulated totals too tightly clustered around the average, which makes
your win probabilities overconfident — you'll show 85% when the truth is 65%.

Two ways to handle it, easiest first:

1. **Simulate the game environment first.** For each NFL game, draw a total
   number of points scored (using the Vegas total as the center). Then draw
   each player's share of that environment. Players in the same game
   automatically move together because they share the draw.
2. **Measure the correlations directly** from historical data — how much did a
   QB's fantasy points and his WR1's fantasy points move together, week to
   week — and sample using those measured relationships.

Start with option 1. It's simpler, it captures most of the effect, and it uses
the Vegas total already in the feature table.

### Validating the simulator

The simulation needs its own accuracy check, separate from the model's.

The check is **calibration**: when you say something is 60% likely, does it
happen about 60% of the time? Collect all your historical 60% predictions and
count how many came true. Plot predicted probability against actual frequency —
a well-calibrated simulator sits close to the diagonal line.

A simulator that's confidently wrong is worse than no simulator, because it
looks authoritative. Do not ship this panel without the calibration plot.

### Deliverables

New module `src/simulate.py`, new notebook `06_monte_carlo.ipynb`:

```python
def sample_player_week(preds, n_sims=10000) -> np.ndarray: ...
def simulate_matchup(lineup_a, lineup_b, n_sims=10000) -> dict: ...
def simulate_season(schedule, rosters, n_sims=10000) -> pd.DataFrame: ...
def calibration_report(sim_probs, actual_outcomes) -> pd.DataFrame: ...
```

Dashboard outputs, all of which are new information versus Sleeper:

- Win probability on each matchup card
- Playoff odds table
- Floor / ceiling ranges on player cards (10th and 90th percentile)
- "Most likely to exceed X points" for start/sit decisions

### Acceptance criteria

- Correlation between teammates is handled, not ignored — and the code says
  which method was used and why
- Calibration plot exists and is shown alongside any probability the dashboard
  displays
- Simulated win probabilities beat a naive baseline (whoever has the higher
  projected total wins) on historical weeks
- 10,000 simulations of one week runs in seconds, not minutes

### Note on Markov chains

Considered and rejected for the projection pipeline. A **Markov chain** models
a system that moves between states, where the next state depends only on the
current one. Weekly fantasy output doesn't work that way — next week depends on
opponent, health, and game script, not on "what state the player was in last
week."

The one place Markov chains genuinely fit football is drive-level modeling
(down, distance, field position → next state → scoring outcome). But nflverse
already ships EPA and win probability built from far better models trained on
decades of data. Rebuilding that produces a worse version of a column already
in `pbp`. Fine as a clearly-labeled side project; not part of this pipeline.

---

## Order of work

1. `src/usage.py` volume + snap features → leakage test → commit
2. Situational + context features → sanity-check distributions → commit
3. xFP → validate against actual points at season level → commit
4. Rolling aggregates → re-run leakage test → commit
5. `03_usage_features.ipynb` exercising all of it → commit
6. Phase 6 model A, walk-forward CV, all four baselines → commit
7. Model B (residual), compare → commit
8. Quantile models for floor/ceiling → SHAP → commit
9. `src/simulate.py` — game-environment sampling, matchup simulation → commit
10. Calibration report → season simulation for playoff odds → commit

Small commits. Re-run the 02 validation after anything that touches
`features.py`.
