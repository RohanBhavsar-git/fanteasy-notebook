# FanTeasy Stats — Python Notebook Outline

A rubric for building the data pipeline that feeds the dashboard's "advanced analytics" panels. The deliverable is a single JSON file (`player_advanced_stats.json`) that the dashboard fetches on load.

> **Updated August 2026.** The nflverse client changed from `nfl_data_py` (deprecated)
> to `nflreadpy`. Phases 1, 2a, 2b, 3', 6, 6.5, 7, and 8 are complete and verified
> (Phase 6 investigated and deliberately not shipped as a model — see its section).
> Phases 4-5 (radar/heatmap) are still design sketches written before anything ran —
> treat those two sections' snippets as intent, not tested code. Check
> `PROJECT_CONTEXT.md`'s Verification status table before treating any other claim
> in this file as settled either.
>
> **`PHASE_2B_6_SPEC.md` (repo root) supersedes the Phase 2 and Phase 6 sections
> below.** It carries the point-in-time correctness rules, the expected-fantasy-points
> work, and a new **Phase 6.5 — Monte Carlo simulation** (matchup win probability,
> playoff odds, floor/ceiling ranges). Read it before starting either phase.

## Goal

Produce one stable, versioned JSON file per week that contains per-player:
1. **Usage trend signal** (Phase 3' — current opportunity share, a
   normalized trend signal, and a rising/falling/stable direction label;
   replaces the original role-classification idea below, see Phase 3')
2. **Position-profile radar metrics** (0-100 scaled)
3. **Field heatmap zones** (target/run/pass location frequencies)
4. **Your custom projection** for the upcoming week

The dashboard already has hooks waiting to consume this file — search `// Hook for your custom model` and `state.advancedStats` in `index.html` to see where it slots in.

---

## Stack recommendation

| Tool | Why |
|---|---|
| **`nflreadpy`** (maintained Python client for nflverse) | Free, public, actively maintained. Play-by-play, weekly stats, schedules, snap counts, NGS, player IDs. Replaced `nfl_data_py`, which nflverse deprecated in 2025 with no further updates planned. |
| **`polars`** | What `nflreadpy` returns. `src/ingest.py` calls `.to_pandas()` at the boundary so downstream code stays pandas — but you could go Polars end-to-end later if you want the speed. |
| **`pandas`** | Standard for tabular work. Everything after ingestion is pandas. |
| **`pyarrow`** | Needed for both the local parquet cache and Polars → pandas conversion. Easy to forget; nothing works without it. |
| **`scikit-learn`** | Pulled in as `lightgbm`'s sklearn-wrapper dependency (`LGBMRegressor`), not for role classification — that idea was dropped, see Phase 3'. |
| **`xgboost`** or **`lightgbm`** | For the custom projection model. Tree-based handles tabular fantasy data well. |
| **`matplotlib` / `seaborn`** | Only for your own EDA — the dashboard renders everything client-side. |
| **GitHub Actions** | Two workflows, not a single scheduled notebook run — see Phase 8. `weekly-update.yml` runs Tuesday mornings (after MNF stats post) and commits the JSON; `retrain.yml` is manual-only. |

Don't bother with PFF/Sportradar paid APIs unless you specifically want contested-catch% or YPRR. Everything else is in nflverse.

---

## Data sources — direct links

### Primary (free, public)

| Source | URL | What you get |
|---|---|---|
| **nflverse hub** | https://nflverse.nflverse.com/ | The master umbrella project. All R/Python NFL data packages live here. |
| **`nflreadpy` docs** | https://nflreadpy.nflverse.com/ | Function reference. Start at the `load_*` API page — signatures differ from the old `import_*` names. |
| **`nflreadpy` (PyPI)** | https://pypi.org/project/nflreadpy/ | `pip install nflreadpy`. |
| **`nfl_data_py` (deprecated)** | https://github.com/cooperdff/nfl_data_py | Historical reference only. Do not add as a dependency. |
| **nflverse-data releases** | https://github.com/nflverse/nflverse-data/releases | Raw parquet/CSV files if you want to bypass the Python package. |
| **Sleeper API docs** | https://docs.sleeper.com/ | League, rosters, projections, players, drafts. Your dashboard already uses this. |
| **Sleeper player endpoint** | `https://api.sleeper.app/v1/players/nfl` | All NFL players with injury_status, depth_chart_position, etc. |
| **Sleeper weekly projections** | `https://api.sleeper.app/v1/projections/nfl/regular/{year}/{week}` | Per-player projection — your model's benchmark to beat. **Use this exact host + path**: it's what `index.html` (~line 1924) uses, so notebook and dashboard agree on the baseline. `api.sleeper.com/projections/nfl/{year}/{week}?season_type=regular` is a newer alternate form kept only as a fallback in `src/ingest.py`. |
| **NFL Next Gen Stats** | https://nextgenstats.nfl.com/ | Browser-only, but the data is in nflverse via `nfl.load_nextgen_stats()`. Air yards, separation, time to throw. |

### Supplementary (free)

| Source | URL | What you get |
|---|---|---|
| **ESPN hidden JSON** | https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard | Live scores, game state, weather conditions in some games. Already in your dashboard. |
| **ESPN injuries endpoint** | https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/injuries | Backup injury source if Sleeper's looks stale. |
| **Open-Meteo** | https://open-meteo.com/ | Weather forecasts for outdoor games. Free, no auth. Already in your dashboard. |
| **Pro-Football-Reference** | https://www.pro-football-reference.com/ | Historical reference, but no free API. Scraping risk-tolerant only. |

### Paid (skip unless specifically needed)

| Source | URL | When you'd pay |
|---|---|---|
| **PFF** | https://www.pff.com/ | Grades, contested-catch %, route participation, YPRR. ~$40/mo. |
| **SportRadar** | https://sportradar.com/ | Enterprise; pricey. Not relevant for personal projects. |
| **FantasyData** | https://fantasydata.com/api | Has injury-news history with timestamps. ~$30/mo for the lowest tier. |
| **OddsAPI** | https://the-odds-api.com/ | Vegas lines for implied team totals. Free tier exists (500 requests/mo). |

### Community resources worth bookmarking

| Source | URL | Why |
|---|---|---|
| **nflverse Discord** | https://discord.gg/5Er2FBnnQa | Active community, fast answers when `nflreadpy` or the underlying data feeds break. |
| **r/fantasyfootball /dataisbeautiful** | https://reddit.com/r/fantasyfootball | Project showcase + feedback if you post yours. |
| **Open Source Football** | https://www.opensourcefootball.com/ | Tutorial blog using nflverse — great for learning patterns. |
| **Ben Baldwin's blog** | https://www.nflfastr.com/ (and his Twitter) | Builds + explains the nflfastR package; foundational reading. |

---

## Modern methods worth learning (corporate-relevant in 2026)

This project is a great vehicle for picking up techniques that show up in real data-science work. Some honest tiering:

### Strongly suggest — fit this project naturally

**Gradient-boosted trees (XGBoost / LightGBM) with proper time-series CV.** Still the dominant tabular regression approach in industry. Use `TimeSeriesSplit` from sklearn — never train on Week 10 to predict Week 5, or you'll fool yourself. Resource: [LightGBM docs](https://lightgbm.readthedocs.io/), [XGBoost docs](https://xgboost.readthedocs.io/).

**SHAP for model explainability.** After your model predicts "Mahomes 22.4," SHAP breaks that down: "+3.2 for matchup quality, +1.8 for recent trend, -0.4 for weather." Every serious ML team uses this. The `shap` library plugs straight into tree models. Could power a "Why this projection?" panel on the player detail page. Resource: https://shap.readthedocs.io/.

**Quantile regression / prediction intervals.** Instead of point estimates, predict ranges: "between 14.8 and 31.2 with 80% confidence." More useful for lineup decisions (knowing the floor matters more than knowing the median for a flex play). Train multiple LightGBM models at different quantiles. Already in my JSON schema as `rohan_floor` / `rohan_ceiling`. Resource: search "lightgbm quantile regression objective".

**Conformal prediction.** A 2024-2026 hot topic — produces *calibrated* prediction intervals with formal coverage guarantees. The `mapie` library wraps it cleanly: https://mapie.readthedocs.io/. Worth a section in your project README if you implement it.

### Industry-standard tooling to know

**Optuna for hyperparameter tuning.** Bayesian optimization instead of grid/random search. Every shop has moved off `GridSearchCV`. Plug-and-play with XGBoost. https://optuna.org/.

**MLflow for experiment tracking.** As you iterate weekly on the projection model, MLflow logs each run's feature set, hyperparams, and validation metrics. Hiring managers expect familiarity with MLflow or DVC. https://mlflow.org/. Alternatively https://dvc.org/ for data versioning.

**FastAPI for serving models.** If you want to serve predictions on-demand instead of pre-computing them in a weekly JSON, FastAPI is the standard. https://fastapi.tiangolo.com/. Optional here, but a strong portfolio piece.

### Frontier — interesting, more lift to learn

**PyMC / Bambi for Bayesian hierarchical models.** Naturally handles "player nested in team nested in division" structure. More principled but steeper learning curve. https://www.pymc.io/.

**TabNet / FT-Transformer.** Neural nets for tabular data. Recent papers claim they beat gradient boosting; in practice they usually don't, but the *attention mechanisms* are transferable knowledge.

### Honest warning about hype

**Don't shoehorn LLMs/RAG/agents into this project just because they're trendy.** A projection model is a tabular regression problem. Forcing an LLM in signals "followed hype" rather than "picked the right tool" — exactly the *opposite* of what hiring managers want to see on a senior-trending data-science resume. The exception: a separate "natural-language assistant" feature that lets users ask "who's my best flex play this week?" is a legitimate LLM use case, but it's a *different* feature than the projection model.

### My specific picks for this project

For maximum corporate-relevance ROI:
1. **XGBoost or LightGBM** projection model with time-series CV
2. **SHAP** explanations — surface them on player detail pages
3. **Quantile regression** for floor/ceiling
4. **Optuna** for tuning
5. **MLflow** for experiment tracking

That stack is what a mid-to-senior data scientist would build today. Ship those five well and the project speaks for itself.

---

## Project structure

Updated to match what actually exists as of Phase 8 (the original sketch below
predated Phase 1 and named files that were never built this way — `src/`'s
actual module list, the notebook names, and the two-workflow `.github/` layout
all differ from that first guess; this is the real structure):

```
fanteasy-notebook/
├── notebooks/
│   ├── 01_data_ingestion.ipynb   # Phase 1
│   ├── 02_custom_scoring.ipynb   # Phase 2a
│   ├── 03_usage_features.ipynb   # Phase 2b steps 1-5
│   ├── 04_usage_trends.ipynb     # Phase 3'
│   └── 07_export_json.ipynb      # Phase 7 (manual/exploratory export -- weekly-update.yml
│                                  #           is the automated path now, see Phase 8)
├── src/
│   ├── ingest.py         # Phase 1 -- data-pull functions, all cached to data/raw/
│   ├── features.py       # Phase 2a -- custom scoring engine
│   ├── usage.py          # Phase 2b + 3' -- usage/efficiency features + trend signal (Family 7)
│   ├── model.py           # Phase 6 -- walk-forward validation, quantile models, CQR
│   ├── simulate.py        # Phase 6.5 -- game-environment + season simulation
│   ├── export.py          # Phase 7 -- JSON assembly
│   ├── pipeline.py        # Phase 8 -- shared fetch/score/feature orchestration
│   └── artifacts.py       # Phase 8 -- model artifact save/load
│   # radar.py / heatmap.py (Phases 4-5) not built yet -- see the phase table above
├── scripts/                # Phase 8 -- CI entry points, no notebook execution in CI
│   ├── retrain.py          # .github/workflows/retrain.yml calls this
│   └── weekly_update.py    # .github/workflows/weekly-update.yml calls this
├── models/                 # Phase 8 -- gitignored except the one committed artifact
│   └── fanteasy_model.joblib
├── data/
│   ├── raw/              # Downloaded play-by-play, stats (gitignored)
│   ├── processed/        # Cached features (gitignored)
│   └── output/
│       └── player_advanced_stats.json   # the one file data/ commits
├── tests/
│   ├── test_no_leakage.py
│   ├── test_model.py
│   ├── test_simulate.py
│   ├── test_trend.py
│   ├── test_export.py
│   ├── test_pipeline.py    # Phase 8
│   └── test_artifacts.py   # Phase 8
├── .github/workflows/
│   ├── retrain.yml         # workflow_dispatch only -- train + walk-forward validate
│   └── weekly-update.yml   # Tuesdays in-season + workflow_dispatch -- inference only
├── requirements.txt
├── .venv/                # local environment, gitignored
├── CLAUDE.md / PROJECT_CONTEXT.md / NOTEBOOK_OUTLINE.md / PHASE_2B_6_SPEC.md
└── README.md
```

**Local environment:** Python 3.12.9 in `.venv`, VS Code with the Microsoft Python
and Jupyter extensions. Deliberately not 3.14 — SHAP and MLflow lag new Python
releases, and local should match CI. Both Phase 8 workflows pin `3.12`, matching
local — the version mismatch noted in earlier drafts of this doc is resolved.

Notebooks for exploration → modules for reusable code → CI for automation. Production code lives in `src/`; notebooks just orchestrate and visualize.

---

## Phase 1 — Data ingestion

### Sources

In practice you call the wrappers in `src/ingest.py`, which add parquet caching and
the Polars → pandas conversion. The underlying `nflreadpy` calls are:

```python
import nflreadpy as nfl   # returns Polars DataFrames

# Per-play data: ~50k rows/season, the master truth source
pbp = nfl.load_pbp([2024, 2025])

# Weekly aggregated stats (easier than aggregating pbp yourself)
weekly = nfl.load_player_stats([2025], summary_level="week")

# Snap counts (critical for snap share / usage trend)
snaps = nfl.load_snap_counts([2025])

# NextGenStats (aDOT, separation, target separation)
# NOTE: argument order is (seasons, stat_type) — the reverse of the old
# nfl_data_py import_ngs_data(). get_ngs_data() in src/ingest.py keeps the
# OLD order so notebooks didn't need editing.
ngs_passing   = nfl.load_nextgen_stats([2025], stat_type="passing")
ngs_receiving = nfl.load_nextgen_stats([2025], stat_type="receiving")
ngs_rushing   = nfl.load_nextgen_stats([2025], stat_type="rushing")

# Schedule for matchup features
schedule = nfl.load_schedules([2025])

# Roster for player_id lookups
rosters = nfl.load_rosters([2025])
```

Two gotchas worth internalizing:

- **Everything comes back Polars.** `src/ingest.py` calls `.to_pandas()` before
  returning. If you call `nflreadpy` directly, remember the API is different
  (`.filter()`, not `[mask]`).
- **Play-by-play is huge in pandas.** Two seasons is ~380 columns and can top a
  gigabyte after conversion. `get_pbp()` takes an optional `columns=` list that
  subsets in Polars *before* converting — use it once you know which columns
  Phase 2 needs.

Plus from Sleeper:
- `/projections/nfl/regular/{year}/{week}` — to compare YOUR model against
- `/players/nfl` — for player_id → Sleeper ID mapping

### Player ID mapping is the hardest part

nflverse uses `gsis_id`. Sleeper uses its own internal `player_id`. You'll need a crosswalk. Two options:
1. `nflreadpy`'s `load_ff_playerids()` gives you a mapping table with multiple ID systems
2. Build your own by matching `(name, position, team)` and hand-resolving collisions

Store the crosswalk once, reload from cache thereafter.

```python
crosswalk = get_id_crosswalk()   # src/ingest.py — cached to data/raw/
slim = crosswalk[['gsis_id', 'sleeper_id', 'name', 'position', 'team']] \
         .dropna(subset=['sleeper_id'])
slim.to_csv('data/processed/id_crosswalk_slim.csv', index=False)
```

### The dtype trap — read this before debugging any empty merge

`sleeper_id` arrives as **float64**, because nulls in the column force the upcast.
That silently turns Sleeper's `"4984"` into `4984.0`. Sleeper's own player IDs are
**strings**, so `crosswalk['sleeper_id'] == '4984'` matches nothing and every join
returns zero rows without raising.

`get_id_crosswalk()` calls `_normalize_id_column()` to strip the trailing `.0` and
return clean strings. **Don't bypass it.** If a Phase 2+ merge comes back empty, this
is the first thing to check — not your merge keys, not your filters.

The dashboard keys everything by Sleeper ID. Don't forget to convert at export time.

---

## Phase 2 — Feature engineering

Aggregate per player per week, then aggregate to per-season.

### QB features

| Feature | Computation |
|---|---|
| `pass_attempts` | sum of `pass_attempt == 1` |
| `completions` | sum of `complete_pass == 1` |
| `comp_pct` | completions / attempts |
| `pass_yards` | sum of `passing_yards` |
| `pass_tds` | sum of `pass_touchdown == 1` |
| `interceptions` | sum of `interception == 1` |
| `air_yards` | sum of `air_yards` for pass attempts |
| `adot` | air_yards / pass_attempts |
| `big_play_completions` | pass completions with `passing_yards >= 20` |
| `big_play_rate` | big_play_completions / completions |
| `sacks` | sum of `sack == 1` |
| `dropbacks` | pass_attempts + sacks |
| `sack_rate` | sacks / dropbacks |
| `rush_attempts` | sum of `rush_attempt == 1 and rusher_player_id == qb_id` |
| `rush_yards` | sum of `rushing_yards` filtered by QB |
| `rush_tds` | sum of `rush_touchdown` filtered by QB |
| `rushing_threat_score` | (rush_yards + rush_tds*6) per game |

### RB features

| Feature | Computation |
|---|---|
| `carries` | sum of `rush_attempt == 1` |
| `rush_yards` | sum of `rushing_yards` |
| `ypc` | rush_yards / carries |
| `rush_tds` | sum of `rush_touchdown == 1` |
| `explosive_runs` | carries with `rushing_yards >= 10` |
| `explosive_rate` | explosive_runs / carries |
| `targets` | sum of `pass_attempt == 1 and receiver_player_id == rb_id` |
| `receptions` | sum of `complete_pass == 1 and receiver == rb_id` |
| `rec_yards` | sum of `receiving_yards` |
| `rec_tds` | sum of `pass_touchdown` filtered to receiver |
| `goal_line_carries` | carries where `yardline_100 <= 5` |
| `goal_line_share` | goal_line_carries / team_goal_line_carries |
| `snaps` | from `snap_counts.offense_snaps` |
| `snap_share` | snaps / team_offensive_snaps |

### WR / TE features

| Feature | Computation |
|---|---|
| `targets` | sum of `pass_attempt == 1 and receiver == wr_id` |
| `target_share` | targets / team_targets |
| `receptions` | sum of `complete_pass == 1 and receiver == wr_id` |
| `catch_rate` | receptions / targets |
| `rec_yards` | sum of `receiving_yards` |
| `adot` | mean of `air_yards` on targets |
| `yac` | mean of `yards_after_catch` on completions |
| `red_zone_targets` | targets where `yardline_100 <= 20` |
| `rz_target_share` | red_zone_targets / team_rz_targets |
| `routes_run` | from NGS receiving (if available) |
| `yprr` | rec_yards / routes_run |
| `snaps`, `snap_share` | from snap_counts |

### K / DEF features

Mostly derivable from weekly stat aggregation directly — no pbp needed.

---

## Phase 3' — Usage trends

**Replaces the original Phase 3 (role classification) below this point —
role labels were dropped, not deferred.** A role label ("Pocket Passer",
"3-Down Back", "Slot") is a category: it forces every player into one
bucket of a fixed set, using thresholds picked by eye against a histogram,
and it says nothing about whether that role is CHANGING right now — which
is the thing a manager actually needs to know week to week. Half the
players in any real distribution sit near a bucket boundary and get
half-fit into whichever side the threshold happens to land on. A trend
signal is continuous and honestly uncertain instead: how much a player's
recent usage is running above or below their own season baseline, with no
bucket to force them into.

Built in `src/usage.py`'s Family 7 (`add_trend_features`,
`get_usage_trend_leaders`) and validated in
`notebooks/04_usage_trends.ipynb`. For snap share (`offense_pct`),
`target_share`, `carry_share`, and a new combined `rz_opportunity_share`
(red-zone targets + carries over the team's red-zone plays — Family 4's
`rz_target_share`/`rz_carry_share` have different denominators and can't be
summed into one share directly):

- **Trend signal** = `(<feat>_ewm3 − <feat>_s2d) / <feat>_vol` — the recent
  (3-week half-life) value minus the season-to-date baseline, divided by
  the player's own season-to-date volatility so the number is comparable
  across a bell-cow and a committee back. The 3-week window is the
  existing `EWM_HALFLIFE` from Family 6, reused rather than re-derived —
  validated empirically (not assumed) by checking whether a usage rise
  measured over 3 weeks predicts the following week's usage holding above
  baseline, against 4- and 5-week windows too; 3 won on every feature
  tested, monotonically. See `PROJECT_CONTEXT.md`'s **Phase 3' findings**
  for the numbers.
- **Direction label** — `rising` / `falling` / `stable`, from the signal
  crossing a data-picked threshold (not sign alone — see the findings for
  why).
- **Riser/faller lists** — top N by trend signal, per position, for a
  given week, gated by a minimum-games-played floor so a two-game sample
  can't top the list on noise.

---

## Phase 4 — Radar metric normalization

The dashboard expects each radar metric to be **0-100 scaled relative to the player's position group**.

```python
def normalize_radar(player_value, position_distribution, lower_is_better=False):
    """
    Convert a raw stat to a 0-100 percentile score within the position group.
    """
    if lower_is_better:
        # e.g. sack rate: lower is better
        return 100 * (1 - (position_distribution < player_value).mean())
    else:
        return 100 * (position_distribution < player_value).mean()
```

Use percentile rank rather than min-max scaling — it's more robust to outliers (one QB with 50 TDs shouldn't compress everyone else).

For each position's radar axes (defined below), compute the percentile for every player in that position group and store the 0-100 score.

**QB radar axes**: Pass Volume, Accuracy, Big Play Rate, Rushing Threat, TD Rate, Sack % (lower better)
**RB radar axes**: Rush Volume, YPC, Receiving Work, Explosive Runs, Goal Line Share, Snap Share
**WR radar axes**: Target Volume, Target Share, aDOT, YAC, Red Zone Share, Contested Catch % (or proxy)
**TE radar axes**: Route Participation, Target Share, YPRR, Red Zone Usage, Blocking Snaps (lower = more pass-catching), YAC
**K radar axes**: FG Attempts, Long FG %, XP %, Touchback %, Game Script, Dome %
**DEF radar axes**: Pressure Rate, Turnover Rate, YPG Allowed (lower better), Red Zone Stops, Sack Rate, ST TDs

---

## Phase 5 — Heatmap zones

Three heatmap types depending on position. Output as a 2D array of frequencies normalized to sum to 1.

### QB passing heatmap (3×3 grid)

```python
def build_qb_heatmap(qb_pbp):
    # Field zones: rows = depth, cols = horizontal
    # depth: deep (>20 air yards), mid (10-20), short (<10)
    # horizontal: left, middle, right
    qb_pbp['depth_bucket'] = pd.cut(qb_pbp['air_yards'], [-100, 10, 20, 100], labels=['short', 'mid', 'deep'])
    qb_pbp['horiz_bucket'] = qb_pbp['pass_location']  # 'left', 'middle', 'right'

    counts = qb_pbp.groupby(['depth_bucket', 'horiz_bucket']).size().unstack(fill_value=0)
    # Reorder: rows top-to-bottom = deep, mid, short
    counts = counts.reindex(['deep', 'mid', 'short'])[['left', 'middle', 'right']]
    # Normalize to frequencies
    return (counts / counts.values.sum()).values.tolist()
```

Output: a `3×3` array of floats summing to ~1.0. Dashboard renders this as a heat-colored grid overlaid on a field SVG.

### RB rushing heatmap (3×3 grid)

```python
# horizontal: rush_gap (left, middle, right) or rush_location
# depth: typically not very informative for rushes — bucket by yards-to-go or by yardline
# Use: ('left_end', 'left_tackle', 'left_guard', 'middle', 'right_guard', 'right_tackle', 'right_end')
# Bucket to 3×3: by rush_direction (run_gap) and yardline range
```

### WR/TE target heatmap

Same as QB but from the receiver's perspective — group their own targets by depth × horizontal.

---

## Phase 6 — Your custom projection model

This is where you make it yours. A reasonable starting structure:

### Training data

One row per (player, week, season) with:
- Past 4 weeks rolling avg fantasy pts
- Past 4 weeks rolling avg targets/carries/attempts
- Opponent defensive ranking vs position (DvP)
- Home/away
- Vegas implied team total — **unverified whether `nflreadpy` exposes an equivalent to the old `import_win_totals`.** Check the docs when you get here; `load_schedules()` carries some betting columns (spread, total), and OddsAPI's free tier is the fallback.
- Weather (if outdoor game — `load_schedules()` carries `roof`, `surface`, `temp`, `wind`)
- Days rest
- Injury status going into the game
- Snap share trend (rolling 4-week)
- Target share trend (rolling 4-week)
- Sleeper's projection for that week (lets your model learn corrections to Sleeper's baseline)

Target: actual fantasy pts scored that week (using your league's scoring format)

### Model choice

XGBoost is a strong default for tabular regression:

```python
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

# Time-series split is critical — don't train on Week 10 to predict Week 5
tscv = TimeSeriesSplit(n_splits=5)
model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, early_stopping_rounds=20)

# Train one model per position — features differ significantly
for position in ['QB', 'RB', 'WR', 'TE']:
    pos_data = train_df[train_df['position'] == position]
    X = pos_data[feature_cols]
    y = pos_data['actual_pts']
    # Train with cv
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

### Inference for upcoming week

Build a feature row for the upcoming week (no actual_pts yet) and predict.

### Validation

Track MAE / RMSE per position. Compare against:
- Sleeper's projection MAE on the same weeks (your baseline to beat)
- A naive "use last 4-week avg" model

If you don't beat Sleeper's projection, the model isn't worth shipping yet — keep iterating on features.

---

## Phase 7 — JSON export schema

The exact shape the dashboard expects:

```json
{
  "generated_at": "2026-01-15T10:00:00Z",
  "season": 2025,
  "through_week": 16,
  "next_week": 17,
  "model_version": "v0.3",
  "players": {
    "<sleeper_player_id>": {
      "role": "Pocket Passer",
      "role_desc": "High-volume passer reliant on dropbacks.",
      "radar": {
        "Pass Volume": 78,
        "Accuracy": 82,
        "Big Play Rate": 65,
        "Rushing Threat": 25,
        "TD Rate": 71,
        "Sack %": 60
      },
      "radar_raw": {
        "Pass Volume": "38.2 att/g",
        "Accuracy": "67.8%",
        "Big Play Rate": "12.4%",
        "Rushing Threat": "8.1 pts/g",
        "TD Rate": "5.2%",
        "Sack %": "5.8%"
      },
      "heatmap": {
        "type": "qb_passing",
        "zones": [
          [0.04, 0.06, 0.05],
          [0.08, 0.12, 0.07],
          [0.14, 0.28, 0.16]
        ],
        "labels": {
          "rows": ["Deep", "Mid", "Short"],
          "cols": ["Left", "Middle", "Right"]
        }
      },
      "projections": {
        "week_17": {
          "rohan_proj": 22.4,
          "rohan_floor": 14.8,
          "rohan_ceiling": 31.2,
          "confidence": 0.74
        }
      },
      "season_advanced": {
        "snap_share": 0.94,
        "target_share": null,
        "yards_per_route_run": null,
        "adot": 8.4,
        "yac_per_rec": null
      }
    }
  }
}
```

### Export code

```python
import json

def export_to_dashboard(output_path):
    payload = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'season': 2025,
        'through_week': 16,
        'next_week': 17,
        'model_version': 'v0.3',
        'players': {}
    }
    for sleeper_id, player_data in computed_players.items():
        payload['players'][sleeper_id] = {
            'role': player_data['role'],
            'role_desc': player_data['role_desc'],
            'radar': player_data['radar_scores'],
            'radar_raw': player_data['radar_raw_values'],
            'heatmap': player_data['heatmap'],
            'projections': {
                f'week_{w}': {'rohan_proj': v}
                for w, v in player_data['projections'].items()
            },
            'season_advanced': player_data.get('advanced', {})
        }
    with open(output_path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))  # minified
```

Aim to keep the file under ~2 MB (1500 active players × ~1 KB each). The dashboard fetches it once on load.

---

## Phase 8 — Deployment

**Done.** The sections below described the plan before anything was built;
what actually shipped is two separate GitHub Actions workflows, not the
single notebook-execution sketch originally drafted here — see
`PROJECT_CONTEXT.md`'s **Phase 8 findings** for the full reasoning,
verification results, and disclosed limitations. Kept below for the parts
that turned out right (hosting the JSON in the same repo) and as a record
of what changed and why.

### Hosting the JSON

Two options were considered:

1. **Commit to the same repo** as `index.html`. Pros: free, no infra. Cons: bloats git history with weekly binary diffs.
2. **GitHub Pages separate branch** or **a release artifact**. Cleaner but slightly more setup.

Option 1 shipped — `data/output/player_advanced_stats.json` is committed
alongside `index.html` (see `.gitignore`'s `data/output/` exception). The
"bloats git history" concern turned out smaller in practice than expected
at this project's scale (one JSON file, weekly at most, no other binary
churn) — not revisited.

### Automation — two workflows, not one

The single "run the notebook, commit the JSON" sketch originally drafted
here doesn't fit once the projection model itself needs training: running
`jupyter nbconvert --execute notebooks/07_export_json.ipynb` on a schedule
would retrain fresh LightGBM models on every single weekly run, meaning
week-to-week output changes would come from BOTH new data AND a moving
model with no way to tell which caused a given swing. Splitting training
from inference fixes this: the model stays fixed between manual retrains,
so `weekly-update.yml`'s output changes only ever reflect new data.

- **`.github/workflows/retrain.yml`** — `workflow_dispatch` only, never
  scheduled (this is the expensive job: fetches all `HISTORICAL_SEASONS`
  from scratch, walk-forward-validates against baselines, trains final
  no-holdout models). Calls `scripts/retrain.py`. Commits
  `models/fanteasy_model.joblib` (a `.gitignore` exception, same pattern
  as the JSON) — **7.57 MB** as of the first local verification run, not a
  git-bloat concern at this workflow's manual/infrequent frequency.
- **`.github/workflows/weekly-update.yml`** — Tuesdays in-season (two cron
  entries to approximate 8am ET across the DST boundary) plus
  `workflow_dispatch`. Inference only: loads the committed artifact,
  fetches ONLY the current season, predicts the upcoming week, commits
  the regenerated JSON. Never retrains. Calls `scripts/weekly_update.py`.

Both workflows: Python 3.12 (matching local, not the `3.11` this doc
originally specified), `contents: write` permission, `pytest -q` as a
required step BEFORE the real work runs (a failing test suite aborts the
job before anything gets committed), and every fetch/validation failure
raises rather than degrading silently — a partial or stale JSON/artifact
is never committed, matching this project's "fail loudly" convention.

The one piece of real design work Phase 8 needed beyond "wire up two YAML
files": `weekly-update.yml`'s "current season only" fetch scope can't by
itself supply what `add_rolling_features` needs from the season BEFORE the
one being predicted (`prev_season_*`) or what `get_export_candidates`
needs (every player who's ever appeared). `retrain.yml` solves this by
embedding a small `history_seed` — a trimmed 2-completed-season slice of
raw feature-input columns, not the full 8-season table — in the artifact
itself. See `PROJECT_CONTEXT.md`'s **Phase 8 findings** for the full
mechanism and two disclosed limitations this accepts (candidate-universe
freshness bounded by retrain cadence; `xfp` running noisier in a season's
first few weeks under weekly-only inference).

---

## Suggested timeline

| Week | Milestone |
|---|---|
| 1 | Phase 1 (ingestion) + Phase 2 (basic features for QB/RB/WR) |
| 2 | Phase 3' (usage trend signal) + Phase 7 (export skeleton with trend only) — wire into dashboard, ship visible progress |
| 3 | Phase 4 (radar metrics) — dashboard radar charts come alive |
| 4 | Phase 5 (heatmaps) — heatmap panels light up |
| 5 | Phase 6 (baseline projection model with last-4-week avg) |
| 6 | Iterate on projection model — add DvP, Vegas totals, weather |
| 7 | Phase 8 (GitHub Actions automation) |

Wire incrementally. Don't wait until everything's perfect to push it live — the dashboard already handles missing fields gracefully (those "Awaiting model output" placeholder cards stay put until each component lands).

---

## Testing checklist before each weekly push

- [ ] JSON validates as well-formed JSON (`python -m json.tool`)
- [ ] `players` dict has at least 800 entries (sanity check — NFL has ~1700 active players)
- [ ] Every player has `trend` (Phase 3' — replaces the old `role` idea), `radar` (with all expected keys for their position), and `projections.week_N`
- [ ] No `NaN` values (replace with `null`) — JSON doesn't support NaN
- [ ] All Sleeper IDs in the file actually exist in the current `/players/nfl` response
- [ ] Spot-check 3 players you know well — does the radar profile match your intuition?
- [ ] Your projections sum to within ±15% of Sleeper's projections in aggregate (if you're way off, something's wrong with scoring format)

That last one is the most important sanity check. Your model and Sleeper's should disagree on individuals but largely agree on totals.

---

## When you get stuck

The most common pain points:
0. **Empty merges caused by ID dtypes** — see the dtype trap in Phase 1. This is not the same problem as a genuine ID mismatch, and it looks identical until you check `.dtype`. Rule out dtype first, every time.
1. **Player ID mismatches** — keep a list of "couldn't match" players, hand-resolve, save the resolved mapping
2. **Snap counts missing for recent weeks** — nflverse updates on a delay; have a fallback
3. **Bye weeks breaking rolling averages** — exclude bye weeks from the rolling window, not zero-fill
4. **Position changes mid-season** — a player might be listed RB in Week 1 and TE in Week 12 (yes, this happens). Use their *current* position for classification, but their historical position for that week's stats

Good luck — once you have even Phase 1-3 wired in, the dashboard will look dramatically more polished.
