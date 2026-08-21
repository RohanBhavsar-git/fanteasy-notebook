# FanTeasy Stats — Project Context

This document captures the "why" behind the FanTeasy Stats project so a new conversation can pick up seamlessly. Read this first before starting any new work.

> **Last updated:** August 2026. Phase 1 (ingestion) and Phase 2a (custom
> scoring) are complete and verified against live data. Phase 2b (steps 1-5,
> full feature table + notebook) is also complete and verified — see
> **Phase 2b progress** below. `SEASONS` defaults to 2018-2025 (8 seasons) in
> both notebooks, per the Phase 6 data-volume result below. Phase 6
> (projection model) was investigated, not shipped — but the earlier
> **2-season conclusion was wrong**: what looked like a signal-to-noise
> ceiling was a data-volume ceiling. At 8 seasons, Formulation A (direct
> prediction of `custom_points`) beats the two weaker baselines at every
> position and closes real ground on Sleeper's own projection without
> catching it. Formulation B (predicting the residual against Sleeper's
> projection instead) was tested too and does not improve on Formulation A.
> See **Phase 6 findings** below for the full picture, including what's
> still wrong. Phase 6's quantile models are now CQR-calibrated (see
> **Phase 6 findings**), and Phase 6.5 step 9 (`src/simulate.py` — game-
> environment sampling, matchup simulation) is done and validated against
> 204 real historical matchups — see **Phase 6.5 findings**: calibration
> looks reasonable where there's enough data to judge it, but simulated
> win probability does **not** clearly beat a naive baseline at this
> sample size, reported plainly rather than reframed. Step 10 (season
> simulation, playoff odds) is not started — see **Verification status**
> near the end before treating any pipeline claim as settled.

---

## The one-line summary

A single-file HTML dashboard for **one specific 14-team Sleeper fantasy football league** ("Fanteasy Football"), plus a Python notebook pipeline (in progress) that produces custom advanced stats and projections for the dashboard to consume.

---

## Who this is for

- **Rohan Bhavsar** — the sole author and user
- **GitHub**: RohanBhavsar-git
- **Email**: rjb16499@uga.edu
- **The League**: "Fanteasy Football" — a **redraft** league, not dynasty (Sleeper league ID: `1389706592789733376` for 2026; `1250182471429931008` for 2025)
- **League config**: 14 teams, 3rd-round-reversal snake draft, custom scoring rules

This is a personal project, not a product. The league is intentionally hard-coded — no runtime league switcher, no user accounts, no way to point the dashboard at other leagues. This is a deliberate data-governance decision to prevent unintended sharing / repurposing.

---

## Design principles

These are non-negotiable — they've shaped every decision we've made:

1. **No fake data, ever.** Every number on screen must come from a real API. Placeholder / demo / seeded data is banned. When we can't compute something honestly, we show an "Awaiting model output" card, not a plausible-looking fake.
2. **Single-file HTML.** `index.html` is the entire dashboard — vanilla JS, embedded CSS, Chart.js loaded via CDN. No build system, no bundler, no framework. This is a deliberate simplicity choice.
3. **Sleeper's public API for everything on the dashboard.** Free, no auth, generous rate limits. Custom analytics come from the separate Python notebook pipeline.
4. **SofaScore-inspired visual design.** Off-white background, cyan accent (`#0891b2`), clean typography, subtle shadows. Desktop-first (fantasy managers are almost always on desktop).
5. **Honest UI copy.** "This league" not "your league" (single-league scope). Tooltips explain what heuristics mean. Nothing pretends to be more precise than the data supports.
6. **Iterative shipping.** Real integrations before polish. If a feature can't be built on real data, we don't ship a fake version to prototype it.

---

## Tech stack

### Dashboard (already built)
- **`index.html`** — single-file, ~7000 lines
- **Vanilla JavaScript** — no framework
- **Chart.js** — CDN-loaded, used for radar and line charts
- **Hosted on GitHub Pages** at `https://rohanbhavsar-git.github.io/fanteasystats`
- **APIs used**: Sleeper public API (stats, projections, players, injuries, drafts, brackets, transactions), ESPN hidden JSON endpoints (NFL scoreboard, game summaries), Open-Meteo (weather, no auth)

### Python notebook (Phases 1 and 2a complete and verified)
- **`nflreadpy`** — the maintained Python client for nflverse data. Returns
  **Polars** DataFrames; `src/ingest.py` converts to pandas at the boundary so
  everything downstream stays pandas-native.
- **pandas** — tabular data
- **pyarrow** — required twice over: `to_parquet()` for the local cache, and
  Polars `.to_pandas()` for the conversion boundary
- **Later phases**: scikit-learn (role clustering), XGBoost/LightGBM (projections), SHAP (explainability), Optuna (hyperparameter tuning), MLflow (experiment tracking)
- **Explicitly avoiding**: LLMs, RAG, agents — these are the wrong tools for tabular fantasy regression. Following industry-relevant patterns, not hype.

> **Migration note (Aug 2026):** this project originally specified `nfl_data_py`.
> nflverse deprecated that package in 2025 in favor of `nflreadpy`, with no further
> maintenance planned. We migrated before running Phase 1, while the cost was near
> zero. Any doc, snippet, or comment still referencing `nfl_data_py` is stale.

### Local development environment
- **Python 3.12.9** (deliberately *not* 3.14 — the ML stack lags new releases, and
  local should match CI)
- **`.venv`** inside the project folder (beside `src/`, `notebooks/`, `requirements.txt`), gitignored. It was originally created one level up, which broke on a folder rename and caused `requirements.txt` to resolve from the wrong place — keep it in the project folder.
- **VS Code** with the Microsoft Python + Jupyter extensions, plus Claude Code
- The Phase 8 GitHub Actions workflow still pins `3.11` and needs bumping to `3.12`

---

## Dashboard sections and their current state

### Dashboard (home)
- League standings table with sortable columns
- Weekly matchup cards with real scores
- League-wide KPI cards (top scorer, biggest blowout, closest game, etc.)
- Activity feed (transactions, waivers, trades) — real Sleeper data
- Season Leaders panel (pending replacement per outstanding item)

### Matchups tab
- Weekly matchup cards with 🏆 winner badge, blowout/nail-biter emoji markers
- Real Sleeper starter/bench data with position-color-coded slots
- Team superlatives (🔥 streaks, ❄️ cold streaks, 🎢 boom-or-bust)
- Icon KPI cards: ⚖️ Margin, 🔥 Top Performer, 💎 Bench High
- Click any matchup → detail view with real stat lines, real Sleeper headshots
- Playoff brackets rendered from winners_bracket + losers_bracket endpoints
- Championship-only display (side games decluttered); toilet bowl reverse-labels winner/loser correctly

### Teams tab
- Per-team detail: roster, weekly scores, matchup history
- Owner handles from Sleeper user data

### Draft tab
- Fixed team columns (never shuffle per round) with snake pick numbers conveying direction
- **3rd Round Reversal support** — `draft.settings.reversal_round` respected; rounds 2 through reversal_round all reverse, then snake resumes
- Real Sleeper draft picks (`/draft/{id}/picks`)
- Position color pills, embedded player metadata, gold keeper badges
- No per-pick "traded" badges (Sleeper's data produces false positives; the standalone Pick Trades panel is the honest place for that info)
- Position distribution by round, First Off Board milestones, Position Runs panel
- Grid uses explicit cell borders (not gap-as-border trick) for reliable gridlines regardless of content overflow

### Injury Report tab
- Compact table grouped by status (IR / Out / Doubtful / Questionable / PUP / Inactive / Suspended)
- Real Sleeper injury data (status, body part, start date)
- Status-based **Expected Return** column with tone-colored pills (`4+ wks`, `This wk`, `GTD`)
- Real headshots, owner team column
- 4 KPI cards with icon tiles (Total Injured, Out/IR, On Rosters, Most Affected Position)
- **This Week's Lineup Risks side panel** (replaces earlier injury news attempt): per-team lineup risks with severity scoring, week filter dropdown, play-probability pills (~75% for Q, ~25% for D, 0% for IR/Out) with status-specific hover tooltips
- Injury news panel was tried and removed — Sleeper's `injury_notes` is often just 1-word body-part descriptors, not narrative news. The lineup-risks panel is the better use of that real estate.

### Players tab
- Real Sleeper player DB (~11k players filtered to active fantasy positions)
- Real headshots (32px) with ESPN team-logo fallback for defenses and initials fallback
- 4 KPI cards (Total, Rostered, Free Agents, Injured)
- **Week filter dropdown** in top-right
- Sortable table columns including **Sleeper Proj** and **My Proj** (placeholder until notebook wired in)
- Column headers show the selected week: "Sleeper Proj (Wk 17)"
- Filters: search, position, NFL team, owner, injury health

### Player Detail page
- Real 64px headshot hero
- 4 KPI cards in icon-tile style: Season Pts, Avg/Game (with position rank like "QB7 overall"), Sleeper Proj, Best Week
- **Position profile panel**: Currently a placeholder card ("Awaiting model output") with a bulleted list of what stats will appear per position once the notebook lands
- **Field heatmap panel**: Same placeholder pattern
- **Weekly Production chart** — real bars, 3 reference lines toggleable via legend:
  - **Season Avg** (dashed gray, on by default) — this player's own avg
  - **Sleeper Proj** (orange line, on by default) — Sleeper's projection per week
  - **Top-N Position Avg** (dashed purple, off by default) — top-N scorers at position each week, where N = league-wide starter count for the position (derived from `roster_positions` with FLEX split as 50% WR / 35% RB / 15% TE)
- Chart footnote explains the top-N math: `(2 WR + 0.5 flex share) × 14 teams ≈ 35`
- **Real Game Log**: Week, Points, Sleeper Proj, vs Proj (green if beat, red if bust), Stat Line

### Player Comparison tab
- Up to 4 players side-by-side
- **Week filter dropdown** for projections
- Selected Players shelf with 48px headshot cards + mini stats (Avg, Total, Sleeper Proj for selected week)
- Stat Comparison table with Total Points, Avg/Game, Games Played, Best Week (with Wk N), Worst Week, Sleeper Proj (Wk N), My Proj (Wk N). Best in green, worst in red per row
- Weekly Production line chart overlaying all compared players (real data)
- Profile Overlay panel: same "Awaiting model output" placeholder as player detail
- Search & Add Players panel at the bottom with same filters + headshots + projection columns

---

## Data integrations already wired

### Sleeper endpoints in use
- `/league/{id}` — league config, scoring settings, roster positions
- `/league/{id}/rosters` — team compositions
- `/league/{id}/users` — owner names/handles
- `/league/{id}/matchups/{week}` — weekly matchup starter/bench data
- `/league/{id}/transactions/{week}` — waivers/trades for activity feed
- `/league/{id}/winners_bracket` + `/losers_bracket` — playoffs and toilet bowl
- `/league/{id}/drafts` → `/draft/{id}/picks` → `/draft/{id}/traded_picks` — draft board
- `/stats/nfl/regular/{year}/{week}` — per-week player stats (lazy-loaded, cached in `state.statsByWeek`)
- `/projections/nfl/regular/{year}/{week}` — Sleeper's projections (lazy-loaded, cached in `state.projectionsByWeek`). **The notebook must use this exact host + path** (`https://api.sleeper.app/v1`, see `index.html` ~line 1924) so the model's benchmark matches the numbers the dashboard renders. Sleeper's newer `api.sleeper.com/projections/nfl/{year}/{week}?season_type=regular` form exists too; `src/ingest.py` keeps it only as a fallback.
- `/players/nfl` — player registry with injury data

### Custom scoring engine
Because the league has custom scoring, Sleeper's pre-aggregated `pts_ppr` / `pts_half_ppr` / `pts_std` fields don't reflect the actual points.

**The actual rules** (confirmed from the live `scoring_settings`, 121 keys — an earlier version of this doc guessed at these and was wrong):

| Rule | Value |
|---|---|
| Passing yards | 0.04 (1 pt / 25 yds) |
| Passing TD | **4** (not 6) |
| Interception | −2 |
| Pick-six thrown | −1 |
| Rush + receiving yards | 0.1 |
| Rush + receiving TD | 6 |
| Reception | 0.5 — flat, **no TE premium** |
| Any 2-pt conversion | 2 |
| Fumble / fumble lost | −1 each, and they **stack** (lost fumble = −2) |

**Every yardage and threshold bonus is 0.0.** No 300-yard passing bonus, no 100-yard rushing bonus, no distance-based TD bonuses. Team defense scoring is unusually detailed (tiered `pts_allow_*` and `yds_allow_*` ladders, `def_3_and_out` +0.25, `def_4_and_stop` +0.5). There's a dedicated **`computeCustomScore(statsObj, scoringSettings, position)`** helper near the top of the script that multiplies every raw stat field by its matching `scoring_settings` weight. **`resolvePts()`** wraps that with a fallback chain. Every fantasy total on the dashboard flows through `resolvePts()` — search that name to see every usage.

### ESPN
- `site.web.api.espn.com` scoreboard for NFL sidebar
- ESPN summary endpoint for gamecast box scores
- ESPN team-logo CDN for defense/team logos

### Open-Meteo
- Weather forecasts for outdoor games in gamecast view

---

## Notebook pipeline plan

See `NOTEBOOK_OUTLINE.md` for the full 8-phase roadmap. Summary:

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Data ingestion — `src/ingest.py` + `01_data_ingestion.ipynb` | **Done** — runs clean, all sources cached to `data/raw/` |
| 2a | Custom scoring — `src/features.py` + `02_custom_scoring.ipynb` | **Done** — 100% validated against Sleeper's actual results |
| 2b | Usage + efficiency features from pbp | Steps 1-5 of 10 **done** (`src/usage.py` + `03_usage_features.ipynb`) — see **Phase 2b progress** below. Steps 6-9 (Phase 6 model A/B + quantile/SHAP/CQR, Phase 6.5 game-environment simulation) are also done — see **Phase 6 findings** and **Phase 6.5 findings**. Step 10 (season simulation, playoff odds) remains. |
| 3 | Role classification — rule-based Pocket Passer / 3-Down Back / Slot / etc. | Not started |
| 4 | Radar metrics — 0-100 percentile normalization within position | Not started |
| 5 | Heatmap zones — field-location frequency tables | Not started |
| 6 | Projection model — XGBoost/LightGBM regression with time-series CV | **Investigated, not shipped.** The earlier 2-season conclusion ("loses to every baseline") was premature — it was a data-volume ceiling, not a feature-quality one. At the 8-season default, Formulation A beats `season_to_date_avg`/`trailing_3wk_avg` at every position and closes (without closing entirely) the gap to `sleeper_proj`. Formulation B (predicting the residual against Sleeper) does not improve on Formulation A. Step 8 (quantile floor/ceiling models + SHAP) is done: coverage is measured and honestly overconfident (67-75% actual vs. 80% target for the 10th-90th interval), and SHAP shows nothing that looks like a leak. See **Phase 6 findings** below. Not abandoned (real, tested code exists in `src/model.py`) and not "done" in the sense of shipping a model — the honest outcome is still deciding not to ship one yet. |
| 6.5 | Monte Carlo simulation — win probability, playoff odds, floor/ceiling | Step 9 **done** (`src/simulate.py` — game-environment sampling, matchup simulation) — see **Phase 6.5 findings**. Validated against 204 real historical matchups: calibration is reasonable where there's enough data to judge it; win-prediction accuracy does not clearly beat a naive baseline at this sample size. Step 10 (season simulation, playoff odds) not started, per explicit scope stop. |
| 7 | JSON export — assemble `player_advanced_stats.json` | Not started |
| 8 | GitHub Actions weekly automation | Not started |

Notebooks are kept single-purpose: `01_data_ingestion.ipynb` does ingestion only,
`02_custom_scoring.ipynb` does scoring only. Exploration and debugging belong in a
separate scratch notebook — mixing them in meant 01 stopped running top to bottom.

The dashboard already has integration hooks waiting. Search `state.advancedStats` and `p.myProj` and `// Hook for your custom model` in `index.html` to see exactly where each output will slot in.

---

## Phase 2b progress (steps 1-4 of 10, complete)

`src/usage.py` runs six idempotent functions in sequence over `weekly_scored.parquet` (QB/RB/WR/TE, REG only), each dropping and recomputing its own output columns:

- `add_volume_features(df, pbp)` — target/air-yards/carry share, WOPR, touches, QB dropback/scramble split
- `add_snap_features(df, snaps, crosswalk)` — `offense_snaps`/`offense_pct` via the `pfr_player_id` crosswalk
- `add_situational_features(df, pbp)` — red-zone/goal-line volume, third-down and two-minute target share
- `add_context_features(df, schedule)` — `is_home`, spread, implied team total, weather
- `add_xfp_features(df, pbp, scoring_settings)` — expected fantasy points from a bucketed opportunity-value rate table
- `add_rolling_features(df)` — `_ewm3`/`_s2d`/`_vol` trailing summaries of every continuous feature above, plus `games_played`, `snap_share_delta_3wk`, and `prev_season_*` baselines

**Leakage-test approach.** `tests/test_no_leakage.py` (34 tests) pairs two patterns per family:
- **Black-box future-truncation** — build the table twice, once with weeks after a boundary removed from the source data, and assert weeks at/before the boundary are unchanged. Catches a feature looking forward.
- **White-box perturbation tests** — for xFP's rate table and the rolling aggregates, future-truncation alone can't prove a feature excludes its OWN week, because removing week N's row also destroys real same-week information the feature legitimately needs (a player's actual opportunities that week; week N's own value that week N+1 needs to see). Instead these tests perturb one real value to an extreme outlier and confirm the SAME week's derived feature is unaffected while the NEXT week's is. Step 4 (rolling aggregates) got the most coverage of any step for exactly this reason — it's where a shift(1) mistake would do the most damage.

**Definitions settled that aren't obvious from the spec alone:**
- `target_share`/`carry_share`/`touch_share` denominators are team **targets** (pass attempts with a recorded receiver), not "team pass attempts" as the spec literally says — that reading doesn't sum to 1.0 per team-week, contradicting the phase's own acceptance criterion.
- This pbp snapshot's `pass_attempt` flag fires on sack rows too — doesn't match the official "attempts" stat alone; verified against a real box score before trusting any pass-attempt-shaped denominator.
- QB kneels are excluded from `designed_rush_attempts` and from xFP's carry population — they set `rush_attempt=1` in this pbp version but aren't a called play in any fantasy-relevant sense.
- `_s2d` (season-to-date expanding mean) and `_vol` (expanding standard deviation) are two separate columns — an earlier pass conflated them under `_std`, describing it in the spec as "expanding mean," which turned out to be a naming error, not a units typo.
- `prev_season_*` holds last season's full-season average for a core subset of role/opportunity features and does **not** reset at the season boundary the way in-season rolling features do — an entire prior season can't leak forward.

Step 5 (`03_usage_features.ipynb`, exercising the full table) is also done. Phase 6 (projection model) was picked up next, investigated, and concluded rather than shipped — see **Phase 6 findings** below. Steps 8-10 (quantile floor/ceiling, `src/simulate.py`, calibration/playoff-odds) remain open, contingent on what happens with Phase 6 next.

---

## Phase 6 findings

`src/model.py` trains one untuned LightGBM model per position (QB/RB/WR/TE), expanding-window walk-forward validation (no shuffling, no KFold). Two things changed the conclusion since the first pass: extending `SEASONS` from 2 to 8 years, and testing a second target formulation. Both are documented here in full, including where the model still falls short.

### The 2-season conclusion was premature

The first version of this section said "Model A loses to every baseline at every position" and diagnosed it as a signal-to-noise ceiling — a feature-quality problem. That was wrong. It was a **data-volume ceiling**: at 2 seasons of training history, the model had already caught the *weakest* of the four baselines (`trailing_3wk_avg`, beaten at 3 of 4 positions) while losing to the stronger three. Stating that as "loses to every baseline" overstated the case in the pessimistic direction.

The real test was whether the gap to the *strong* baselines closed as training history grew — both improving together would prove nothing. It did close:

| Position | Volume | Model MAE | season_to_date_avg | trailing_3wk_avg | sleeper_proj | trailing_xfp |
|---|---|---|---|---|---|---|
| QB | 2 seasons | 6.82 | 6.61 | 6.79 | **5.54** | N/A |
| QB | 4 seasons | 6.49 | 6.61 | 6.79 | **5.54** | N/A |
| QB | 8 seasons | **6.38** | 6.61 | 6.79 | **5.54** | N/A |
| RB | 2 seasons | 4.42 | 4.25 | 4.49 | **3.87** | 4.24 |
| RB | 4 seasons | 4.20 | 4.25 | 4.49 | **3.87** | 4.20 |
| RB | 8 seasons | **4.18** | 4.25 | 4.49 | **3.87** | 4.21 |
| WR | 2 seasons | 4.08 | 3.99 | 4.22 | **3.77** | 4.02 |
| WR | 4 seasons | 3.97 | 3.99 | 4.22 | **3.77** | 4.00 |
| WR | 8 seasons | **3.93** | 3.99 | 4.22 | **3.77** | 4.00 |
| TE | 2 seasons | 3.23 | 3.12 | 3.32 | **2.88** | 3.03 |
| TE | 4 seasons | 3.09 | 3.12 | 3.32 | **2.88** | 3.02 |
| TE | 8 seasons | **3.06** | 3.12 | 3.32 | **2.88** | 3.02 |

(`sleeper_proj` and `trailing_xfp` baselines don't move with training-data volume — they're independent of the model — so their MAE is constant across rows; shown for reference. Baselines were computed once, held fixed, and the model was retrained on 2/4/8 seasons of history against the identical locked evaluation folds — 2024 Wk5 through 2025 Wk18 — so the only thing changing between rows is training-data volume.)

**At 8 seasons, the model beats `season_to_date_avg` and `trailing_3wk_avg` at every position.** It also beats `trailing_xfp` at RB and WR (not just "closes ground" — it's ahead), and comes within 0.04 MAE of it at TE. Against `sleeper_proj` it closes real ground without catching it: the QB gap shrinks from 1.28 MAE at 2 seasons to 0.84 at 8; RB from 0.55 to 0.31; WR from 0.32 to 0.16; TE from 0.35 to 0.17 — roughly halved at every position, but not zero anywhere. Sleeper's projection still wins clearly everywhere — it encodes beat-writer/injury/depth-chart information nflverse-derived features structurally can't see.

**Gains diminish from 4→8 seasons.** The 2→4 season jump closes 2-3x more of the gap to `season_to_date_avg` than the 4→8 jump does, at every position (e.g. WR: 2→4 closes 0.11 MAE, 4→8 closes only 0.05 more). More history keeps helping, but with clearly decreasing returns — consistent with a model that's approaching what this feature set can extract, not one that's data-starved indefinitely.

### Feature-count reduction is not the fix — and the first attempt at this diagnosis was itself wrong

An initial ablation (rank all ~180 features by SHAP importance from one model trained on the *full* dataset, then test top-10/25/50/all) showed a clean "peaks at 25, degrades at scale" pattern at every position — textbook overfitting, or so it looked. Redone properly, deriving the SHAP ranking *inside each walk-forward fold, from that fold's own training data only*: the pattern **disappeared**. 25 features doesn't peak anywhere — it's the single worst option at QB, and RB/WR/TE all prefer the full feature set. The first result was an artifact: ranking features on the full dataset lets the ranking "see" the same weeks later used to score the ablation, which flatters small feature sets in a way that doesn't hold up once the ranking itself is confined to what a fold could actually have known. **This is a distinct failure mode from the training-data leakage `tests/test_no_leakage.py` guards against** — that suite verifies no *feature value* depends on future weeks; it says nothing about whether the *choice of which features to use* was made with future information. Worth remembering for any future feature-selection work in this project, not just this one model.

An earlier residual diagnostic (predict `custom_points − season_to_date_avg` at 2-season volume) found a real but weak signal — predicted and actual residuals correlated at r≈0.11-0.15 — that didn't survive reconstruction (worse MAE than the raw baseline). That result is superseded by the properly-scoped test below, run at 8 seasons against the strongest baseline instead of the weakest.

### Formulation B: predicting the residual against Sleeper

Formulation A predicts `custom_points` directly. Formulation B instead trains on `custom_points − sleeper_projection` and reconstructs `pred = predicted_residual + sleeper_projection` at inference time — the question is whether the feature set can improve on Sleeper's own number rather than just beat trailing averages. Same 8-season volume, same walk-forward setup, same locked evaluation folds (2024 Wk5-2025 Wk18), no tuning.

| Position | Formulation B MAE | season_to_date_avg | trailing_3wk_avg | sleeper_proj | trailing_xfp |
|---|---|---|---|---|---|
| QB | 6.10 | 6.61 | 6.79 | **5.54** | N/A |
| RB | 4.22 | 4.25 | 4.49 | **3.87** | 4.21 |
| WR | 4.00 | 3.99 | 4.22 | **3.77** | 4.00 |
| TE | 3.21 | 3.12 | 3.32 | **2.88** | 3.02 |

Formulation B beats `trailing_3wk_avg` at every position and `season_to_date_avg` at QB and RB, but **loses to `season_to_date_avg` at WR and TE**, and to `trailing_xfp` at RB and TE — a worse record than Formulation A's, which beat both weaker baselines everywhere and even overtook `trailing_xfp` at RB/WR. Like Formulation A, it never catches `sleeper_proj`.

The diagnostic explains why. The predicted residual correlates with the actual residual at only r≈0.03-0.05 across all four positions — essentially no case-by-case signal — and its standard deviation (1.7-3.3) is far tighter than the true residual's (4.3-7.4). In practice the model has learned a roughly constant upward shift (mean predicted residual 0.57-0.81, close to the true mean residual of 0.46-0.64) rather than anything player- or week-specific: Sleeper's projections undershoot actual scoring by about half a point on average in this sample, and the model mostly just reproduces that constant correction. That's enough to edge out the weakest baseline everywhere and the second-weakest sometimes, but it isn't forecasting signal, and it's why Formulation A — which at least has the full `custom_points` scale to work with — outperforms it head to head.

**Conclusion: Formulation A is the better of the two approaches tested, and neither ships yet.** Direct prediction of `custom_points` at 8 seasons has closed real ground on every baseline except Sleeper's own projection, without catching it. Predicting the residual against Sleeper doesn't help — the residual itself carries almost no learnable signal beyond a near-constant bias correction, and reconstructing from it performs worse than direct prediction against everything except the single weakest baseline. The Phase 2b feature table isn't wasted work either way — it powers the analytical panels (usage trends, xFP/luck, role changes) that Sleeper doesn't show, which was always half the point (see **What this is actually for** in `PHASE_2B_6_SPEC.md`). The dashboard keeps showing Sleeper's own projections, labeled as Sleeper's, rather than a model that still can't beat them. More data, a materially different feature strategy, or blending with Sleeper's number are future decisions, not something concluded here.

### Step 8: quantile models (floor/ceiling) and SHAP

Five LightGBM quantile models per position (`objective='quantile'`, alpha ∈ {0.10, 0.25, 0.50, 0.75, 0.90}), Formulation A's feature set, same 8-season walk-forward setup and locked evaluation folds (2024 Wk5-2025 Wk18) as everything above, no tuning.

**Quantile crossing happens, and is fixed by rearrangement.** LightGBM fits each quantile as an independent model, so nothing enforces `pred_q10 <= pred_q25 <= ... <= pred_q90` on a given row. It crossed on 3.2-9.8% of rows depending on position (QB 8.6%, RB 5.2%, WR 3.2%, TE 9.8%) — common enough to require handling, not an edge case. Fixed with the standard rearrangement approach (Chernozhukov, Fernandez-Val & Galichon 2010): sort each row's five predicted values into non-decreasing order and reassign them back to the same columns. This changes nothing about any single quantile's marginal calibration — same predicted values, same column-by-column accuracy — it only removes the crossing. `fix_quantile_crossing()` / `quantile_crossing_rate()` in `src/model.py`.

**Coverage — the check that matters.** Fraction of actual outcomes at or below each predicted quantile, target vs. actual:

| Position | q10 | q25 | q50 | q75 | q90 |
|---|---|---|---|---|---|
| Target | 10% | 25% | 50% | 75% | 90% |
| QB | 16.8% | 31.5% | 52.1% | 71.4% | 84.7% |
| RB | 13.2% | 28.7% | 51.5% | 72.5% | 86.9% |
| WR | 12.9% | 28.4% | 50.7% | 73.5% | 87.9% |
| TE | 19.1% | 32.9% | 51.8% | 72.2% | 85.8% |

The median (q50) is well-calibrated everywhere — 50.7-52.1% against a 50% target. Every other quantile shows the same systematic pattern at every position: the low quantiles (q10, q25) run **high** (more actuals fall below them than they should) and the high quantiles (q75, q90) run **low** (more actuals exceed them than they should). Put together, the predicted spread is **too narrow** — these are overconfident intervals, not just noisy ones.

The 10th-90th interval, meant to cover ~80% of outcomes, in practice:

| Position | Below q10 | Within | Above q90 |
|---|---|---|---|
| Target | 10% | **80%** | 10% |
| QB | 16.8% | **67.9%** | 15.3% |
| RB | 13.2% | **73.7%** | 13.1% |
| WR | 12.9% | **75.0%** | 12.1% |
| TE | 19.1% | **66.7%** | 14.2% |

None of these hit the "exceeded 30% of the time" failure mode that would make the interval worse than useless, but all four are meaningfully overconfident (67-75% actual coverage against an 80% target) — both tails get breached 1.2-1.9x more often than the interval claims. **Treat the floor/ceiling as directional, not literal probabilities**, until this is corrected (widening the interval, or a proper conformal calibration pass, are the standard fixes — not attempted here per "no tuning").

**TE's zero-inflated target is a genuine data characteristic, not a bug, and it explains TE's worse-than-others floor.** 19.4% of TE player-weeks score exactly 0.0 `custom_points` (rostered but inactive, a healthy scratch, or a snap count too low to record any counting stat) — almost exactly TE's 19.1% q10 coverage figure above. With a fifth of the true distribution sitting at one single point, no quantile near that point can cleanly carve out only 10% of the mass; the model's q10 prediction often lands at or near 0 (correctly — that really is close to the 10th percentile of TE output), but ties between a near-zero prediction and an exactly-zero actual inflate the measured "below" rate well past what a smooth continuous distribution would produce at the same quantile. This is a real property of backup/committee usage at the position, not a modeling failure to fix.

*(Caught and fixed while building this: an earlier version of `quantile_interval_coverage()` used strict `<` for its "below" comparison while `quantile_coverage()` used `<=` — mathematically the same comparison, but with 19% of TE's target at exactly one value, the tie-breaking convention alone moved the reported TE "below q10" figure from 6.5% to 19.1% and "within" from 79.4% to 66.7%, reversing which position looked best-calibrated. Both functions now use the same `<=` convention; `tests/test_model.py::test_quantile_interval_coverage_matches_quantile_coverage_at_the_boundary` locks the two in agreement going forward.)*

**SHAP on the median (q50) model, top 20 features per position — nothing looks like it shouldn't matter.** All four positions' top 20 are dominated by exactly the categories the feature table was built to surface: rolling volume/role share (`target_share_ewm3`, `touch_share_ewm3`, `offense_snaps_ewm3`, `offense_pct_ewm3`), efficiency (`epa_per_dropback_ewm3`, `yards_per_carry_ewm3`, `xfp_ewm3`), pregame game context (`team_implied_total`, `game_total`, `spread`, `temp`, `wind`, `roof` — all genuinely knowable before kickoff), and prior-season carryover (`prev_season_*`). No current-week outcome stat, no ID-like column, and nothing structurally leaky made it into any position's top 20 — consistent with `FEATURE_COLUMNS` being restricted to Family 5/6 by construction (see the module docstring in `src/model.py`).

Two things worth naming, neither a leak: (1) `fp_over_expected` (the touchdown-luck-stripped efficiency signal) appears 3-4 times per position among its own `_ewm3`/`_s2d`/`_vol`/`prev_season_` variants (RB: 4 of 20, WR: 3 of 20) — correlated representations of one underlying quantity splitting SHAP credit between them, so the "20 features" list overstates how many independent signals are actually driving the model. (2) TE's SHAP list includes `avg_cushion_vol` (volatility of the pre-snap coverage cushion, an NGS field) — a legitimate coverage/role signal (man vs. zone, in-line vs. slot usage), just a step further from raw volume than everything else in the list; flagged here for visibility, not because it looks wrong.

`walk_forward_predict_quantile()`, `quantile_crossing_rate()`, `fix_quantile_crossing()`, `quantile_coverage()`, `quantile_interval_coverage()`, and `shap_top_features_median_model()` are all in `src/model.py`. These quantile predictions are exactly what Phase 6.5's simulation is specified to sample from (see `PHASE_2B_6_SPEC.md`'s Phase 6.5 section) — Phase 6.5 flagged the undercoverage above as a prerequisite blocker for its own step 9; the following subsection addresses it.

### Calibrating the quantile intervals: conformalized quantile regression (CQR)

Standard CQR (Romano, Patterson & Candès 2019), applied separately per position and per interval pair (10th-90th, 25th-75th), on the same quantile models from step 8 — no retraining, no tuning, purely a post-hoc statistical correction:

1. **Conformity score** per calibration row: `E = max(pred_lower - actual, actual - pred_upper)` — positive if the actual outcome fell outside the predicted interval on either side, a negative margin if it fell inside.
2. **Widening amount**: the finite-sample-corrected empirical quantile of those scores — the `ceil((n+1) × target_coverage)`-th order statistic, which is what gives CQR its coverage guarantee at finite n (a naive `np.quantile` under-widens slightly).
3. **Apply**: subtract that amount from every predicted lower bound, add it to every predicted upper bound, in the evaluation set. One constant per position per interval pair — not scaled by the prediction's own magnitude.

**The calibration split, stated plainly:** calibration uses every walk-forward fold from 2018 Wk5 through 2024 Wk4 — the entire history *before* the locked evaluation window — and evaluation is the same locked folds as everything else in this document (2024 Wk5-2025 Wk18). These are generated in a single pass of `walk_forward_predict_quantile()` with `eval_min_season=None`, so the calibration predictions are already genuinely out-of-sample (each fold's model trained on strictly earlier weeks only) and the split is chronologically disjoint by construction — calibration is never computed on the fold being evaluated or on anything later. This is one fixed calibration reservoir (not re-expanded fold-by-fold through the evaluation window); simpler to reason about, and with 5.5-6 years of history behind it, more than large enough (3,774-14,158 calibration rows depending on position) for a stable widening constant.

**Coverage, before and after:**

| Position | Interval | Target | Before | After | Width before | Width after | Width ×  |
|---|---|---|---|---|---|---|---|
| QB | 10-90 | 80% | 67.9% | **82.6%** | 16.36 | 20.98 | 1.28x |
| QB | 25-75 | 50% | 39.9% | **53.2%** | 8.56 | 11.09 | 1.30x |
| RB | 10-90 | 80% | 73.7% | **84.5%** | 11.31 | 12.77 | 1.13x |
| RB | 25-75 | 50% | 43.9% | **56.3%** | 5.76 | 6.69 | 1.16x |
| WR | 10-90 | 80% | 75.0% | **86.0%** | 11.17 | 12.38 | 1.11x |
| WR | 25-75 | 50% | 45.2% | **55.8%** | 5.57 | 6.33 | 1.14x |
| TE | 10-90 | 80% | 66.7% | **84.7%** | 7.97 | 8.91 | 1.12x |
| TE | 25-75 | 50% | 39.3% | **56.7%** | 3.98 | 4.69 | 1.18x |

Median (q50) coverage is untouched by CQR (there's no interval around a single point to widen) and stays at 50.7-52.1% across all four positions — already well-calibrated, as found in step 8. Every interval moved from clearly undercovered to within a few points of nominal, at the cost of 11-30% wider bounds depending on position and interval — QB needed the most correction (widest before, most correction after), the skill positions needed the least. This is the expected trade the request called out, and it's a modest one: nowhere does fixing coverage require doubling the interval.

**The correction overshoots slightly, and does so asymmetrically — worth knowing before sampling from it.** Every corrected interval landed a little *above* its target (82.6-86.0% against 80%; 53.2-56.7% against 50%), and the two tails don't share the overshoot evenly: the lower bound consistently ends up more conservative than the upper bound (e.g. QB 10-90 after: below=7.5% against a 10% target, above=9.9% — almost exactly on target). This isn't a bug — a single symmetric widening constant responds to whichever tail was worse in the calibration data, and at every position the *floor* was the worse-calibrated side before correction (see step 8's finding that low quantiles ran high everywhere). Fixing the floor by the amount needed pulls the ceiling along with it by the same constant, which overshoots the ceiling's smaller original problem. An asymmetric CQR variant (a separate widening constant per side) would target each tail independently; not implemented here since the request specified widening the interval by one empirical quantile of the combined score, and the aggregate coverage achieved is already close to nominal.

**Checked whether undercoverage is uniform across the prediction range, since that changes whether one global constant is enough — it's mostly uniform, except at TE.** Bucketing each position's calibration set into predicted-magnitude terciles and looking at the *raw*, pre-CQR breach rate:

| Position | Low tercile within | Mid tercile within | High tercile within |
|---|---|---|---|
| QB | 65.1% | 64.5% | 65.5% |
| RB | 68.4% | 68.3% | 68.7% |
| WR | 69.1% | 70.6% | 68.2% |
| TE | **60.3%** | 67.9% | 67.4% |

QB, RB, and WR are flat within 1-2 points across the whole predicted range — a single global constant is a reasonable fit for all three. **TE is not**: its bottom tercile (the lowest-predicted, most backup/committee-usage third of TE player-weeks — exactly where the 19.4% zero-inflation concentrates) is 6-8 points worse-covered than the other two-thirds of the range. A single global TE correction, calibrated on the whole range, therefore under-corrects precisely the low-usage TEs it matters most for and slightly over-corrects the rest — a real, disclosed limitation, not something the current fix resolves.

One direction-of-miscoverage pattern held at all four positions, independent of the terciles-are-flat-or-not finding above: within every tercile, low-predicted rows breach low (actual busts below the floor) more than they breach high, and high-predicted rows breach high (actual booms above the ceiling) more than they breach low — the floor is the bigger problem for backup-caliber weeks, the ceiling is the bigger problem for every-week starters. The single symmetric constant used here doesn't correct for this directional shift with predicted magnitude, only for the aggregate rate; that's the same root cause as the overshoot asymmetry above.

**Tie-breaking:** the coverage checks reported before and after reuse `quantile_coverage()`/`quantile_interval_coverage()` unchanged from step 8's `<=` fix, so the same TE zero-inflation tie-breaking care applies identically on both sides of this comparison — no new inconsistency was introduced computing "after" numbers. The CQR conformity score itself has no boolean comparison to get wrong (it's a continuous `max()` of two differences), so ties at the boundary (a near-zero prediction against an exactly-zero actual, common for TE) just produce a legitimate `E ≈ 0` conformity score rather than an ambiguity — nothing to fix there, just worth naming since it's the same population of rows.

**Bottom line for Phase 6.5 step 9:** the prerequisite `PHASE_2B_6_SPEC.md` flagged is addressed — aggregate interval coverage is now within a few points of nominal at every position, using a leak-free, time-respecting calibration split, with no tuning of the underlying models. The residual asymmetry (floor slightly over-conservative relative to ceiling) and the TE low-usage-tercile gap are disclosed limitations to carry into the simulator, not blockers: sampling from these calibrated intervals will be slightly more conservative than necessary on the floor side for every position, and specifically still a bit optimistic for backup/committee TEs. `conformity_scores()`, `conformal_quantile()`, `apply_conformal_widening()`, and `interval_breach_by_prediction_bucket()` are in `src/model.py`.

---

## Phase 6.5 findings (step 9: game-environment simulation)

`src/simulate.py` implements the spec's "option 1" correlation approach: `sample_player_week()`, `simulate_matchup()`, `calibration_report()`. `simulate_season()` and playoff odds (step 10) are **not implemented** — stopped here per explicit instruction, after matchup-level results.

**Mechanism: a one-factor Gaussian copula.** Every player-week draws a percentile `u = Phi(sqrt(rho)*z_game + sqrt(1-rho)*z_player)`, where `z_game` is ONE shared standard-normal draw per real NFL game (grouped by `game_id`) and `z_player` is that row's own idiosyncratic draw. Two players sharing a `game_id` have percentile correlation exactly `rho`; players in different games are independent — regardless of which *fantasy* team they're on, so two opposing managers who each started a player from the same real game still move together, matching the spec's framing that a shootout lifts everyone in it. `rho = 0.35` is a fixed constant, not fit to data — the spec's own option 2 ("measure the correlations directly") is explicitly deferred; this is option 1, the one the spec says to start with.

Each player's percentile is mapped through *their own* 5 CQR-calibrated quantile points (`pred_q10_cqr`, `pred_q25_cqr`, `pred_q50`, `pred_q75_cqr`, `pred_q90_cqr`) via a piecewise-linear inverse-CDF, linearly extrapolated beyond the 10th/90th (`np.interp`'s default flat-clipping would understate exactly the tail variance a simulation needs most). Building this surfaced a genuine new finding: CQR's 10-90 and 25-75 interval pairs are widened by *different* constants (see above), which can reintroduce quantile crossing *between* pairs (e.g. `pred_q25_cqr` below `pred_q10_cqr`) even though each pair is individually monotonic after its own widening. `sample_player_week()` defensively re-sorts all 5 points per row before building the inverse-CDF — the same rearrangement fix as `fix_quantile_crossing()`, applied again at a new seam it didn't originally cover.

**K/DST and the ~0.5% of skill-position starters without model coverage** use Sleeper's own point projection as a fixed, zero-variance contribution — all 5 quantile columns set to the same value, since interpolating identical points always returns that constant regardless of the percentile drawn. No special-casing was needed in `simulate.py` itself for this; it falls out of feeding it degenerate quantiles. This mirrors the dashboard's existing convention of showing Sleeper's K/DST numbers labeled as Sleeper's, not the model's — consistent with CLAUDE.md's scope boundary, not a new exception.

**Validated on 204 real historical matchups** (2024 Wk5-17 and 2025 Wk1-17 — the locked evaluation window, minus fantasy-playoff bye weeks; 2,837 starters had real model coverage, 816 were K/DST fallbacks, and 18 (~0.5% of all starter-slots) were unexpected missing-skill-coverage fallbacks not investigated further given the size):

- **Win-prediction accuracy: simulation does not beat the naive baseline.** Simulation picked the actual winner 61.3% of the time (125/204); naive (whichever team has the higher summed point-estimate projection) picked it 62.7% of the time (128/204). The spec's own acceptance criterion — "simulated win probabilities beat a naive baseline... on historical weeks" — **is not met** by this result, reported plainly rather than reframed.
- **The gap is structural and tiny, not a clear loss.** Correlation and variance change the *spread* of a simulated total, not its mean, so the simulation's implied favorite (win probability > 50%) matches naive's point-estimate favorite in 191/204 matchups (93.6%) by construction. The two methods only diverge in genuine toss-up games — every one of the 13 disagreements has a simulated win probability between 43% and 59%. On those 13 games, naive got 8 right and simulation got 5 — a gap fully explainable by chance at this sample size (13 coin flips), not a reliable difference either way. 204 matchups isn't enough to distinguish the two methods; more historical seasons of real matchup data would be needed for a fair test.
- **Calibration is reasonable where there's enough data, untested where there isn't.** Binning both sides of every matchup's probability (408 observations total) into deciles: the six well-populated middle bins (0.2-0.8, 385/408 = 94% of the sample) track the predicted rate within about 2-4 points — e.g. predicted 74.7% in the 0.7-0.8 bin, actual 73.5%. The four extreme bins (0.0-0.2 and 0.8-1.0) show a directional overconfidence pattern — e.g. predicted 93.5%, actual only 50% — but each has just 2-10 observations, nowhere near enough to call this conclusive rather than suggestive.
- **10,000-sim matchups run in a fraction of a second** (see `tests/test_simulate.py`'s timing test), well inside the spec's "seconds, not minutes" bar.

**Bottom line:** the correlation mechanism is implemented and tested as specified, but the two validation checks land differently — calibration looks reasonable in the range where there's enough data to judge it, while the win-accuracy comparison does not show simulation beating naive, and the sample is too small (204 matchups, 13 disagreements) to say whether that's a real gap or noise. Worth revisiting once more historical matchup weeks are available, not something to paper over now.

---

## Design decisions worth preserving (the "why")

Things that took real conversation to arrive at — a new Claude should NOT re-litigate these unless Rohan explicitly asks:

- **Position averages use Top-N**, not "everyone who scored". N is derived from `roster_positions` (direct slots + FLEX share where FLEX = 50% WR / 35% RB / 15% TE). This matches how fantasy managers actually think ("QB1 average", "RB2 average").
- **Draft board columns are fixed per team**, not reshuffled per round. Snake flow is conveyed by ascending/descending pick numbers within the row, not by columns swapping around.
- **No "traded pick" badges on individual draft cells**. Sleeper's `traded_picks` data doesn't cleanly identify which specific picks were affected, so the cross-reference produces false positives. Trade info lives in a separate panel or is omitted.
- **The injury news feed was removed** in favor of Lineup Risks. Sleeper's `injury_notes` field is often just single-word descriptors ("Surgery", "Sprain") — not narrative news. The richer news feed in the Sleeper app comes from a proprietary content service that isn't in the public API. Rather than scrape (legally fuzzy, CORS issues, brittle), we built the Lineup Risks panel which uses only existing data to answer a more actionable question.
- **Play probabilities are league-wide averages, NOT player-specific.** Sleeper has no per-player play likelihood. Q → 75%, D → 25%, O/IR/PUP → 0%. Every pill has a status-specific tooltip explaining this. This is a deliberate honesty choice.
- **"This league" not "your league"** in UI copy. The dashboard is scoped to one specific league; users don't own it, they view it.
- **The league is hard-coded, no runtime switcher.** Deliberate data-governance decision. To change leagues in the future, edit the `LEAGUE_ID` constant at the top of the script and redeploy.
- **Migrated off `nfl_data_py` before writing any dependent code.** nflverse deprecated it; building a portfolio project on a package whose own README says to stop using it is a bad look in a code review. `src/ingest.py` keeps identical function names and signatures so notebooks didn't have to change — including `get_ngs_data(stat_type, seasons)`, which preserves the old argument order even though `nflreadpy.load_nextgen_stats()` takes `(seasons, stat_type)`. The wrapper absorbs the flip.
- **ID columns are normalized to strings in `get_id_crosswalk()`.** The player-ID table arrives with `sleeper_id` as float64 (nulls force the upcast), which turns Sleeper's `"4984"` into `4984.0`. Sleeper's own IDs are strings, so every join silently returns zero rows. `_normalize_id_column()` strips the trailing `.0`. **Every Phase 2+ join depends on this** — if a merge comes back empty, check dtypes first.
- **Sleeper fetch failures raise instead of returning empty.** A silent empty projections frame reads as "this week isn't published yet" when the real cause is a wrong URL or a stale league ID. Loud failure is the honest default here.
- **The custom scorer was built by diffing against Sleeper's own results**, not by reading the settings dict. `validate_against_sleeper()` pulls `/league/{id}/matchups/{week}`, which returns the per-player points Sleeper actually awarded — ground truth, since Sleeper ran the league. Seven non-standard rules turned up this way that are documented nowhere:
  1. `fum` counts `fumbles_total`, **not** the sum of rushing/receiving/sack fumbles — there are fumble categories (aborted snaps, muffed returns) outside those three.
  2. `fum` and `fum_lost` **stack**: a lost fumble is −2, a self-recovered one −1.
  3. `fum_rec` (+1) does **not** apply to offensive players. 19 of 19 rostered players with a recovery reconcile without it.
  4. `fgm_yds_over_30` is **per kick**, not on aggregate distance. Computing it as `total_distance − 30 × made` lets short field goals eat into long ones' credit.
  5. Blocked PATs count as misses (`pat_blocked` adds to `xpmiss`).
  6. `fgmiss` (−1) only applies to misses **under 50 yards**. Established empirically: 4/4 kickers whose only miss was short reconciled exactly; 0/12 with a 50+ miss did.
  7. `pass_int_td` needs play-by-play, and requires `td_team == defteam`. Without that condition, a defender fumbling an interception return into his own end zone scores as a pick-six when it's the opposite.
- **K and DST are deliberately out of scope for the projection model.** Kicker output depends on how often the offense stalls in FG range, which is close to noise week to week; DST would need a team-defense model layered on an offense model. The scorer handles both correctly, but the dashboard keeps showing Sleeper's projections for K/DST, labeled as Sleeper's. This keeps Phase 6 finishable.
- **Monte Carlo simulation is in scope; Markov chains are not.** Simulation (playing the week out thousands of times with randomness, then counting outcomes) answers questions no platform surfaces: matchup win probability, playoff odds, and "who is most likely to exceed 25 points" for start/sit calls. Markov chains were considered and rejected — weekly fantasy output isn't Markovian (next week depends on opponent, health, and game script, not on last week's "state"), and the one place they do fit football is drive-level modeling, where nflverse's existing EPA columns are already better than anything we'd rebuild.
- **The simulator must handle correlation between teammates.** Sampling each starter independently makes simulated totals cluster too tightly, producing overconfident win probabilities (showing 85% when the truth is 65%). Approach: draw the game environment first from the Vegas total, then draw each player's share within it. Any probability shown on the dashboard needs a calibration plot behind it — a confidently wrong simulator is worse than none, because it looks authoritative.
- **`fantasy_points_ppr` is not a valid target.** It's full PPR. The model trains on `custom_points` from `compute_custom_score()`, and any comparison against Sleeper's projections must use the same.
- **xFP (`src/usage.py::add_xfp_features()`) is RB/WR/TE only.** The bucket rate table covers targets and carries — there's no bucket for a QB's own pass attempts, so a QB's `xfp` would only reflect their rushing, leaving their (much larger) passing production unmodeled. That would make every passing QB read as wildly "over expected" regardless of actual luck — an artifact of what wasn't built, not a real signal. `xfp`/`fp_over_expected` are explicitly forced to null for QB rows rather than left to silently mislead; fixing this means adding a passing-yardage bucket family, not reinterpreting the null.
- **xFP strips most, not all, of the persistent skill signal — verified, not assumed.** Split-half correlation of `fp_over_expected` (first 9 weeks of a season vs. last 9) is **0.22**, against **0.73** for raw `custom_points` over the same split. If xFP fully isolated touchdown luck, this would be near zero; it isn't, so treat the residual 0.22 as real, modest, repeatable opportunity-independent efficiency (contested-catch ability, red-zone role) that a two-dimensional air-yards/field-position bucket can't fully separate from league-average value — not a bug, but a disclosed limit of this bucket scheme.
- **The xFP rate table is an expanding window that crosses season boundaries — a two-season compromise, not a permanent design.** With only 2024-2025 cached, resetting the rate table at each September would leave both 2025's early weeks and *all* of 2024 (which has no prior season at all) starved of data for the rarer buckets. Week 1 of 2025 currently draws on the entirety of 2024. **Revisit this once more seasons are cached** — with 4-5 years of history, an in-season-only or recency-weighted window becomes viable and would better reflect any real year-over-year shift in league-wide scoring efficiency.
- **xFP deliberately excludes two-point conversion attempts** from both the rate table and the opportunities it scores. They're rare (278 across 2024-2025) and always run from the 2-yard line, which would badly distort the inside-10 target buckets. The tradeoff: a player who converts a 2pt catch or run shows real `custom_points` about 2 points higher than `xfp` that week, with no counterpart in the model — a small, known, one-directional gap, confirmed on real player-weeks during validation (see Verification status).
- **Legal/financial advice pattern for LLM discussion**: When Rohan asked about "modern methods gaining traction," the answer was tiered (Strongly Suggest / Industry-Standard Tooling / Frontier) with an **explicit warning against shoehorning LLMs into a tabular regression project**. Follow the pattern — don't just list every trendy technique. Match tool to problem.

---

## Verification status

Be precise about what's actually been confirmed, so a fresh session doesn't inherit
assumptions as facts.

| Claim | Status |
|---|---|
| `nfl_data_py` is deprecated in favor of `nflreadpy` | **Verified** — nflverse's own announcement |
| `nflreadpy` function names + signatures used in `src/ingest.py` | **Verified** against the published API reference |
| Local env: Python 3.12.9, `.venv` kernel resolves, `src/` importable | **Verified** |
| Package install completed | **Verified** — nflreadpy 0.1.5, polars 1.43.2, pandas 3.0.5, pyarrow 25.0.0 |
| `01_data_ingestion.ipynb` runs end-to-end | **Verified** — 98,263 pbp rows, weekly stats, snaps, NGS, schedule all cached |
| Sleeper league ID is current | **Verified** — 2026 is `1389706592789733376`, chained from 2025 via `previous_league_id`. Sleeper mints a new ID each season for **all** league types, not just dynasty. Needs updating each August. |
| ID crosswalk joins nflverse ↔ Sleeper | **Verified** — Sleeper `'4984'` → gsis `'00-0034857'`, both strings, 37 weekly rows returned |
| Custom scorer reproduces league scoring | **Verified** — 100% exact, 0 mismatches across 739 rostered player-weeks (2025 wks 5/8/10/12/15), every position including K |
| pandas 3.x compatibility | **Verified in practice** — full pipeline runs on pandas 3.0.5 / numpy 2.5.1 |
| `pfr_player_id` → `gsis_id` join (needed for snap share) | **Verified** — 99.67% match for QB/RB/WR/TE (14,281/14,328 snap-count rows, 2024-2025). The 0.33% miss is fringe/practice-squad players absent from the crosswalk entirely, not a format bug. O-line (T/G/C/OL) and long-snapper (LS) match at 0-18% — `load_ff_playerids()` carries almost no offensive linemen (53 OT, 6 C, 1 T, no `G` category in 12,470 rows) since it's sourced from fantasy-platform rosters and o-linemen are never fantasy-relevant. Out of scope for the projection model regardless (skill positions only), so this doesn't block Phase 2b. |
| Season/week boundary handling | **Partly verified** — requesting a season nflverse hasn't published (e.g. 2026 in the offseason) 404s with a raw traceback rather than a readable message |
| xFP (`add_xfp_features()`) reproduces `custom_points` on real plays | **Verified** — per-play scores summed over a player's actual weekly plays matched `custom_points` exactly on 6 of 8 spot-checked player-weeks (2025 Wk10 WR/RB sample); the other 2 differed by exactly −2.00, fully explained by the deliberate two-point-conversion exclusion (see design decisions above) — the only known discrepancy. Caught and fixed one real bug in the process: an early version silently added `0.04 × passing_yards` to every target's score because raw pbp's own play-level `passing_yards` column leaked into the synthetic play frame. |
| `src/simulate.py` matchup simulation beats a naive baseline on real historical matchups | **Verified false** — tested against 204 real Sleeper matchups (2024 Wk5-17, 2025 Wk1-17); simulation and naive disagreed on the favorite in only 13 of them, and naive was right on more of those (8 vs 5) — a difference explainable by chance at this sample size, not a demonstrated edge for simulation. Calibration itself looks reasonable in the well-populated 0.2-0.8 probability range. See **Phase 6.5 findings**. |

## What's outstanding

- **`PHASE_2B_6_SPEC.md`** at the repo root is the working spec for Phases 2b, 6, and 6.5. Fold it into this doc and `NOTEBOOK_OUTLINE.md` once those phases are complete.
- Bump the Phase 8 CI workflow from Python 3.11 to 3.12
- **Phase 8 will need incremental season fetching, not a full refetch every run.** `SEASONS` defaults to 2018-2025 (8 seasons) as of the Phase 6 data-volume result — the cached pbp file alone is 142 MB for that range (`data/raw/` is 311 MB total), and a weekly automation job re-downloading all 8 seasons from scratch every run is wasteful and slow. The right shape is closer to "keep 2018-2024 cached as-is, fetch only the current season's new weeks" — not designed yet.
- Update `DEFAULT_LEAGUE_ID` in `src/ingest.py` each August when Sleeper rolls the league over
- Consider a season guard in `ingest.py` so requesting an unpublished season fails with a readable message instead of a 404 traceback
- Push the latest `index.html` changes to GitHub Pages (all recent work is local)
- **Activity feed panel** sizing vs matchups panel — layout issue, minor
- **NFL sidebar** currently shows preseason week labels; should default to last completed regular-season week
- **Season Leaders panel** on the dashboard is pending replacement
- **Build out the Python notebook pipeline**, phases 2 through 8
- **Historical champion data** — plan is to maintain a small `champions.json` file by hand for the league's history

---

## What Rohan values in a working session

Observed over many hours of collaboration:

- **Direct communication.** Rohan wants honest answers about tradeoffs, not just "sure, doing it now." When a request has issues (like fake radar data or false-positive trade badges), name them clearly.
- **Concise summaries after code changes.** After every change, provide: what changed, why, and how to commit/push. No fluff.
- **Iterative shipping.** Multiple small round-trips are preferred over one giant PR. Rohan often catches issues at each stage that would compound if bundled.
- **Career-relevant technique choices.** For the notebook, Rohan explicitly wants to pick up techniques that show up in real data-science roles. Don't recommend obscure or hype-driven tools without justification.
- **Straight talk on limitations.** When something can't be done cleanly with free data (scraping tradeoffs, per-player play likelihood), say so plainly. Don't invent workarounds that produce untrustworthy output.

---

## Where to find things in `index.html`

Rough map — line numbers drift as edits happen so use grep:

- **Config** (LEAGUE_ID, API) — near line 1900
- **Data fetchers** (`fetchAllRealData`, `fetchWeekStats`, `fetchWeekProjections`) — around 1880-2000
- **Custom scoring** (`computeCustomScore`, `resolvePts`) — around 1930
- **Mappers** (`mapSleeperPlayers`, `mapInjuryStatus`, `formatInjuryDate`) — around 2700
- **State object** — near 2340
- **View renderers** — search `function render{View}View`:
  - `renderDashboardView`, `renderMatchupsView`, `renderMatchupDetail`, `renderTeamsView`, `renderTeamDetail`
  - `renderDraftView` — around 5900
  - `renderInjuryView` — around 6100
  - `renderPlayersView` — around 4920
  - `renderPlayerDetail` — around 5100
  - `renderComparisonView` — around 5450
- **Helpers** — `getTeamName`, `getOwnerHandle`, `buildLineupForRoster`, `getAvailableWeeks`, `getStatLine`, `getInitials`, `positionStarterCount`

---

## Getting started (new conversation prompt template)

If you're a new Claude picking this up, expect Rohan to say something like:

> "Continuing my FanTeasy Stats project. Attached is the current `index.html`, the notebook outline, and the project context doc. Ready to work on Phase 2b of the notebook."

Read this doc, skim `NOTEBOOK_OUTLINE.md` for the current phase, and confirm you understand the project before proposing changes. Don't ask questions this doc already answers.

Check the **Verification status** table before treating any pipeline claim as settled.
