# FanTeasy Stats — Project Context

This document captures the "why" behind the FanTeasy Stats project so a new conversation can pick up seamlessly. Read this first before starting any new work.

> **Last updated:** August 2026. Phase 1 (ingestion) and Phase 2a (custom
> scoring) are complete and verified against live data. Phase 2b (steps 1-5,
> full feature table + notebook) is also complete and verified — see
> **Phase 2b progress** below. Phase 6 (projection model) was investigated
> and concluded rather than shipped — the model loses to simple baselines
> at every position, diagnosed as a signal-to-noise ceiling, not a bug —
> see **Phase 6 findings** below. Phase 6.5 (simulation) and Phase 6 steps
> 8-10 are still design sketch, contingent on what happens with Phase 6
> next — see **Verification status** near the end before treating any
> pipeline claim as settled.

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
| 2b | Usage + efficiency features from pbp | Steps 1-4 of 10 **done** (`src/usage.py`) — see **Phase 2b progress** below. Steps 5-10 remain. |
| 3 | Role classification — rule-based Pocket Passer / 3-Down Back / Slot / etc. | Not started |
| 4 | Radar metrics — 0-100 percentile normalization within position | Not started |
| 5 | Heatmap zones — field-location frequency tables | Not started |
| 6 | Projection model — XGBoost/LightGBM regression with time-series CV | **Investigated and concluded** — Formulation A loses to every baseline at every position; diagnosed as a signal-to-noise ceiling, not a bug or a tuning gap. See **Phase 6 findings** below. Not abandoned (real, tested code exists in `src/model.py`) and not "done" in the sense of shipping a model — the honest outcome was deciding not to ship one yet. |
| 6.5 | Monte Carlo simulation — win probability, playoff odds, floor/ceiling | Not started |
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

## Phase 6 findings (Formulation A investigated and concluded)

`src/model.py` trains one untuned LightGBM model per position (QB/RB/WR/TE), direct prediction of `custom_points`, expanding-window walk-forward validation (no shuffling, no KFold). The honest conclusion: **the current feature set doesn't yet forecast better than simple baselines, and that's a signal-to-noise finding, not a tuning gap.**

**Model A loses to every baseline at every position.** Corrected numbers (see below for why "corrected" matters), best-performing configuration per position:

| Position | Model | season_to_date_avg | trailing_3wk_avg | sleeper_proj | trailing_xfp |
|---|---|---|---|---|---|
| QB | 6.83 MAE | 6.61 | 6.79 | **5.51** | N/A (xFP is RB/WR/TE only) |
| RB | 4.44 MAE | 4.29 | 4.50 | **3.86** | 4.26 |
| WR | 4.10 MAE | 4.06 | 4.27 | **3.77** | 4.04 |
| TE | 3.27 MAE | 3.13 | 3.31 | **2.86** | 3.02 |

Sleeper's projection wins clearly everywhere — it encodes beat-writer/injury/depth-chart information nflverse-derived features structurally can't see.

**Feature-count reduction is not the fix — and the first attempt at this diagnosis was itself wrong.** An initial ablation (rank all ~180 features by SHAP importance from one model trained on the *full* dataset, then test top-10/25/50/all) showed a clean "peaks at 25, degrades at scale" pattern at every position — textbook overfitting, or so it looked. Redone properly, deriving the SHAP ranking *inside each walk-forward fold, from that fold's own training data only*: the pattern **disappeared**. 25 features doesn't peak anywhere — it's the single worst option at QB, and RB/WR/TE all prefer the full feature set. The first result was an artifact: ranking features on the full dataset lets the ranking "see" the same weeks later used to score the ablation, which flatters small feature sets in a way that doesn't hold up once the ranking itself is confined to what a fold could actually have known. **This is a distinct failure mode from the training-data leakage `tests/test_no_leakage.py` guards against** — that suite verifies no *feature value* depends on future weeks; it says nothing about whether the *choice of which features to use* was made with future information. Worth remembering for any future feature-selection work in this project, not just this one model.

**The residual test — the most direct diagnostic run.** Trained the model to predict `custom_points − baseline_season_to_date_avg` (isolating whether the feature set adds anything on top of "how has he been scoring lately," the same idea as Formulation B but against the season-to-date average instead of Sleeper). The model did not collapse to predicting near-zero — predicted and actual residuals correlate at r≈0.11-0.15 across all four positions, a real, non-degenerate signal. But reconstructing `baseline + predicted_residual` produced a *worse* MAE than the raw baseline at every position (4-7% worse), and worse Spearman too. The signal is real and too weak to survive noise at this data volume.

**Conclusion: these features are better at explaining than forecasting.** `target_share_ewm3`, `xfp_s2d`, and the rest of the Phase 2b feature table clearly describe *what happened* — the SHAP rankings surface exactly the volume/opportunity signals a manager would want to see (that part worked). They don't yet predict *what happens next* better than a plain trailing average. The dashboard reflects this honestly: it shows Sleeper's own projections, labeled as Sleeper's, rather than a model that loses to a three-line pandas baseline. The Phase 2b feature table isn't wasted work — it powers the analytical panels (usage trends, xFP/luck, role changes) that Sleeper doesn't show, which was always half the point (see **What this is actually for** in `PHASE_2B_6_SPEC.md`). Revisiting Formulation B, more data, or a materially different feature strategy is a future decision, not something concluded here.

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

## What's outstanding

- **`PHASE_2B_6_SPEC.md`** at the repo root is the working spec for Phases 2b, 6, and 6.5. Fold it into this doc and `NOTEBOOK_OUTLINE.md` once those phases are complete.
- **Phase 2b** — usage and efficiency features from pbp (target share, air-yards share, red-zone touches, aDOT, snap share). Start with snap share to shake out the `pfr_player_id` join early.
- Bump the Phase 8 CI workflow from Python 3.11 to 3.12
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
