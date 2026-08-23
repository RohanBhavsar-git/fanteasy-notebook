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
> **Phase 6 findings**), and Phase 6.5 (`src/simulate.py` — game-
> environment sampling, matchup simulation, season simulation, playoff-
> qualification odds) is done — see **Phase 6.5 findings**. Both
> validations land the same honest way: calibration is reasonable where
> there's enough data to judge it (204 real matchups; 8 season-snapshot
> combinations), and the win-accuracy "beat naive" criterion was revised
> after learning it was testing something correlation/variance don't
> change (a total's mean) rather than something they do (its spread).
> Rho sensitivity for season-long playoff odds is small (~0.7pt mean
> across rho 0.2-0.5) — smaller than the compounding intuition alone
> would suggest, for a reason documented there. Championship/bracket-
> round odds are a separate, not-yet-started piece of work. Phase 3'
> (usage trend signal — `src/usage.py` Family 7) is done, replacing the
> original Phase 3 role-classification idea in `NOTEBOOK_OUTLINE.md`
> outright rather than deferring it: a role label forces every player into
> one bucket of a fixed set and says nothing about whether that role is
> changing, which is what a trend signal is for. The 3-week EWM window
> (already `EWM_HALFLIFE`, reused rather than re-derived) was validated
> empirically against 4- and 5-week alternatives before being kept — see
> **Phase 3' findings** below. Phase 8 (automation) is done — **two**
> GitHub Actions workflows, not one: `retrain.yml` (`workflow_dispatch`
> only, never scheduled) fetches all historical seasons, walk-forward-
> validates, and commits a model artifact; `weekly-update.yml` (Tuesday
> mornings in-season plus `workflow_dispatch`) is inference-only — it
> never retrains, loads the committed artifact, fetches only the current
> season, and commits the regenerated JSON. Verified end-to-end locally
> against real cached data before either workflow's first real run — see
> **Phase 8 findings** below for the mechanism (a `history_seed` embedded
> in the artifact is what makes "current season only" possible at all),
> the artifact size, and two disclosed limitations (a candidate universe
> that's only as fresh as the last retrain; `xfp` running noisier in a
> season's first few weeks under weekly-only inference). Phase 4 (radar
> percentiles) is done — six axes per position chosen from already-computed
> Family 1-4/xFP features (not `NOTEBOOK_OUTLINE.md`'s aspirational axis
> list, which named several stats never actually built), percentiled
> against a real startable-player pool ported from `index.html`'s own
> `positionStarterCount()`, with an honest ineligible state below a
> games-played floor rather than a misleading shape — see **Phase 4
> findings** below. Phase 5 (field heatmap zones) is also done — real
> target/carry zones derived directly from play-by-play (never from a
> pre-aggregated feature column), a fixed-shape SVG grid per zone kind so
> two players stay visually comparable, and thin zones shown as `sparse`
> rather than dropped or merged — see **Phase 5 findings** below, which
> also documents a real `nflreadpy.load_pbp()` bug this phase's own
> verification run caught and fixed before it reached a committed export.
> Phase 9 (season archives + selector) is also done — replaces the old
> silent `previous_league_id` fallback with an explicit season picker that
> switches Sleeper league data, the NFL sidebar, and which export loads
> all together, fixing a real incoherence (a full real season in the KPI
> cards next to "0 of 5 games played" in the radar). Only 2025 is archived
> so far, a deliberate scope decision made AFTER checking real feasibility
> for all 5 of this league's completed seasons (2021-2025) — see **Phase 9
> findings** for the concrete numbers, including a real candidate-selection
> bug (dropping 47-89% of a season's real rostered players) that had to be
> fixed before archiving was viable at all. See **Verification status**
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
- **Later phases**: LightGBM (projections, plus its own sklearn wrapper — `scikit-learn` is a direct dependency of that, not a role-clustering tool; role classification was dropped, see `NOTEBOOK_OUTLINE.md`'s Phase 3'), SHAP (explainability, done), Optuna (hyperparameter tuning, not used — no model in this pipeline is tuned), MLflow (experiment tracking, not used)
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
- **Season selector** (Phase 9, top nav) — switches the ENTIRE page (Sleeper league/rosters/matchups, the NFL sidebar, and which `player_advanced_stats.json`/archive is loaded) to a different season in one action. Replaces the old silent `previous_league_id` fallback outright — see Phase 9 findings. Currently offers the live 2026 season and the 2025 archive; defaults to whichever actually has real data (2025, since 2026 is still pre-draft).
- League standings table with sortable columns
- Weekly matchup cards with real scores
- League-wide KPI cards (top scorer, biggest blowout, closest game, etc.)
- Activity feed (transactions, waivers, trades) — real Sleeper data
- **Usage Trending panel** (Phase 8 round 1) — biggest risers/fallers by target/carry/snap-share trend signal, one row per player showing current share + direction. `red_zone_share` deliberately excluded (hold-rate under 50%, see Phase 3' findings). Honest empty state ("No trend signal yet... populate from Week 6 onward") when `player_advanced_stats.json` hasn't accumulated 5 games this season yet — this is what actually renders as of the 2026 pre-season export.
- **xFP Regression panel** (Phase 8 round 1) — season `fp_over_expected` leaders/laggards, RB/WR/TE only, explicitly labeled "luck indicator, not a prediction." Pulls from the most recently COMPLETED season, so unlike Usage Trending it already shows real 2025 numbers even in the 2026 pre-season export, not an empty state.
- **Win probability on matchup cards** (Phase 8 round 2) — "N% sim" badge next to each team, from `src/simulate.py`'s `simulate_matchup` via the export's new `simulation` block. Only shown when the export's `simulation.week` matches the currently viewed week (same "don't show a stale number" rule `getMyProj()` already follows) — null on the 2026 pre-draft export (no real matchups exist yet), which is what actually renders today. A visible caption (not tooltip-only) beneath the matchup list states both the calibration limitation and that this doesn't out-pick "higher projection wins."
- **Playoff Odds column in the standings table** (Phase 8 round 2) — from `simulate_season`, whole-percent, with the same visible caveat caption beneath the table. Column always renders; shows `—` per row (not hidden) when there's no simulation to show.

### Matchups tab
- Weekly matchup cards with 🏆 winner badge, blowout/nail-biter emoji markers
- **Win probability badges** (Phase 8 round 2) — same mechanism and caveat caption as the Dashboard's matchup cards, shown next to each team's W-L record line
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
- **Opportunity Shares panel** (Phase 8 round 1) — plain snap %/target share/carry share/red-zone share, no interpretation layered on. Carry share and red-zone share are read from the export's `trend.<feature>.current` (no `usage.*` counterpart exists for those two); red-zone share specifically uses the properly-combined `trend.red_zone_share`, not the two separate `usage.rz_target_share_ewm3`/`rz_carry_share_ewm3` (different denominators, can't be summed — see design decisions below). Per-stat empty state ("— · No games yet") when a value is null, not a whole-panel placeholder.
- **Position profile panel** (Phase 4) — a real 6-axis Chart.js percentile radar per position (QB/RB/WR/TE, each axis its own genuinely-informative mix of volume/share/efficiency/situational stats, not a forced identical template), percentiled against this league's real startable-player pool (`position_starter_counts()`, ported from the Weekly Production chart's own `positionStarterCount()`). A raw-value list below the chart shows the actual stat behind each percentile — "Nth percentile" is a rank among startable players at the position, explicitly labeled as such, not a 0-100 quality score. Honest "Not enough games yet" empty state (with the real games-played count) when a player hasn't cleared the games floor; the earlier "Awaiting model output" placeholder still shows for an export that predates this key entirely.
- **Field heatmap panel** (Phase 5) — a real SVG field-zone grid per position, derived directly from play-by-play (not from any pre-aggregated feature table): receivers (WR/TE, and RB's receiving work) zoned by air-yards depth x field position, runners by direction x field position, QBs by pass location x depth. A fixed-shape grid per zone kind so two players' charts stay visually comparable — an empty cell means zero real plays there, a solidly-colored cell is a real, well-sampled tendency, and a dashed/`~`-marked cell is real but thin (below `HEATMAP_SPARSE_THRESHOLD` plays) so it's shown, not hidden, without being read as confidently as a solid one. RB gets two independent grids (rushing and receiving, matching `getHeatmapTitle()`'s "Rushing Direction & Receiving") since a carry and a target don't share a denominator. Same games-played eligibility floor and honest empty states as the radar panel above.
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
| 2b | Usage + efficiency features from pbp | Steps 1-5 of 10 **done** (`src/usage.py` + `03_usage_features.ipynb`) — see **Phase 2b progress** below. Steps 6-10 (Phase 6 model A/B + quantile/SHAP/CQR, Phase 6.5 game-environment + season simulation) are all done — see **Phase 6 findings** and **Phase 6.5 findings**. Remaining work is Phase 7 (JSON export) and Phase 8 (automation), plus anything beyond this spec's original 10-step list (e.g. championship/bracket odds). |
| 3' | Usage trend signal — replaces role classification (see `NOTEBOOK_OUTLINE.md`'s Phase 3') | **Done** — `src/usage.py` Family 7 (`add_trend_features`, `get_usage_trend_leaders`) + `04_usage_trends.ipynb`. Window (3-week EWM half-life) and direction threshold (z > 0.25) both validated against real hold-vs-revert data, not assumed — see **Phase 3' findings** below. Wired into Phase 7's export as a new `trend` key. |
| 4 | Radar metrics — 0-100 percentile normalization within position | **Done** — `src/export.py`'s `RADAR_METRICS`/`position_starter_counts()`/`build_radar_snapshot()`, wired into the export as a new `radar` key and rendered as a real Chart.js radar on Player Detail. See **Phase 4 findings** below. |
| 5 | Heatmap zones — field-location frequency tables | **Done** — `src/usage.py`'s `receiving_zone_plays`/`passing_zone_plays`/`rushing_zone_plays` (real pbp bucketing) plus `src/export.py`'s `build_heatmap_snapshot()`, wired into the export as a new `heatmap` key and rendered as a real SVG field-zone grid on Player Detail. See **Phase 5 findings** below. |
| 6 | Projection model — XGBoost/LightGBM regression with time-series CV | **Investigated, not shipped.** The earlier 2-season conclusion ("loses to every baseline") was premature — it was a data-volume ceiling, not a feature-quality one. At the 8-season default, Formulation A beats `season_to_date_avg`/`trailing_3wk_avg` at every position and closes (without closing entirely) the gap to `sleeper_proj`. Formulation B (predicting the residual against Sleeper) does not improve on Formulation A. Step 8 (quantile floor/ceiling models + SHAP) is done: coverage is measured and honestly overconfident (67-75% actual vs. 80% target for the 10th-90th interval), and SHAP shows nothing that looks like a leak. See **Phase 6 findings** below. Not abandoned (real, tested code exists in `src/model.py`) and not "done" in the sense of shipping a model — the honest outcome is still deciding not to ship one yet. |
| 6.5 | Monte Carlo simulation — win probability, playoff odds, floor/ceiling | Steps 9-10 **done** (`src/simulate.py` — game-environment sampling, matchup + season simulation, playoff-qualification odds) — see **Phase 6.5 findings**. Validated against 204 real historical matchups and 8 season-snapshot combinations: calibration is reasonable where there's enough data to judge it in both. Untuned rho=0.35 sensitivity for playoff odds is small (~0.7pt mean, ~3pt max across rho 0.2-0.5). Championship/bracket-round odds are a separate, not-yet-started piece of work. |
| 7 | JSON export — assemble `player_advanced_stats.json` | **Done** — `src/export.py` + `07_export_json.ipynb`. Predicts the real upcoming week (2026 Wk1) by reusing the existing point-in-time-safe feature pipeline on a stub row, not new future-facing logic. 300 players (2026 league is `pre_draft` as of this run, so scope is top-300 by projection until the real draft happens — picks up real rosters automatically on a re-run, no code change needed), 219 KB (grew from 132 KB after Phase 3''s `trend` key was added), crosswalk match rate 98.99%. `trend` (Phase 3') is now a real per-player key, entirely null in the current pre-draft/Wk1 export by construction (no in-season games yet) — will populate from week 6 onward once real games exist. `radar`/`heatmap` (Phases 4-5) can still slot in as new per-player keys later without restructuring anything. |
| 8 | GitHub Actions automation | **Done** — two workflows, not one: `.github/workflows/retrain.yml` (train + validate, `workflow_dispatch` only) and `.github/workflows/weekly-update.yml` (inference only, Tuesdays in-season + `workflow_dispatch`). See **Phase 8 findings** below. |

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

## Phase 3' findings

Replaces `NOTEBOOK_OUTLINE.md`'s original Phase 3 (rule-based role
classification) outright, not just reorders it — see that doc's Phase 3'
section for why a role label was the wrong shape for this problem. Built
as `src/usage.py` Family 7 (`add_trend_features`, `get_usage_trend_leaders`),
validated in `04_usage_trends.ipynb`.

**The window question, settled with data.** Before writing any production
code: does a usage rise measured over a 3-week EWM half-life actually
predict the FOLLOWING game's usage staying above season baseline
("holds"), rather than reverting? Checked against 4- and 5-week windows
too, for `target_share`, `carry_share`, `offense_pct` (snap share), and a
new combined `rz_opportunity_share`, on 21,000+ real player-weeks per
feature (2018-2025, `games_played >= 5` eligibility so 3/4/5 are compared
on identical rows):

| feature | window | hold-rate | corr with next-week usage |
|---|---|---|---|
| target_share | 3 | 0.498 | 0.102 |
| target_share | 4 | 0.497 | 0.100 |
| target_share | 5 | 0.496 | 0.099 |
| carry_share | 3 | 0.412 | 0.212 |
| carry_share | 4 | 0.410 | 0.207 |
| carry_share | 5 | 0.408 | 0.204 |
| offense_pct | 3 | 0.619 | 0.279 |
| offense_pct | 4 | 0.616 | 0.273 |
| offense_pct | 5 | 0.615 | 0.270 |
| rz_opportunity_share | 3 | 0.389 | 0.049 |
| rz_opportunity_share | 4 | 0.388 | 0.049 |
| rz_opportunity_share | 5 | 0.387 | 0.048 |

3 wins on **every** feature, on **both** metrics, monotonically (3 > 4 > 5
throughout) — not a close call decided by cherry-picking a per-feature
winner. This is also exactly the already-existing `EWM_HALFLIFE` used
throughout Family 6, so the trend signal reuses the existing `_ewm3`/`_s2d`
columns for target_share/carry_share/offense_pct directly, with no second,
competing half-life constant introduced.

**Comparability across players required normalizing by volatility, not
just picking a window.** The raw `ewm3 − s2d` gap isn't comparable across
players — a bell-cow RB's `target_share` swings more in absolute
percentage points than a committee back's, so a fixed raw-gap threshold
would flag high-volume players more often for no informative reason.
Dividing by the player's own season-to-date volatility (the already-
existing `<feat>_vol` expanding-std column) into `signal = gap / vol`
turned out to matter empirically, not just conceptually: at `z > 0.5`,
hold-rate reaches 0.71 (target_share) / 0.60 (carry_share) / 0.79
(offense_pct) — well above the ~0.39-0.62 hold-rate a loose "any positive
gap" (`z > 0`) threshold gets on the same features. `z > 0.25` was kept as
`TREND_DIRECTION_THRESHOLD` over `z > 0.5` specifically because `z > 0.5`
only flags ~1-2% of eligible weeks (too sparse for a riser/faller list
anyone would actually check weekly), while `z > 0.25` still meaningfully
beats the `z > 0` baseline and flags a workable ~13-20% of weeks per
direction.

**Minimum games played: 5, not picked arbitrarily.** Raising the
eligibility floor from 2 to 8 prior in-season games monotonically improved
correlation with next-week usage (target_share: 0.085 at ≥2 games vs.
0.123 at ≥8) — the season-to-date mean/vol a thin-sample player's signal
divides by are themselves noisy early. 8 games would exclude nearly half
of every season from ever appearing on a riser/faller list, though, so 5
was kept as the floor — the same one the window/threshold study above was
run under, not a second, separately-justified number.

**Honesty note: `rz_opportunity_share` is real but clearly noisier than
the other three.** Even at the validated window and threshold, its
hold-rate stays under 50% (0.42 at `z > 0.25`) — a "rise" reverts more
often than it holds, and averaging the next TWO games instead of one only
brings it to 0.47. Red-zone opportunities are a low-volume, high-variance
event category; this wasn't hidden or dropped (it was explicitly asked
for) but is shipped with an explicit caveat in `src/export.py`'s
`CAVEATS`, the same honesty pattern already used for "Sleeper's
projections are more accurate than this model."

**`rz_opportunity_share` itself is new** — Family 4 has `rz_target_share`
and `rz_carry_share` separately, with different denominators (team RZ
targets vs. team RZ carries), so they can't be summed into one honest
combined share directly. Built as `(rz_targets + rz_carries) /
(team_rz_targets + team_rz_carries)`, rolled with the same halflife=3
mechanics as `add_rolling_features` — but deliberately kept OUT of
`ROLLING_SOURCE_COLUMNS`/`FEATURE_COLUMNS`. Adding it there would silently
make it a new `src/model.py` training feature and retroactively change the
already-published, already-validated Phase 6 model without anyone asking
for that — this family is a display/export-layer derived feature,
downstream of the model, not a new model input.

**Current export state**: the committed `player_advanced_stats.json`'s
`trend` block is entirely null for every player — expected, not a bug.
2026 is `pre_draft` and this is Week 1: `games_played == 0` for every stub
row (nothing has been played yet THIS season), the same reason
`usage.target_share_ewm3` is already null at week 1 today. Will populate
naturally from week 6 onward (once real players clear the
`MIN_GAMES_FOR_TREND` floor) with no code change needed — same pattern as
Phase 7's own rostered-players note above.

---

## Phase 4 findings

`src/export.py`'s `RADAR_METRICS`, `position_starter_counts()`, and
`build_radar_snapshot()` — a `radar` sibling key alongside
`projection`/`usage`/`trend`/`xfp`, computed entirely in Python and wired
through `scripts/weekly_update.py` and `07_export_json.ipynb` the same way
every other export-layer field has been. `index.html`'s Position Profile
panel (Player Detail) now renders a real Chart.js radar instead of the
placeholder card.

**Axes are chosen from what Families 1-4/xFP actually compute, not from
`NOTEBOOK_OUTLINE.md`'s Phase 4 sketch.** That sketch (written before any
feature work existed — see CLAUDE.md's "Phases 3-8... intent, not tested
code") names several stats this pipeline has never built: Big Play Rate, TD
Rate, Sack %, Explosive Runs, Contested Catch %, YPRR, Blocking Snaps, Dome
%, ST TDs. Six real axes per position were picked instead, per position,
from already-computed `_s2d` (season-to-date expanding mean) columns —
`_s2d`, not the `_ewm3` value the Opportunity Shares panel and trend
indicator already show elsewhere on the same page, because a "position
profile" is meant to characterize a real-season sample, not the last three
weeks, and `_s2d` is the more stable input for a percentile RANK
specifically. QB: Pass Volume, Rush Volume, Yards/Carry, Scramble Rate,
EPA/Dropback, CPOE. RB: Touch Volume, Touch Share, Target Share,
Yards/Carry, Goal-Line Share, Snap Share. WR: Target Share, Air Yards
Share, aDOT, Catch Rate, YAC/Reception, Red-Zone Target Share. TE: same as
WR but Snap Share in place of Air Yards Share (TE snap share varies far
more than WR's and is the more differentiating "real receiving threat vs.
blocker" signal at that position). No NGS-only column
(`avg_separation`/`avg_cushion`/`time_to_throw`) is used, so no axis is
ever null for a position it's assigned to.

**Percentile pool is "startable players," not "everyone at the position" —
ported from `index.html`, not reimplemented independently.**
`position_starter_counts()` is a direct Python port of `index.html`'s
`positionStarterCount()` (same direct-slot + FLEX-share + SUPER_FLEX logic,
including JS's round-half-up rounding rather than Python's round-half-to-
even, in case a future league config lands exactly on a `.5` boundary) —
pinned against this league's REAL `roster_positions` in
`tests/test_export.py::test_position_starter_counts_matches_this_leagues_real_roster_positions`
so the two copies can't silently drift apart. At 14 teams this league's
real startable counts are QB=14, RB=33, WR=35, TE=16 — WR's 35 matches the
number already published in the Weekly Production chart's own footnote.
The pool itself is the top-N players per position by season-to-date total
`custom_points`, restricted to players who themselves clear the
games-played floor — "startable" means both "ranks near the top" and "has
enough sample to rank honestly." Every candidate at the position (pool
member or not) is percentiled against this same pool, not just pool
members — a bench player's profile still reads as "where would this rank
among actual starters," which only works if the denominator excludes deep
backups (a bench-inclusive pool would compress every real contributor
toward the top of the range, the exact failure mode this was built to
avoid).

**Eligibility is a single whole-player gate, reusing Phase 3's
`MIN_GAMES_FOR_TREND` (5), not a second, separately-justified number.** A
player short on games gets `{"eligible": false, "games_played": N,
"min_games": 5}` — no axes at all, not a partial radar with some axes
plotted and others silently missing, which would draw a misleading shape
on the chart. The dashboard reads `games_played`/`min_games` straight from
the payload for its empty-state copy, so the UI's stated reason can never
drift from the actual gate.

**Verified against the real, completed 2025 season (league
`1250182471429931008`), replayed at week 10 the same point-in-time-safe
way Phase 8 Round 2's verification did** (history truncated to weeks < 10,
real model artifact, real Sleeper rosters/matchups) — 391 of 555 candidates
eligible. Five real, well-known players spot-checked by hand, each reading
exactly as their real-world reputation would predict:
- **Christian McCaffrey (RB, SF)**: 98th percentile in Touch Volume, Touch
  Share, Target Share, AND Snap Share — the archetypal 3-down workhorse —
  but only 8th percentile in Yards/Carry, a sane "so much volume his
  per-touch efficiency naturally regresses" story, not a contradiction.
- **James Cook (RB, BUF)**: 95th percentile Touch Volume and Yards/Carry
  (elite volume + elite per-touch efficiency) but only 20th percentile
  Target Share — matches his real profile as a rushing-first back Buffalo
  doesn't feature much in the passing game.
- **Justin Jefferson (WR, MIN) vs. Puka Nacua (WR, LAR)**, both high
  target-share receivers, read as clearly different types: Jefferson —
  90th percentile Air Yards Share, 87th YAC/Reception, only 44th Catch
  Rate (a true downfield alpha whose catch rate is unremarkable because his
  targets are harder); Nacua — 93rd percentile Catch Rate, only 41st aDOT
  (a high-floor, high-catch-rate possession receiver). The radar captures a
  real stylistic difference between two elite-target-share WRs, not just
  "who gets more targets."
- **Travis Kelce (TE, KC)**: only 34th percentile Snap Share (matches his
  real, reported diminished workload in his age-35 2025 season) but still
  78th percentile Red-Zone Target Share and 84th YAC/Reception — still a
  red-zone weapon and dangerous after the catch even in a reduced role.
- **Joe Burrow (QB, CIN)**: correctly `eligible: false` at 4 of 5 games —
  matches his real, injury-shortened 2025 season — the empty state fires
  for exactly the real-world reason it should, not a data artifact.

Also caught a real name-collision in Sleeper's player DB during this spot
check (a linebacker also named "Justin Jefferson," CLE) — correctly absent
from the export entirely (LB isn't a QB/RB/WR/TE candidate position),
confirming candidate filtering works as intended rather than silently
mixing the two players up by name.

Frontend verified separately via Playwright against the same real week-10
export: the eligible case (McCaffrey) renders a real 6-point Chart.js
radar polygon plus a raw-value list, with `Chart.getChart(canvas).data`
matching the Python-computed percentiles exactly; the ineligible case
(Burrow) renders the specific "4 of 5 games played" empty state, not a
generic placeholder. Zero new console errors in either case.

**Radar is deliberately NOT gated on `meta.season` matching the displayed
league's season, unlike `getMyProj()`/the Phase 8 Round 2 simulation
getters.** This was checked against the same audit that found the
win-probability/playoff-odds gap (see the verification-status row on that
fix): radar's own `games_played >= MIN_GAMES_FOR_TREND` gate structurally
can't go eligible until roughly week 5-6 of a season, by which point real
games have already been decided and `fetchAllRealData()`'s
`previous_league_id` fallback (the mechanism that caused the win-probability
gap) has already cleared on its own — there's no realistic window where
radar has real eligible data AND the dashboard is still showing a different
season. This makes radar's un-gated read consistent with Opportunity
Shares/Usage Trending/xFP (its closest siblings, all of which read
`state.advancedStats` the same way), not an oversight.

**Comparison tab's "Profile Overlay" placeholder is unchanged, deliberately
out of scope for this round** — the request was Player Detail's Position
Profile specifically; overlaying multiple players' radars on one chart is
a distinct UI problem (Chart.js's `radar` type supports multiple datasets,
so the data is already there whenever this is picked up) left for later,
not silently dropped.

---

## Phase 5 findings

`src/usage.py`'s `receiving_zone_plays()`/`passing_zone_plays()`/
`rushing_zone_plays()` (real pbp bucketing) and `src/export.py`'s
`build_heatmap_snapshot()` (per-player aggregation into eligible/groups/
zones) — a `heatmap` sibling key alongside `projection`/`usage`/`trend`/
`xfp`/`radar`, wired through the same two callers (`scripts/
weekly_update.py`, `07_export_json.ipynb`) the same way. `index.html`'s
Field Heatmap panel now renders a real SVG field-zone grid instead of the
placeholder card.

**Zones are derived straight from play-by-play, not from any
pre-aggregated feature column** — the one explicit ask this phase's
request called out, and the one place a fabricated PRNG shape used to
live before it was deleted (see the dashboard cleanup pass that removed
~525 lines of dead `getRadarStats`/`renderFieldHeatmap` code). Every zone
count in the export traces to a real `pass_attempt`/`rush_attempt` row.

**Bin definitions are deliberately SEPARATE from xFP's, despite sharing
the same two raw ingredients (air-yards depth, yardline_100 field
position).** xFP's buckets (`TARGET_AIR_YARDS_BINS`/`TARGET_FIELD_POS_BINS`
in `src/usage.py`) are shaped for a league-average RATE ESTIMATE that
needs enough plays per bucket to be reliable across every player at once
(hence xFP's thin-bucket MERGE step, `_TARGET_BUCKET_MERGES`). A heatmap
zone's reliability is about one player's OWN sample, a different problem
solved a different way (the per-zone `sparse` flag, not a global bucket
merge) — see the eligibility section below. New `HEATMAP_DEPTH_BINS`
(behind_los / short / intermediate / deep) and `HEATMAP_FIELD_POS_BINS`
(red_zone / midfield / backfield) exist for exactly this reason: fewer,
display-shaped bands, not xFP's rate-estimation-shaped ones.

**Three zone kinds, one per real usage type, not six axes forced onto
every position the way the radar's axes are picked per position.**
Receivers (WR/TE, and RB's receiving work): depth x field position, same
shape as the xFP target bucket's two ingredients. Runners (RB only —
matching `getHeatmapTitle()`'s already-live "Rushing Direction &
Receiving" title, which doesn't promise QB rushing): direction
(`run_location`: left/middle/right) x field position. QBs: pass location
(`pass_location`: left/middle/right) x depth — matching `getHeatmapTitle()`'s
"Pass Distribution" title, no rushing zones for QBs even though they do
carry the ball, the same scope choice the title already made before this
phase started. `HEATMAP_POSITION_KINDS` in `src/usage.py` is the single
place this mapping lives.

**A real pbp gotcha caught before it reached the export, the same class
already documented for `pass_attempt`:** `pass_attempt==1` fires on sack
rows too (see PROJECT_CONTEXT.md's existing note on pass-attempt-shaped
denominators), which have no real `pass_location`/`air_yards` (verified
directly: 1,287 of 1,367 real-2025 rows missing `pass_location` on a
`pass_attempt==1` frame were sacks, the rest spikes). `passing_zone_plays`
and `receiving_zone_plays` both build from the exact same "real target"
population `_target_play_frame` already established for xFP (a recorded
receiver, not just `pass_attempt==1`), which excludes sacks and spikes
automatically — no separate `sack` filter needed, confirmed with a
dedicated test (`tests/test_heatmap.py::test_passing_zone_plays_groups_by_passer_and_excludes_sacks`)
rather than assumed. `run_location` coverage on real 2025 carries is
99.3% (14,074/14,168) — the small remainder drops rather than guesses a
direction.

**Eligibility reuses radar's exact gate, per the request — one whole-player
decision, not a per-zone one.** `games_played >= MIN_GAMES_FOR_TREND` (the
same Phase 3'-validated floor radar already reuses, not a third
separately-justified number) gates the ENTIRE heatmap: a player short on
games gets `{"eligible": false, "games_played": N, "min_games": 5}`, never
a chart with some zones from a real sample and others from a 1-play
fluke, which would draw a misleading shape.

**Thin zones are shown, not hidden or merged — `HEATMAP_SPARSE_THRESHOLD =
3`, an explicitly-labeled DISPLAY judgment call, not an empirically
derived number like `MIN_GAMES_FOR_TREND` was.** A zone below 3 real plays
this season gets `sparse: true` (a dashed border and a `~` mark in the
UI) rather than being dropped: the play genuinely happened, so hiding it
would understate real (if noisy) usage, but drawing it at full visual
weight would overstate confidence in a 1-2-play sample. This is a
per-PLAYER, per-ZONE decision, deliberately different from xFP's
per-LEAGUE bucket merge — the two solve different reliability problems
(one player's own small sample vs. a league-wide rate table's sample),
so they use different fixes.

**Real bug caught before either reached a committed export, same root
cause in two places (`scripts/weekly_update.py` and
`07_export_json.ipynb`):** `nflreadpy.load_pbp()` raises a bare
`ValueError("Season must be between 1999 and <n>")` for a season that
hasn't started yet — a client-side range check, not the ConnectionError/404
shape `_is_unpublished_season_error` already handles for
`get_weekly_stats`. Running the real weekly-update script locally against
the live pre-draft 2026 league surfaced this immediately (it's the exact
real condition, not a hypothetical): `build_raw_features`'s own
`if weekly_scored.empty: return weekly_scored` early-out means it never
even calls `get_pbp` when nothing's been played yet, so the heatmap step
was the FIRST caller in this run to actually try fetching 2026 pbp. Fixed
with the same architectural choice `build_weekly_scored`'s own tolerance
already made: the "unpublished season is fine, treat as empty" leniency
lives at this ONE caller, not inside `get_pbp` itself, so every other
`get_pbp` caller (which requests seasons known to be published) still
fails loudly on a real bug. `build_heatmap_snapshot` treats a completely
empty `pbp` (zero columns, not just zero rows) as "nothing to zone" before
touching any column, which is provably correct in this exact scenario:
5+ games played is impossible without pbp for those games existing, so
every candidate is already ineligible via the games-played gate regardless
of what the zone functions would have returned.

**Verified against the real, completed 2025 season (league
`1250182471429931008`), replayed point-in-time-safe at week 10 the same
way Phase 4's did.** Real play counts reported before trusting any
number, per the request: 391/555 candidates eligible, real per-zone counts
ranging from single-digit (correctly flagged `sparse`) to 141 real carries
for a workhorse back. Four real players spot-checked specifically for the
requested archetype contrasts, selected from real, already-verified radar
percentiles (highest/lowest aDOT among real 20+-target WRs; highest
goal-line share / highest target share among real RBs) rather than
hand-picked:
- **Tyquan Thornton (WR, KC) vs. Khalil Shakir (WR, BUF)** — a genuine
  deep-threat/possession-receiver contrast. Thornton: 60.7% of his 28 real
  targets in the two "Deep" zones combined (32.1% Deep|Backfield, 28.6%
  Deep|Midfield), nothing in the Red Zone. Shakir: 75%+ of his 49 real
  targets in Short/Behind-LOS zones, a single sparse Deep zone (4.1%, one
  play). The two players' grids light up on opposite sides of the depth
  axis with zero overlap in their dominant zones — exactly the "should
  look different" the request asked to confirm, on real 2025 usage, not
  a designed example.
- **Josh Jacobs (RB, GB) vs. Christian McCaffrey (RB, SF)** — a genuine
  goal-line-back/passing-down-back contrast. Jacobs: 141 real carries
  spread fairly evenly across all 9 direction x field-position cells
  (real red-zone volume: Left/Middle/Right x Red Zone sum to 27.1% of his
  carries) but only 28 real targets, concentrated almost entirely in
  Behind-LOS/Short x Backfield (checkdown-shaped, not real route-running).
  McCaffrey: 168 real carries in a similar rushing shape, but 80 real
  targets spread across 10 different zones including real (if sparse)
  Deep and Intermediate receiving work and real red-zone targets — a
  materially fuller receiving profile than Jacobs' on the same real
  season. Both real backs' RADAR profiles already told part of this story
  (Jacobs 97th percentile Goal-Line Share; McCaffrey 98th percentile
  Target Share) — the heatmap grids make the underlying real usage
  visible rather than just ranked.

Frontend verified via Playwright against the same real week-10 export:
correct panel titles and grid counts per position (`Route Tree` / 1 grid
for WR, `Rushing Direction & Receiving` / 2 grids for RB), the ineligible
state (Joe Burrow, 4 of 5 games) rendering its specific reason rather than
a generic placeholder, and zero new console errors across a full
click-through of every dashboard view. `data/output/player_advanced_stats.json`
regenerated for real against the live 2026 pre-draft league — the same
`ValueError` bug above was caught and fixed via this exact run, not
discovered later.

**Also fixed while here, unrelated to Phase 5 itself but directly visible
in these verification screenshots:** the radar panel's percentile text
used a naive `+ 'th'` suffix (e.g. "3th pctl", "51th pctl"). Added a small
`ordinal()` helper (`index.html`) handling the 11th/12th/13th special
case, used by both the radar axis list and its Chart.js tooltip.

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

## Phase 6.5 findings

### Step 9: game-environment simulation

`src/simulate.py` implements the spec's "option 1" correlation approach: `sample_player_week()`, `simulate_matchup()`, `calibration_report()`.

**Mechanism: a one-factor Gaussian copula.** Every player-week draws a percentile `u = Phi(sqrt(rho)*z_game + sqrt(1-rho)*z_player)`, where `z_game` is ONE shared standard-normal draw per real NFL game (grouped by `game_id`) and `z_player` is that row's own idiosyncratic draw. Two players sharing a `game_id` have percentile correlation exactly `rho`; players in different games are independent — regardless of which *fantasy* team they're on, so two opposing managers who each started a player from the same real game still move together, matching the spec's framing that a shootout lifts everyone in it. `rho = 0.35` is a fixed constant, not fit to data — the spec's own option 2 ("measure the correlations directly") is explicitly deferred; this is option 1, the one the spec says to start with.

Each player's percentile is mapped through *their own* 5 CQR-calibrated quantile points (`pred_q10_cqr`, `pred_q25_cqr`, `pred_q50`, `pred_q75_cqr`, `pred_q90_cqr`) via a piecewise-linear inverse-CDF, linearly extrapolated beyond the 10th/90th (`np.interp`'s default flat-clipping would understate exactly the tail variance a simulation needs most). Building this surfaced a genuine new finding: CQR's 10-90 and 25-75 interval pairs are widened by *different* constants (see above), which can reintroduce quantile crossing *between* pairs (e.g. `pred_q25_cqr` below `pred_q10_cqr`) even though each pair is individually monotonic after its own widening. `sample_player_week()` defensively re-sorts all 5 points per row before building the inverse-CDF — the same rearrangement fix as `fix_quantile_crossing()`, applied again at a new seam it didn't originally cover.

**K/DST and the ~0.5% of skill-position starters without model coverage** use Sleeper's own point projection as a fixed, zero-variance contribution — all 5 quantile columns set to the same value, since interpolating identical points always returns that constant regardless of the percentile drawn. No special-casing was needed in `simulate.py` itself for this; it falls out of feeding it degenerate quantiles. This mirrors the dashboard's existing convention of showing Sleeper's K/DST numbers labeled as Sleeper's, not the model's — consistent with CLAUDE.md's scope boundary, not a new exception.

**Validated on 204 real historical matchups** (2024 Wk5-17 and 2025 Wk1-17 — the locked evaluation window, minus fantasy-playoff bye weeks; 2,837 starters had real model coverage, 816 were K/DST fallbacks, and 18 (~0.5% of all starter-slots) were unexpected missing-skill-coverage fallbacks not investigated further given the size):

- **Win-prediction accuracy essentially ties the naive baseline — expected by construction, not a failure.** Simulation picked the actual winner 61.3% of the time (125/204); naive (whichever team has the higher summed point-estimate projection) picked it 62.7% of the time (128/204). `PHASE_2B_6_SPEC.md`'s original acceptance criterion asked simulation to *beat* naive on this metric; that criterion has since been revised (Aug 2026) because it was testing the wrong thing. Correlation and variance change the *spread* of a simulated total, not its mean, so the simulation's implied favorite (win probability > 50%) matches naive's point-estimate favorite in 191/204 matchups (93.6%) **by construction** — the two methods can only possibly disagree on genuine toss-up games, and every one of the 13 disagreements here has a simulated win probability between 43% and 59%. On those 13 games, naive got 8 right and simulation got 5 — a gap fully explainable by chance at this sample size (13 coin flips), not a reliable difference either way. 204 matchups isn't enough to distinguish the two methods on this axis regardless; that's not what simulation is for here.
- **Calibration is reasonable where there's enough data, untested where there isn't.** Binning both sides of every matchup's probability (408 observations total) into deciles: the six well-populated middle bins (0.2-0.8, 385/408 = 94% of the sample) track the predicted rate within about 2-4 points — e.g. predicted 74.7% in the 0.7-0.8 bin, actual 73.5%. The four extreme bins (0.0-0.2 and 0.8-1.0) show a directional overconfidence pattern — e.g. predicted 93.5%, actual only 50% — but each has just 2-10 observations, nowhere near enough to call this conclusive rather than suggestive.
- **10,000-sim matchups run in a fraction of a second** (see `tests/test_simulate.py`'s timing test), well inside the spec's "seconds, not minutes" bar.

**Bottom line:** the correlation mechanism is implemented and tested as specified. Calibration looks reasonable in the range where there's enough data to judge it. The win-accuracy tie with naive is the expected result, not a shortfall — see the acceptance-criterion revision above — and the sample (204 matchups, 13 disagreements) is too small to resolve anything finer than that either way.

### Step 10: season simulation and playoff-qualification odds

`src/simulate.py::simulate_season()` simulates the *remaining* regular-season schedule and reports each roster's probability of qualifying for the playoffs. Scoped deliberately to qualification, not bracket-round outcomes: which teams make the playoffs is fully decided by regular-season standings (wins, then points_for as tiebreak) *before* the bracket starts, so simulating the bracket itself isn't needed to answer "who makes it" — that would be a separate, later piece of work (championship odds). `src/ingest.py::get_sleeper_bracket()` and `playoff_participants_from_bracket()` pull the real `winners_bracket` and recover exactly which roster_ids made the playoffs in a completed season (every bracket slot without a `t1_from`/`t2_from` reference is a real seed — this correctly finds top-seed byes too, without needing to already know the league's `playoff_teams` count).

Each remaining week's matchups are simulated together in one `sample_player_week()` call across every roster playing that week, so game-environment correlation applies across every real NFL game that week, not just within one fantasy matchup — the same mechanism as step 9, extended across the season. Different weeks are treated as independent of each other.

**Validated on 8 (season, snapshot-week) combinations** — 2024 and 2025, each snapshotted after weeks 5, 8, 10, and 12 (playoff_week_start is 15 for both seasons, so weeks 1-14 are the regular season) — simulating the real remaining schedule with real historical starters and comparing each of the 14 teams' simulated playoff probability against whether they actually made the playoffs (112 nominal observations, `n_sims=3000`, `rho=0.35`):

- **Calibration is reasonable at the extremes, where most of the sample sits, and noisier in the middle, where the per-bin counts are too small to read much into it.** The bottom bin (0-10% predicted, n=26) and top bin (90-100% predicted, n=24) — together 45% of the sample — track almost exactly (3.9% and 100.0% actual against 2.5% and 97.2% predicted). The upper-middle bins (60-90%, n=25) are similarly close. The lower-middle bins (10-60%, n=37) show more scatter — most visibly the 50-60% bin, predicted 53.3% but only 1 of 7 teams (14.3%) actually qualified — but every one of these bins has 5-10 observations, where that much scatter is well within chance (e.g. a truly 53%-calibrated process still lands at ≤1/7 successes about 5% of the time on its own, and that's before accounting for testing 10 bins at once).
- **The nominal n=112 overstates how much independent information this is.** The 4 snapshots per season follow the *same* 14 teams through *one* season's realized outcome — a team headed for the playoffs tends to show a high probability at every snapshot, so these aren't 112 independent trials, closer to 2 (the number of actually-completed seasons). This validation is suggestive of reasonable calibration, not a confident confirmation of it; more completed seasons would be needed for that.
- **Rho sensitivity is smaller than the compounding intuition alone would suggest.** Running every snapshot at `rho = 0.2, 0.35, 0.5`: mean `|P(rho=0.5) − P(rho=0.2)|` across all 112 (season, snapshot, roster) combinations is **0.0065** (two-thirds of a percentage point), with a max of **0.0327** (one roster, one snapshot). This is smaller than "correlation compounds across many weeks" alone would predict, and there's a coherent reason: rho changes the *variance* of each week's score difference, which mostly affects that week's own win probability at the margin — but a season's cumulative win total sums many largely-independent weekly outcomes, and that averaging (central limit theorem) damps how much a single shared correlation parameter can move a season-long summary statistic like a top-7 cutoff. The exception is teams sitting right on the playoff bubble across most of the remaining schedule, which is exactly where the largest movements concentrate (e.g. 2025 roster 4 after week 10: 31.7% at rho=0.2 vs 28.5% at rho=0.5). **Practical takeaway: the untuned rho=0.35 choice is not a fragile assumption for playoff-odds specifically** — a team's probability would need to be reported to a precision finer than about ±2-3 points before this uncertainty would matter, which is finer than this validation's own calibration can currently support anyway.
- Full per-team, per-snapshot, per-rho numbers are in the session's working files, not committed (they're a validation run's output, not pipeline code or documentation).

**Bottom line:** step 10 is implemented and validated the same honest way as step 9 — calibration looks reasonable where the sample supports judging it, with the caveat that 2 completed seasons is a thin basis no matter how the snapshots are sliced, and the untuned game-environment correlation constant turns out to matter much less for playoff-qualification odds than the "correlation compounds" intuition alone would suggest.

---

## Phase 8 findings

Two GitHub Actions workflows, not one — `.github/workflows/retrain.yml`
(`workflow_dispatch` only, never scheduled) and
`.github/workflows/weekly-update.yml` (Tuesdays in-season plus
`workflow_dispatch`) — backed by `scripts/retrain.py` / `scripts/
weekly_update.py` and two new modules, `src/pipeline.py` (shared
fetch-score-feature orchestration, factored out of what
`02_custom_scoring.ipynb`/`03_usage_features.ipynb` already do cell by
cell) and `src/artifacts.py` (model artifact save/load). Both workflows
were verified end-to-end against real cached data BEFORE either one's
first real GitHub Actions run — see the Verification status table.

**The core design problem: "fetch current season only" and "the model
needs last season's data" are in tension, and the resolution is a small
embedded history seed, not a compromise on either.** `add_rolling_features`
needs `prev_season_*` (last season's full-season average) and
`get_export_candidates` needs every player who's ever appeared, for ANY
prior season — neither is available from a bare current-season fetch,
which is all `weekly-update.yml` does (ephemeral runners have no
persistent `data/raw/` to be incremental against, so there's nothing to
avoid re-fetching except by not fetching multi-season history at all
during a weekly run). The fix: `retrain.yml` embeds a `history_seed` in
the artifact — a trimmed slice of the most recent `HISTORY_SEED_SEASONS_BACK`
(2) completed seasons' RAW feature-input columns only (`src/pipeline.py`'s
`HISTORY_SEED_COLUMNS`: `player_id`/`position`/`team`/`season`/`week`/
`custom_points` + every `ROLLING_SOURCE_COLUMNS` entry — NOT the full
~340-column feature table). `weekly_update.py` concatenates this seed with
the current season's freshly-fetched raw features and a target-week stub
row, then reuses `src/export.py::build_target_week_features` COMPLETELY
UNMODIFIED to run `add_context_features`/`add_rolling_features`/
`add_trend_features` over the combined frame — the exact point-in-time
mechanism the original single Phase 7 export already used for a full
8-season table, just with a much smaller "historical" base. Verified
directly: `build_feature_table([2024, 2025])`'s rolling outputs for 2025
match the full `build_feature_table([2018..2025])`'s 2025 rows on every
one of the ~170 rolling columns EXCEPT `xfp`/`fp_over_expected` and their
`_ewm3`/`_s2d`/`_vol` derivatives (see the xFP caveat below) — 2 seasons of
seed is enough for everything that only needs "one season back" (`prev_season_*`)
or "this season's own weeks" (`_ewm3`/`_s2d`/`_vol` for every other feature).

**Two real, disclosed limitations this design accepts rather than hides**
(both are in the exported JSON's `meta.caveats` under weekly-only runs,
`WEEKLY_EXTRA_CAVEATS` in `scripts/weekly_update.py`):
- **The candidate universe and `prev_season_*` are only as fresh as the
  last retrain's `history_seed`.** A player absent from both the seed's 2
  seasons and the current season isn't a projection candidate until the
  next retrain re-seeds with newer history. In practice this only affects
  players who haven't played an NFL snap in 2+ years — a real edge case
  (a comeback after a long injury/retirement absence), not a routine one.
- **`xfp`/`fp_over_expected` run noisier in the first few weeks of a
  season under weekly-only inference.** `add_xfp_features`'s bucket rate
  table is an expanding window over whatever `pbp` it's given (already
  documented as "a two-season compromise, not a permanent design" even in
  the FULL-history case — see the design decisions below); fed only the
  CURRENT season's own plays during weekly inference, an early week's rate
  table has just that week's few hundred plays to average, not the
  multi-season history `retrain.py` trains the model against. This is a
  genuine train/serve skew on a handful of the ~180 model features (xfp's
  RB/WR/TE-only rolled versions), bounded and self-correcting — it
  converges toward normal as the season accumulates its own sample — not
  fixed here, because a real fix means feeding this call multiple prior
  seasons' `pbp` too, exactly the fetch cost "current season only" exists
  to avoid. `src/pipeline.py::build_raw_features`'s docstring has the full
  mechanism.

**`retrain.py` deliberately does NOT re-derive the CQR floor/ceiling
widening constants on every run.** Doing that properly needs walk-forward
QUANTILE predictions across the ENTIRE multi-season history (calibration
= every fold before the eval window, not just the eval window itself) —
the single most expensive computation in this whole pipeline (see the CQR
section above: it required its own dedicated analysis pass). Unlike
point-MAE-vs-baselines, that constant doesn't meaningfully drift retrain
to retrain (no hyperparameter tuning happens anywhere in this pipeline, so
what it's correcting for doesn't change either). `CQR_WIDEN_BY_10_90` in
`src/export.py` is carried into the artifact as-is; recomputing it is a
deliberate, separate, manual step, not something the automated retrain
does.

**Bug caught while wiring this up: `src/pipeline.py::build_weekly_scored`
initially skipped `add_pick_six_column`.** `pass_int_td` is the one active
scoring rule with no weekly-stats column — pick-sixes have to be derived
from `pbp` — and this went unnoticed until `compute_custom_score` printed
its own "Non-zero scoring rules with no column mapping: ['pass_int_td']"
warning on the first real run. Fixed by calling `add_pick_six_column`
before `compute_custom_score`, matching `02_custom_scoring.ipynb`'s own
cell order exactly. Regression-checked directly afterward:
`build_weekly_scored([2025])`'s `custom_points` matches the already-
validated `weekly_scored.parquet` (built by the notebook) to within
floating-point noise (max abs diff ~4e-7) across all 6,037 rows.

**A second, smaller bug caught the same way: `HISTORY_SEED_COLUMNS` listed
`offense_pct` twice** (once explicitly, once already inside
`ROLLING_SOURCE_COLUMNS` via `SNAP_OUTPUT_COLUMNS`). Selecting it produced
a DataFrame with two identically-named columns, which doesn't fail at
selection time — it fails much later, inside `pd.concat`, with a confusing
`"Reindexing only valid with uniquely valued Index objects"` error.
`dict.fromkeys(...)` dedupes while preserving order; a regression test
(`tests/test_pipeline.py::test_history_seed_columns_has_no_duplicates`)
locks this down directly rather than trusting it by inspection.

**Local end-to-end verification, before either workflow's first real
GitHub Actions run:**
- `python scripts/retrain.py` against warm local caches (`data/raw/`
  already had all 8 seasons cached) completed cleanly: 45,693-row x
  346-column feature table, walk-forward performance over the 2024-2025
  eval window closely matches the already-published Phase 6 numbers (QB
  6.36 vs. the published 6.38 model MAE; RB 4.17 vs. 4.18; WR 3.95 vs.
  3.93; TE 3.03 vs. 3.06 — small differences expected, since this script's
  baseline-comparison convention differs slightly from the original
  hand-run analysis's, not because anything regressed), and wrote a
  **7.57 MB** artifact (`models/fanteasy_model.joblib`).
- `python scripts/weekly_update.py` against that artifact, for the real
  live 2026 league (`pre_draft`, zero games played): correctly detected
  target week 1 from the real published schedule, correctly treated the
  current season's stats/pbp fetch coming back HTTP 404 (nothing published
  for an unstarted season yet — a real, expected condition, distinguished
  from a genuine fetch failure via `src/pipeline.py::_is_unpublished_season_error`
  checking for a 404 specifically, not swallowing connection errors
  generally) as zero current-season rows rather than crashing, and
  produced a 300-player, 220,020-byte JSON — matching the shape of the
  already-committed Phase 7 export (219,330 bytes, 300 players) almost
  exactly.
- Full test suite (97 tests: the original 87 plus 10 new — `tests/
  test_artifacts.py`, `tests/test_pipeline.py`, and additions to `tests/
  test_model.py` for `train_final_models`/`predict_with_models`) passes.

**Artifact size is not a git-bloat concern at `retrain.yml`'s frequency.**
7.57 MB, committed only on a manual, infrequent `workflow_dispatch` — not
on every `weekly-update.yml` run, which never touches `models/`. A few
retrains a year adding ~7-8 MB each is a trivial cost to git history;
revisit only if retrain frequency ever becomes routine/scheduled (it
deliberately isn't).

**First real GitHub Actions runs: `retrain.yml` succeeded outright,
`weekly-update.yml` caught a genuine flakiness gap in the fetch layer.**
`retrain.yml`'s first real `workflow_dispatch` run completed in ~5 minutes
and committed the 7.57 MB artifact (`models/fanteasy_model.joblib`,
`model_version` = the commit that fixed `test_no_leakage.py`'s local-cache
dependency, see below) — its walk-forward performance numbers match the
local run almost exactly (QB/RB/WR model MAE identical to two decimal
places; TE `sleeper_mae` 2.85 vs. 2.86, expected variance from a fresh
Sleeper API snapshot). `weekly-update.yml`'s first run then failed at
`pytest` with 45 `ConnectionError`s, all tracing to ONE download —
`stats_player_week_2024.parquet` from nflverse-data's GitHub-releases
CDN resetting mid-transfer — not a code bug (the identical test suite had
just passed inside `retrain.yml` minutes earlier; a `scope="module"`
fixture meant one failed fetch cascaded into 45 dependent test failures in
`tests/test_no_leakage.py`). Since this same flakiness risk sits in the
PRODUCTION fetchers too, not just the test's fixture, and `weekly-
update.yml` runs unattended on a cron with nobody there to click retry:
added `src/ingest.py::_retry_transient()` (3 attempts, exponential
backoff starting at 2s) around every `nflreadpy` `load_*` call site.
Deliberately does NOT retry an HTTP 404 — that's `_is_unpublished_season_error`'s
"this season genuinely isn't published yet" signal, and retrying a
resource that doesn't exist just delays the same failure. Covered by
`tests/test_ingest.py` (4 tests: retry-then-succeed, exhausts-then-raises,
never-retries-a-404, args/kwargs pass through cleanly).

`weekly-update.yml`'s SECOND real run then hit a different, unrelated bug:
its commit step ran `git pull --rebase` BEFORE staging/committing
`scripts/weekly_update.py`'s own output, so rebase refused to run against
a dirty working tree ("cannot pull with rebase: You have unstaged
changes"). `retrain.yml` had the identical latent bug in the same
copy-pasted three-line pattern — it only didn't trigger on retrain's first
run because `models/fanteasy_model.joblib` was still untracked that time
(an untracked file doesn't block rebase the way a modified TRACKED one
does). Fixed in both workflows by reordering to commit first, then
`git pull --rebase`, then push.

**Both workflows are now verified successful end to end on real GitHub
Actions infrastructure, not just locally.** `weekly-update.yml`'s third
run completed in ~2 minutes and committed a real, correct
`data/output/player_advanced_stats.json`: `meta.model_version` is
`a011dc8` (the ARTIFACT's own trained-model commit, not the workflow run's
own HEAD — confirms the "model stays fixed between retrains" design is
actually holding in production), `meta.performance` matches the artifact's
walk-forward numbers exactly, `meta.caveats` includes both the base
caveats and the two weekly-only ones, 300 players at 220,002 bytes
(matching the local dry run's 220,020 almost exactly), and a spot-checked
player's `usage`/`trend` blocks show the expected pattern for week 1 of a
new season — every in-season `_ewm3` value null, `prev_season_*` populated
from real 2025 data, `floor <= point <= ceiling` holding.

### Round 2: wiring src/simulate.py into the export (win probability + playoff odds)

`src/simulate.py` (matchup win probability, playoff-qualification odds)
was computed and validated back in Phase 6.5 but never written into
`player_advanced_stats.json` — the dashboard had no way to show it. Round
2 adds a `simulation` block (sibling to `meta`/`players`, not per-player)
via `src/export.py`'s new `build_team_game_id_lookup`/
`build_starter_quantile_rows`/`build_matchup_simulation`/
`build_playoff_odds`/`assemble_simulation_block`/`validate_simulation`,
orchestrated by `scripts/weekly_update.py::build_simulation_block`, and
surfaced in `index.html` as win probability on matchup cards (Dashboard +
Matchups view) and a Playoff Odds column in the standings table.

**The blocking problem: the simulator needs 5 calibrated quantile points
per player, and the Phase 8 artifact only ever trained 2.** `sample_player_week`
needs `pred_q10_cqr`/`pred_q25_cqr`/`pred_q50`/`pred_q75_cqr`/`pred_q90_cqr`
— both the 10-90 AND 25-75 CQR-widened interval pairs, not just the 10-90
pair `predict_target_week`'s floor/ceiling already used. Fixed by:
extending `train_final_models`'s `quantile_alphas` to the full
`SIMULATION_QUANTILE_ALPHAS = (0.10, 0.25, 0.50, 0.75, 0.90)` (backward
compatible — `predict_with_models`'s floor/ceiling logic only ever reads
the 0.10/0.90 keys out of whatever's trained, so it doesn't care that
more keys now exist); adding `src/model.py::predict_quantiles_with_models`
as the parallel prediction path (mirrors `predict_with_models`'s shape,
same per-position loop, same artifact); and deriving
`CQR_WIDEN_BY_25_75 = {"QB": 1.265, "RB": 0.465, "WR": 0.380, "TE": 0.355}`
in `src/export.py` from the ALREADY-PUBLISHED Phase 6 CQR before/after-width
table above (`widen_by = (width_after − width_before) / 2`) rather than
re-running the expensive calibration pass — verified by reproducing the
already-known 10-90 constants the same way first (matched to within the
source table's own 2-decimal rounding) before trusting the method on the
25-75 row. `scripts/retrain.py` now trains the full 5-quantile set and the
artifact carries both CQR dicts; the local artifact grew from 7.57 MB to
**11.16 MB** as a result (still a manual, infrequent commit — not a
git-bloat concern at that cadence).

**Predicting a real matchup needs a real starting lineup, and Sleeper's
`starters` field is Sleeper IDs, K/DST included, with no model coverage
guarantee.** `build_starter_quantile_rows` resolves each starter to either
its own 5 model-predicted quantile points (QB/RB/WR/TE with crosswalk +
candidate coverage) or Sleeper's own point projection as a fixed,
zero-variance fallback (K/DST, or a skill player missing coverage) —
`sleeper_projected_points()` scored via the league's real
`scoring_settings`, same convention `src/simulate.py`'s own module
docstring documents. A starter on a bye (no resolvable `game_id` for their
team that week) is dropped, not zeroed — there's no real game to attach
them to.

**Playoff odds need every remaining week's matchups predicted, not just
the upcoming one — and that reuses `build_target_week_features` completely
unmodified.** Nothing new has actually happened between the upcoming week
and a later one either (both are unplayed), so Family 6's rolling/
`prev_season_*` features come out IDENTICAL across every remaining week's
stub-row prediction; only Family 5's per-week schedule context (opponent,
home/away, spread) legitimately differs, and `build_target_week_features`
already recomputes that fresh per call. Two disclosed simplifications this
accepts rather than hides: (1) each remaining week's starters are whatever
`get_sleeper_matchups` reports for that week — for a real forward-looking
run that's effectively "today's lineup, held constant" (Sleeper carries
the current default forward until a manager changes it; there's no honest
way to predict a future lineup change), while a *historical* week
(verification below) gets that week's own real, different lineup, since
the season already happened; (2) K/DST/uncovered players contribute a
fixed Sleeper-projection amount with zero simulated variance.

**Two real bugs caught during this build, both before they reached a
committed export:**
1. `predict_quantiles_with_models` was initially called on
   `build_target_week_features`'s FULL combined output (every historical
   row plus the one stub week), not just that week's stub rows — with
   `build_starter_quantile_rows`'s later `drop_duplicates(subset=["player_id"])`
   silently keeping an ARBITRARY past week's prediction per player instead
   of the intended future one. Fixed by applying the same
   `(season == X) & (week == week)` mask `predict_target_week_from_artifact`
   already uses for the single-target-week path, inside the per-remaining-week
   `week_quantiles()` closure.
2. `build_simulation_block` hardcoded the module-level `DEFAULT_LEAGUE_ID`
   inside two `get_sleeper_matchups` calls instead of taking a `league_id`
   parameter — invisible in the real flow (weekly_update.py always predicts
   the live league anyway) but made the function impossible to verify
   against a different, real historical league, and surfaced immediately
   as a silently-empty simulation the moment verification tried to point it
   at 2025's real league instead of 2026's pre-draft one. Fixed by adding
   `league_id` as an explicit parameter, threaded through from `main()`.

**Verified against a real, completed 2025 week (Week 10, same week Round
1 used) before touching the live 2026 export**, using the REAL 2025
Sleeper league (`1250182471429931008` — historical league data stays
fetchable after a new league_id is minted for the following season): 7
matchups simulated for a 14-team league, every pair of win probabilities
summing to exactly 100 (46/54, 33/67, 54/46, 57/43, 98/2, 42/58, 75/25),
and playoff odds for all 14 rosters (0-100%, sane spread given each
team's week-10 record). Frontend verified by temporarily pointing a local
copy of `index.html` at the real 2025 league (never committed) so the
dashboard had real matchup/roster data whose roster_ids matched the test
export — Playwright + a UTF-8-aware local server confirmed win-probability
badges on matchup cards (Dashboard and Matchups view) and the Playoff Odds
standings column, each with the mandatory calibration/accuracy caveat
visibly captioned (not just tooltip-only) beneath the panel, whole-percent
rounding throughout, and zero new console errors. The null-safe path was
verified separately against the real, pre-simulation-key 2026 export (no
`simulation` key at all, a stricter test than an explicit `null`) —
matchup cards render with no win-probability badges or caveat caption, and
the standings table keeps its Playoff Odds column with an honest `—` per
row rather than hiding the column or crashing.

**Honesty requirements from the request, and where each one actually
lives, not just where it's supposed to:** whole-percent rounding happens
once, in `src/export.py`'s `build_matchup_simulation`/`build_playoff_odds`
(`round(x * 100)`, Python's `round()` returning a plain `int`), so no UI
code can accidentally reintroduce a decimal. The calibration caveat
(~2 independent seasons, not certified) and the accuracy caveat (93.6%
agreement with "higher projection wins" is by construction, not
out-picking) are both stored in the export itself
(`calibration_caveat`/`accuracy_caveat`) and read from there by
`index.html`'s `simulationCaveatText()` — with a hardcoded fallback of the
identical wording only for the case where `state.advancedStats` predates
this key entirely, so the caveat text has exactly one source of truth,
not two copies that could drift.

### Round 3: Player Detail enrichment (My Proj, Volatility, weekly xFP, trend indicator)

Four additions to Player Detail, all reusing `.injury-stat-card`/
`.injury-stat-grid`/`.panel` — no new CSS patterns:

- **KPI grid: 4 → 6 cards**, not more (My Proj + Volatility added
  alongside the existing Season Pts/Avg-Game/Sleeper Proj/Best Week). The
  grid stays `.injury-stat-grid` with an inline `grid-template-columns:
  repeat(3, 1fr)` override for this one 6-card instance — the injury tab's
  own 4-card usage of the same class is untouched.
- **My Proj** reuses `getMyProj()` (already existed, previously only used
  on the Players/Comparison tabs) — point plus a floor–ceiling range,
  gated on the export's `meta.season`/`week` matching what's being viewed,
  same rule every other "My Proj" surface already follows.
- **Volatility** exports `xfp_vol` (Family 6's expanding std of xFP) for
  the first time — it rode `USAGE_EXPORT_COLUMNS` like every other
  `usage.*` field, so no new top-level export key was needed, just one
  more entry in an existing list. Chosen deliberately over a raw
  usage-share volatility: no column in this pipeline is fantasy-POINTS
  volatility itself (Family 6 excludes `custom_points`, the model's own
  target, from `ROLLING_SOURCE_COLUMNS` by design — see `src/model.py`'s
  module docstring), and xFP already blends targets, carries, and field
  position into one points-scale number, making its volatility the
  closest already-computed proxy to "boom/bust in points" this pipeline
  has. Shown as a plain points figure ("4.5 pts, week-to-week swing"), not
  normalized into a 0-100 score, per the request's "label it plainly."
  Null for QB by construction, same disclosed gap as every other
  xFP-derived field.
- **Weekly xFP chart line** needed a genuinely new kind of export field:
  everything else in `player_advanced_stats.json` is a single
  upcoming-week snapshot, but a chart line needs one value per
  ALREADY-PLAYED week. `src/export.py::build_weekly_xfp` scopes to
  `target_season`'s played weeks only (the Weekly Production chart only
  ever plots one season's bars), drops null xfp (QB; any real gap) rather
  than zero-filling, and `assemble_player_advanced_stats` groups the
  result into `{player_id: {week_str: xfp}}` BEFORE its per-row loop —
  unlike `usage`/`trend`/`xfp_summary`, this can't be a plain `.merge()`,
  since it's one row per (player, week) rather than one row per player.
  Added defensively at the grouping step too (`.dropna(subset=["xfp"])`
  before grouping, not just trusting `build_weekly_xfp`'s own contract) —
  JSON has no `NaN`, and a caller passing an unfiltered frame shouldn't be
  able to leak one through. On the chart itself: a new dataset, same
  `hidden: true` off-by-default convention as the existing Top-N Position
  Avg line, distinct color/dash (`#059669` green, `[2,2]` dotted) so it
  reads as its own thing next to Sleeper Proj (solid orange) and Top-N
  (dashed purple).
- **Trend indicator** reuses the SAME per-player "biggest signal"
  selection the Dashboard's Usage Trending panel (Round 1) already used —
  refactored out of that panel's inline loop into a shared
  `getPlayerTrendHeadline()` so the leaderboard and this compact,
  one-line indicator can never disagree about which feature is "the"
  trend for a given player. One line above the Opportunity Shares grid
  (direction arrow + feature name), not its own panel, per the request.

**Verified against the real, completed 2025 season (same league,
`1250182471429931008`) before touching the live 2026 export, at TWO
target weeks specifically to exercise both the "export's week matches
what's live" and "it doesn't" paths:**
- **Week 10** (weeks 1-9 real, matching Round 1/2's own test week):
  `corr(actual, xfp)` = **0.788** across 2,668 real (player, week) rows —
  strong positive, not 1.0, exactly the expected signature of a signal
  that's opportunity-driven but not outcome-driven. Individual gaps read
  sensibly by hand: Zach Ertz Wk 2 (actual 15.4, xfp 8.9, gap +6.5 — a big
  TD game beating opportunity) and Wk 8 (actual 3.6, xfp 6.2, gap −2.6 —
  real opportunity that didn't convert).
- **Week 17** (weeks 1-16 real, chosen specifically to match the live
  browser's own detected "current week" so `getMyProj()`'s week-matching
  gate would actually pass): same 0.788 correlation held at the larger
  4,748-row sample. Jonathan Taylor's real Wk 10 explosion (47.1 actual
  points) shows an xFP of 24.64 that week — the single clearest "luck, at
  a glance" moment in the whole verification: a real 22-point gap between
  the bar and the line, on a week that really was a historically lucky
  outing relative to his own opportunity level that game.
- **Frontend**: Playwright confirmed all 6 KPI cards, the trend indicator,
  and the chart's new dataset render correctly in both the populated case
  (My Proj "14.9 / 7.5–25.2 range", Volatility "4.5 pts", trend "↑ Carry
  Share rising") and a null case (the pre-Round-2/3 export, which has
  neither `simulation` NOR `weekly_xfp` NOR `usage.xfp_vol` at all — a
  stricter test than an explicit `null` for any of the three) — every new
  element degraded to its designed empty state (`—`, "Not available", "No
  usage trend yet this season", the xFP legend entry present but simply
  empty of data) with zero new console errors. One tooling note, not a
  product bug: Chart.js renders its legend on `<canvas>`, so Playwright's
  text-based click can't toggle it in headless testing — verified the
  `hidden: true` default and the dataset's actual per-week values instead,
  via `Chart.getChart(canvas).data.datasets`, and used the same API to
  force the line visible for the confirming screenshot.

---

## Phase 9 findings — season archives + selector

**The problem this fixes, stated the way it was found:** on a Player Detail
page, the 6 KPI cards read LIVE Sleeper data for whichever season the
dashboard happened to be showing (a full 16-17-game 2025 season, via the
old `previous_league_id` fallback), while the radar/heatmap/opportunity
shares read `state.advancedStats`, which only ever held ONE export at a
time — the live 2026 pre-draft export, `games_played: 0` for everyone.
Both halves of the page were individually correct and honest; shown
together, a full season next to "0 of 5 games played" reads as broken.
The fix isn't a smarter gate on either side — it's making "which season"
one explicit choice that both sides answer from, instead of two
independent mechanisms that happened to agree only by accident.

### Feasibility, answered before scope was decided

**This league's own Sleeper history, walked live from `DEFAULT_LEAGUE_ID`'s
`previous_league_id` chain, not assumed:** 2021, 2022, 2023, 2024, 2025 all
report `status: "complete"`; 2026 is `pre_draft`. Five real, complete
seasons exist to archive. nflverse data (pbp/weekly/ngs/snaps/schedule) is
already cached for all of 2018-2025 (the model's own training window), so
raw-data coverage was never the constraint for any of these five.

**The real constraint was a live bug in candidate selection, not data
availability, and it gets worse the further back you go.**
`get_export_candidates()` requires a player to have a CURRENT team on
file in TODAY's live Sleeper player DB — correct for the live weekly
export (a free agent with no team has no upcoming game to attach context
to), completely wrong for an archive (a 2021 season's real rostered
player who has since retired still played real 2021 games; "no current
team" just means they're no longer active TODAY, which has nothing to do
with whether 2021 is archivable). Checked directly against this league's
real historical rosters before deciding anything, not assumed: reusing
`get_export_candidates()` unmodified would have silently dropped

| Season | Real rostered players | Would survive unmodified `get_export_candidates` |
|---|---|---|
| 2021 | 179 | 84 (46.9%) |
| 2022 | 179 | 107 (59.8%) |
| 2023 | 182 | 125 (68.7%) |
| 2024 | 177 | 143 (80.8%) |
| 2025 | 179 | 159 (88.8%) |

— i.e. even the MOST RECENT completed season would have silently lost 1
in 9 of its real rostered players, and the oldest would have lost more
than half. This is why archiving isn't "call the existing export
functions with a different season" — a real fix was needed:
`get_archive_candidates()` (`src/export.py`) resolves `team` from a
player's own LAST REAL historical row (always available for anyone who
actually played) instead of today's live Sleeper snapshot, dropping the
current-team requirement entirely — a stub row for a week that already
happened (or in an archive's case, a week that never will) doesn't need a
CURRENT team, `add_context_features` finds no real schedule entry for it
either way, so the value only needs to be non-null. With this fix,
`n_with_current_team == n_crosswalk_matched` for every one of the 5
seasons checked — nobody is dropped for a reason that shouldn't apply to
a historical archive.

**File size is real and larger than a first guess based on the current
committed export would suggest — measured, not assumed.** The live
`player_advanced_stats.json` is currently ~250-270 KB, but that reflects
the 2026 PRE-DRAFT export, where radar/heatmap are null for everyone
(nothing played yet). A populated export — real radar percentiles, real
heatmap zones, more real candidates surviving the fixed team filter — is
much bigger: the real, committed 2025 archive is **921,516 bytes (900
KB)**, and an in-season LIVE export will grow to roughly the same size
once 2026's players clear the games-played floor. At that real size, 5
seasons (2021-2025) would be ~4.5 MB total — still trivially cheap for
git (the model artifact alone is already 7.57-11.16 MB, committed
routinely, see Phase 8 findings), just a materially different number than
a 200 KB-per-season guess would suggest.

**Bottom line: all 5 completed seasons (2021-2025) are technically
feasible now, but only 2025 is actually shipped.** `src/ingest.py`'s
`SEASON_LEAGUE_IDS` lists all 5 real league_ids (so the infrastructure
doesn't need to change to add more), but `index.html`'s `SEASON_OPTIONS`
and the committed `data/output/archive/` directory currently carry only
2025, matching the explicit "2025 to start" scope of the request. Any of
2021-2024 can be added later by running `python scripts/archive_season.py
<year>` and adding one entry to `SEASON_OPTIONS` — no code changes needed
beyond that.

### Archive generation (`scripts/archive_season.py`)

Reuses the exact point-in-time-safe machinery `scripts/weekly_update.py`
already established for a live week — a stub row for `target_week`,
`add_context_features`/`add_rolling_features` re-run over the combined
frame — with `target_week` set to ONE PAST that season's real final REG
week (`determine_archive_target_week()`, the same "one past the last
completed week" logic `weekly_update.py`'s `determine_target_week()`
already uses, just applied to a frozen season instead of a live one).
This is why "full-season aggregates" and "point-in-time-safe" aren't in
tension: historical_features covers every real week of the season
(1 through the final week), the stub row is for a week that never
happened, and there's nothing at or after it to leak from BY
CONSTRUCTION — not a new leakage argument, the identical one
`weekly_update.py`'s live path already relies on every single week.

Reads the full cached `weekly_features.parquet` (2018-2025) directly
rather than `weekly_update.py`'s constrained `history_seed` approach —
this is a manual, occasional script, not a CI job under a fetch-cost
budget, and the full history is already on disk. Uses the already-
committed model artifact (same one `weekly_update.py` uses) rather than
training a fresh one — an archive is a snapshot with today's best model,
not a retrain. `simulation` is always `null` — win probability for a
week that never happened isn't a real thing to compute, so this script
doesn't try building it at all.

`projection.point/floor/ceiling` in an archive describes a hypothetical
week after the season ended, not a real number anyone should act on —
disclosed via `ARCHIVE_EXTRA_CAVEATS`, appended to the same `meta.caveats`
list every other caveat already lives in, not a new mechanism.

Generated for real, not just tested against synthetic data: `python
scripts/archive_season.py 2025` produced `data/output/archive/2025.json`
(921,516 bytes, 465 players, 466/1468 radar/heatmap-eligible candidates,
crosswalk match rate 99.0%) — committed alongside this work.

### Season selector (`index.html`)

`SEASON_OPTIONS` (near `LEAGUE_ID`) is a small, hand-maintained array —
`{season, leagueId, live, exportPath}` — one entry per season this SAME
hard-coded league has existed under, extended by hand the same way
`LEAGUE_ID`/`SEASON_LEAGUE_IDS` already are. This is NOT the runtime
league-switcher CLAUDE.md's non-negotiables forbid: every entry is still
this one league, just a different season of it — there's no way to type
in an arbitrary league_id.

**Replaces `previous_league_id` fallback outright, not alongside it.**
`fetchAllRealData()` now takes a `season` argument and looks up that
season's real `leagueId`/`exportPath` from `SEASON_OPTIONS` — no
`previous_league_id` fetch, no silent swap. `determineDefaultSeason()` is
the one place "does the live season have real games yet" still gets
checked (the same `hasGames` logic the old fallback used), now to pick an
honest DEFAULT selection rather than to silently substitute which
league's data loads — the user can always override it via the selector.

**Switching a season switches everything together, including things that
were previously silent, persistent caches keyed only by week number.**
`state.statsByWeek`/`state.projectionsByWeek` (Sleeper per-week stats/
projections, used by Player Detail's KPI cards and game log) were keyed
by week number ALONE, never season — safe when only one season was ever
loaded per page load, a real cross-season contamination bug the moment a
switch became possible (2025's week-1 stats would silently answer for
2026's week 1, both cached under the key `"1"`). Caught before it shipped
by reasoning through "what does 'no mixed state' actually require,"
not just testing the happy path — fixed by clearing both caches (and
their in-flight-fetch dedupe sets) on every season load, in the one
function (`loadSeason()`) both `init()` and the selector's `switchSeason()`
funnel through. `switchSeason()` also resets every active detail-view ID
(`activeMatchupId`/`activeTeamId`/`activePlayerId`/`activeNflGameId`) --
"no mixed-season pages" applies to navigation state, not just data, since
a stale roster_id or player_id from the old season's data could otherwise
resolve to a different real entity (or silently nothing) under the new
season.

**The NFL sidebar is season-gated too, not just fetched-and-ignored.**
ESPN's scoreboard endpoint always returns TODAY's games regardless of what
season is asked for — there's no honest way to scope it to an archived
season, so `fetchAllRealData()` skips the ESPN fetch entirely when
`!seasonOption.live`, and `renderSidebar()` shows an explicit "Archived
season — live NFL scores aren't shown" message rather than an empty
"no games" state that would read as a fetch failure.

**Banner reworded, not just re-purposed.** `#season-banner` (renamed from
`#season-fallback-banner`) now states plainly, for an archived season,
that it's a completed-season archive — real final standings, nothing
live. For the live season with no real games yet, it says so without the
old "hasn't drafted yet, so we're showing you something else" framing,
since the selector itself already makes clear which season is being
viewed; there's nothing left for the banner to explain away.

**Verified exactly the way the request asked — 2025 switched-to, then
switched away from, both checked against every panel the request named:**
- **On 2025 (real archive):** page defaulted to 2025 automatically (2026
  has zero real games); banner correctly reads "completed-season
  archive"; Christian McCaffrey's Player Detail showed Season Pts 354.9 /
  Avg 22.2 / Best Week 35.6 (real full 2025 season) alongside a FULLY
  ELIGIBLE radar ("Percentile vs. 33 startable RBs") and heatmap (311 real
  rushing plays, 129 real receiving plays) and real Opportunity Shares
  (Snap 83.0% / Target 20.9% / Carry 64.8% / Red-Zone 62.0%) and a Weekly
  Production chart with 17 real bars (weeks 1-17) — every panel the
  request named reporting the same real season, no partial/mixed state
  anywhere.
- **Switched to 2026 (live, pre-draft):** standings table correctly went
  to 0 rows (no real rosters yet); McCaffrey's Player Detail correctly
  showed Season Pts 0.0 / Best Week 0.0, radar and heatmap BOTH correctly
  reverted to "Not enough games yet (0 of 5)" — not a trace of 2025's real
  354.9-point season leaking through. Sleeper Proj (17.1) and My Proj
  (19.0) correctly still populated, since those are legitimately about
  the live upcoming week, not the completed season.
- Confirmed via Playwright against the real production config (unmodified
  `LEAGUE_ID`, real committed 2025 archive), including a mid-navigation
  season switch (while on the Draft view) and a full click-through of
  every tab in both seasons. Zero new console errors throughout (the only
  noise was the already-established ESPN CORS artifact of local
  off-origin serving).

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
| Season/week boundary handling | **Partly verified** — requesting a season nflverse hasn't published (e.g. 2026 in the offseason) still 404s with a raw traceback in `src/ingest.py` generally. `src/pipeline.py::build_weekly_scored` now catches exactly this case (verified live: `scripts/weekly_update.py` hit the real 2026 stats_player 404 and correctly degraded to zero current-season rows instead of crashing) — narrowly, for Phase 8's single-current-season caller only, not a general `ingest.py` fix. |
| xFP (`add_xfp_features()`) reproduces `custom_points` on real plays | **Verified** — per-play scores summed over a player's actual weekly plays matched `custom_points` exactly on 6 of 8 spot-checked player-weeks (2025 Wk10 WR/RB sample); the other 2 differed by exactly −2.00, fully explained by the deliberate two-point-conversion exclusion (see design decisions above) — the only known discrepancy. Caught and fixed one real bug in the process: an early version silently added `0.04 × passing_yards` to every target's score because raw pbp's own play-level `passing_yards` column leaked into the synthetic play frame. |
| `src/simulate.py` matchup win probability is calibrated in the populated range | **Verified** — tested against 204 real Sleeper matchups (2024 Wk5-17, 2025 Wk1-17). The well-populated 0.2-0.8 probability decile bins (385/408 observations) track the actual rate within ~2-4 points; simulation and naive point-estimate comparison agree on the favorite in 191/204 matchups (93.6%, expected by construction — correlation/variance affect a total's spread, not its mean), so the two are not distinguishable on win-pick accuracy at this sample size, which isn't what this check was revised to test for. See **Phase 6.5 findings**. |
| `simulate_season()` playoff-qualification probability is calibrated | **Partly verified** — tested on 8 (season, snapshot-week) combinations across the 2 completed seasons available (2024, 2025), 112 nominal team-observations. The best-populated bins (0-10% and 90-100% predicted, 45% of the sample) track the actual rate almost exactly; the middle bins scatter more but every one has only 5-10 observations, where that's expected noise, not a demonstrated bias. The 4 snapshots per season aren't independent of each other (same 14 teams, one realized season), so the true independent sample size is closer to 2 than 112 — suggestive, not a confident confirmation. See **Phase 6.5 findings**. |
| `rho=0.35` sensitivity for season-long playoff odds | **Verified small** — re-ran every validation snapshot at rho ∈ {0.2, 0.35, 0.5}; mean \|P(rho=0.5) − P(rho=0.2)\| across all 112 (season, snapshot, roster) combinations is 0.0065, max 0.0327. Smaller than the "correlation compounds across weeks" intuition alone would suggest — a season's cumulative win total averages many largely-independent weekly outcomes, which damps how much one shared weekly correlation parameter can move a season-long summary statistic, except for teams sitting on the playoff bubble across most of the remaining schedule. See **Phase 6.5 findings**. |
| Phase 3' trend window (3 vs. 4 vs. 5-week EWM half-life) | **Verified** — 3 beats 4 and 5 on both hold-rate and correlation with next-week usage, for every one of target_share/carry_share/offense_pct/rz_opportunity_share, over 21,000+ real player-weeks per feature (2018-2025). See **Phase 3' findings**. |
| Phase 3' trend leakage (`add_trend_features`) | **Verified** — future-truncation and same-week-perturbation tests both pass (`tests/test_no_leakage.py`), same two-pronged pattern used for Family 6's rolling aggregates. |
| `scripts/retrain.py` runs end-to-end and matches published Phase 6 numbers | **Verified** — locally against warm `data/raw/` caches, AND on real GitHub Actions infrastructure (`workflow_dispatch`, cold runner, ~5 min, committed a 7.57 MB artifact whose walk-forward performance numbers match the local run). See **Phase 8 findings**. |
| `scripts/weekly_update.py` runs end-to-end for a real pre-season week | **Verified** — locally, AND on real GitHub Actions infrastructure (third `workflow_dispatch` attempt succeeded after two real bugs found and fixed by the first two attempts — a test-fixture network dependency and a git commit-step ordering bug, both now fixed for future runs too). Correctly detected target week 1, correctly handled the real "2026 stats not published yet" 404, committed a 300-player, 220,002-byte JSON with `meta.model_version` correctly pinned to the artifact's own trained commit. See **Phase 8 findings**. |
| `history_seed` (2 seasons) is sufficient for Family 6 rolling/prev_season_* correctness | **Verified** — `build_feature_table([2024, 2025])`'s 2025 rolling outputs match the full 8-season build's 2025 rows exactly on every `ROLLING_OUTPUT_COLUMNS` entry except `xfp`/`fp_over_expected` and their derivatives (a disclosed, separate, self-correcting gap — see **Phase 8 findings**). |
| `CQR_WIDEN_BY_25_75` derivation method (Phase 8 round 2) | **Verified** — reproducing the already-known, already-published `CQR_WIDEN_BY_10_90` constants from the Phase 6 CQR table's before/after-width columns via `widen_by = (width_after − width_before) / 2` matches the hardcoded values (QB 2.309, RB 0.730, WR 0.606, TE 0.467) to within the source table's own 2-decimal rounding, before trusting the same method on the 25-75 row (never independently re-run). |
| `scripts/weekly_update.py::build_simulation_block` produces correct win probabilities and playoff odds | **Verified** — against the real, completed 2025 season (league `1250182471429931008`, week 10): 7 real matchups simulated for a 14-team league, every pair of win probabilities summing to exactly 100, playoff odds returned for all 14 rosters. See **Phase 8 findings** for the two real bugs this caught before either reached a committed export. Not yet run on real GitHub Actions infrastructure — that's the next `retrain.yml` + `weekly-update.yml` `workflow_dispatch` pair, still to happen. |
| Round 2 dashboard panels (win probability, playoff odds) render correctly and null-safely | **Verified** — Playwright + a UTF-8-aware local server, both against the populated case (index.html temporarily pointed at the real 2025 league so matchup roster_ids matched the test export — never committed) and the null case (the real, pre-simulation-key 2026 export, a stricter test than an explicit `null`). Zero new console errors either way; see **Phase 8 findings** for what each screenshot showed. |
| `getMatchupWinProb()`/`getPlayoffOdds()` gate on `meta.season` matching the displayed league's season, not just week | **Verified — a real gap found in a full-dashboard audit, since fixed.** Neither getter checked `meta.season` (`getPlayoffOdds` checked nothing at all — not season, not week — just whether a `playoff_odds` entry existed for the roster id). This mattered specifically because of `fetchAllRealData()`'s `previous_league_id` fallback: right after a new season's draft, before any of that season's games are played, the dashboard still displays the PRIOR season (`hasGames` still false) while `weekly-update.yml` may already have written a `simulation` block for the NEW season's week 1 — unguarded, that block's week-1 win probabilities/playoff odds would silently attach to the prior season's already-decided week-1 matchups and standings just because the week numbers happen to collide. `getPlayoffOdds` had a second, same-season risk too: `roster_id` isn't guaranteed to mean the same team across two different seasons' leagues. Fixed with a shared `simulationMatchesCurrentView()` gate (`index.html`) requiring both `meta.season === league.season` and `simulation.week === selectedWeek`, applied to both getters and to the three inline caveat-footer conditions that previously only checked week (so a caveat could show with no data behind it). This is a deliberate behavior change for `getPlayoffOdds` specifically: it now reads `—` for any week other than the one the export covers, not only for a season mismatch, trading the previous "always show the latest known odds" behavior for the same never-show-a-stale-number rule `getMyProj()` already followed. Verified against the real, completed 2025 season (league `1250182471429931008`): a real week-10 simulation block was rebuilt the same way Round 2's did (point-in-time-safe history truncated to weeks < 10, real model artifact, real Sleeper matchups/rosters) and served two ways — with `meta.season: 2025` (matching the displayed league), all 14 win-probability badges (7 real matchups) and all 14 rosters' playoff-odds percentages rendered, with both caveat footers visible; with the identical export relabeled `meta.season: 2026` (season mismatch only, week held constant at 10 in both), every badge and every Playoff Odds cell blanked to `—` and both caveat footers disappeared. Confirmed via Playwright against a local server, zero unexplained console errors in either case (the only console noise was the ESPN scoreboard sidebar's CORS failure, a known artifact of serving off of GitHub Pages' origin, unrelated to this change). |
| Phase 4 radar percentiles (`position_starter_counts()`, `build_radar_snapshot()`) are correct and match `index.html`'s startable-count logic | **Verified** — `position_starter_counts()` pinned against this league's real `roster_positions` (14 teams) reproduces QB=14/RB=33/WR=35/TE=16, matching the Weekly Production chart's own already-published "~35" WR footnote. Replayed against the real, completed 2025 season at week 10 (point-in-time-safe history, real model artifact): 391/555 candidates eligible, and 5 real, well-known players (Christian McCaffrey, James Cook, Justin Jefferson, Puka Nacua, Travis Kelce) each produced a percentile profile matching their real-world reputation on inspection, plus Joe Burrow correctly came back ineligible (4 of 5 games) for his real, injury-shortened 2025 season. Frontend verified via Playwright: the eligible case renders a real Chart.js radar whose `Chart.getChart(canvas).data` matches the Python-computed percentiles exactly; the ineligible case renders the specific games-played empty state. Zero new console errors. See **Phase 4 findings**. |
| Phase 5 heatmap zones (`receiving_zone_plays()`/`passing_zone_plays()`/`rushing_zone_plays()`, `build_heatmap_snapshot()`) derive real zones from real pbp and read correctly per position | **Verified** — replayed against the real, completed 2025 season at week 10 (point-in-time-safe history, real model artifact, real pbp): 391/555 candidates eligible, real play counts reported and spot-checked (single digits up to 168 real carries for a workhorse back). Two archetype contrasts specifically requested were confirmed on real, not hand-picked, players — Tyquan Thornton (60.7% of real targets in the two Deep zones) vs. Khalil Shakir (75%+ in Short/Behind-LOS zones, near-zero deep usage) for deep-threat-vs-slot; Josh Jacobs (real red-zone-heavy rushing, checkdown-only receiving on 28 real targets) vs. Christian McCaffrey (a similar rushing shape but 80 real targets spread across 10 zones including real intermediate/deep work) for goal-line-vs-passing-down. A real production bug was caught and fixed in the process: `nflreadpy.load_pbp()` raises `ValueError` (not the already-handled ConnectionError/404 shape) for a season with no pbp published yet, surfaced by running `scripts/weekly_update.py` for real against the live pre-draft 2026 league — fixed at that one caller (and the notebook's equivalent cell), matching where `build_weekly_scored`'s own analogous tolerance already lives, not pushed into `get_pbp` itself. Frontend verified via Playwright: correct panel titles/grid counts per position, the ineligible state renders its specific games-played reason, zero new console errors across a full click-through. See **Phase 5 findings**. |
| Season archives + selector: `get_archive_candidates()` doesn't silently drop real historical rostered players, the generated 2025 archive is well-formed, and the UI switches every panel together with no mixed-season state | **Verified** — `get_archive_candidates()` checked directly against this league's real historical rosters for all 5 completed seasons (2021-2025): `n_with_current_team == n_crosswalk_matched` for every one, vs. 46.9%-88.8% survival if `get_export_candidates()` had been reused unmodified (see the per-season table in **Phase 9 findings**). `scripts/archive_season.py 2025` generated a real, validated 921,516-byte archive (465 players, 466/1468 radar/heatmap-eligible, crosswalk match rate 99.0%). Frontend verified via Playwright against the real production config (unmodified `LEAGUE_ID`, real committed archive): defaulted to 2025 automatically (2026 has zero real games), and on 2025 every named panel (KPI cards, radar, heatmap, opportunity shares, weekly chart) reported the same real full season for Christian McCaffrey with no partial/mixed state; switching to 2026 correctly reverted radar/heatmap to "0 of 5 games" with zero 2025 data leaking through, and the standings table correctly emptied. Also caught and fixed a real latent bug in the process: `state.statsByWeek`/`state.projectionsByWeek` were cached by week number alone, which would have let one season's per-week stats silently answer for another's after a switch — fixed by clearing both caches on every season load, verified via a full click-through of every tab in both seasons plus a mid-navigation season switch, zero new console errors. See **Phase 9 findings**. |

## What's outstanding

- **`PHASE_2B_6_SPEC.md`** at the repo root is the working spec for Phases 2b, 6, and 6.5. Fold it into this doc and `NOTEBOOK_OUTLINE.md` once those phases are complete.
- ~~Trigger the first real `retrain.yml` and `weekly-update.yml` `workflow_dispatch` runs on GitHub Actions itself.~~ **Done** — both succeeded on real GitHub Actions infrastructure; see **Phase 8 findings** for the two real bugs their first attempts caught (a test-fixture network dependency, a git commit-step ordering bug) and how they were fixed.
- **`weekly-update.yml` hasn't yet run on its actual Tuesday cron schedule** — every verification so far is `workflow_dispatch`. The season hasn't started (2026 preseason as of this writing), so there's nothing scheduled to observe yet; worth a spot-check once the first real Tuesday during the season rolls around.
- **Phase 8 round 2's simulation wiring hasn't run on real GitHub Actions infrastructure yet** — verified locally (real 2025 week 10 data) and in a real browser, but `retrain.yml` (now training 5 quantiles instead of 2, 11.16 MB artifact) and `weekly-update.yml` (now producing the `simulation` block) both need one more real `workflow_dispatch` pair to confirm the CI runners handle the larger artifact and the extra per-week simulation fetches within their timeouts.
- Update `DEFAULT_LEAGUE_ID` in `src/ingest.py` each August when Sleeper rolls the league over
- **`HISTORICAL_SEASONS` in `scripts/retrain.py`** (currently `2018-2025`, mirroring every other model in this pipeline) needs bumping by hand once nflverse publishes a new season's data — same manual-update pattern as `DEFAULT_LEAGUE_ID`, not automatic.
- `src/ingest.py`'s fetchers still 404 with a raw traceback for an unpublished season in the general case — only `src/pipeline.py::build_weekly_scored`'s single-current-season path was fixed (see Verification status). A general `ingest.py`-level season guard is still undone, same open item as before Phase 8.
- Push the latest `index.html` changes to GitHub Pages (all recent work is local)
- **Activity feed panel** sizing vs matchups panel — layout issue, minor
- **NFL sidebar** currently shows preseason week labels; should default to last completed regular-season week
- **Build out the Python notebook pipeline** — Phases 4 (radar) and 5 (heatmap) are both done; the outline's remaining phases (6 shipping a model, further iteration) are the only ones left unstarted
- **Comparison tab's "Profile Overlay" placeholder** still needs wiring — Phase 4's `radar` key has the data, but overlaying multiple players' radars on one Chart.js chart (vs. Player Detail's single-player radar) wasn't part of that round; see **Phase 4 findings**. No equivalent multi-player heatmap overlay was requested or built for Phase 5 either.
- **`get_pbp()` still raises a raw, unguarded `ValueError` for an unpublished season in the general case** — only `scripts/weekly_update.py`'s and `07_export_json.ipynb`'s single-current-season heatmap-building callers were fixed (see **Phase 5 findings**), matching the same narrowly-scoped-fix pattern already true of `get_weekly_stats`'s equivalent 404 case (see the `src/ingest.py`-level season guard item below, which this is the same open item as, just a second exception shape).
- **Historical champion data** — plan is to maintain a small `champions.json` file by hand for the league's history
- **Season archives beyond 2025** — `src/ingest.py::SEASON_LEAGUE_IDS` already has real league_ids for 2021-2024 (this league's full Sleeper history), and `get_archive_candidates()` was verified against all 5 seasons before scoping to just 2025 (see Phase 9 findings). Adding one is `python scripts/archive_season.py <year>` plus one new entry in `index.html`'s `SEASON_OPTIONS` — no code changes needed, just a decision to do it.
- **`data/output/player_advanced_stats.json` now regenerates automatically** via `weekly-update.yml` (Tuesdays in-season, or `workflow_dispatch` any time) — the old "re-run `07_export_json.ipynb` by hand after the draft" step is superseded by this for ongoing updates; the notebook still exists and still works for manual/exploratory runs.

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
