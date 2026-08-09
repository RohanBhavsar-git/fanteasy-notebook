# FanTeasy Stats — Project Context

This document captures the "why" behind the FanTeasy Stats project so a new conversation can pick up seamlessly. Read this first before starting any new work.

> **Last updated:** August 2026, during local environment setup for Phase 1.
> See **Verification status** near the end before trusting anything in the
> notebook-pipeline sections — some of it is decided-but-not-yet-run.

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

### Python notebook (Phase 1 scaffolded, not yet run end-to-end)
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
- **`.venv`** in the project root, gitignored
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
Because the league has custom scoring (6-pt pass TDs, TE-premium PPR, yardage bonuses, etc.), Sleeper's pre-aggregated `pts_ppr` / `pts_half_ppr` / `pts_std` fields don't reflect the actual points. There's a dedicated **`computeCustomScore(statsObj, scoringSettings, position)`** helper near the top of the script that multiplies every raw stat field by its matching `scoring_settings` weight. **`resolvePts()`** wraps that with a fallback chain. Every fantasy total on the dashboard flows through `resolvePts()` — search that name to see every usage.

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
| 1 | Data ingestion — `src/ingest.py` + `01_data_ingestion.ipynb` | **Scaffolded + migrated to `nflreadpy`** — not yet run end-to-end |
| 2 | Feature engineering — per-player aggregates from pbp | Not started |
| 3 | Role classification — rule-based Pocket Passer / 3-Down Back / Slot / etc. | Not started |
| 4 | Radar metrics — 0-100 percentile normalization within position | Not started |
| 5 | Heatmap zones — field-location frequency tables | Not started |
| 6 | Projection model — XGBoost/LightGBM regression with time-series CV | Not started |
| 7 | JSON export — assemble `player_advanced_stats.json` | Not started |
| 8 | GitHub Actions weekly automation | Not started |

The dashboard already has integration hooks waiting. Search `state.advancedStats` and `p.myProj` and `// Hook for your custom model` in `index.html` to see exactly where each output will slot in.

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
- **Legal/financial advice pattern for LLM discussion**: When Rohan asked about "modern methods gaining traction," the answer was tiered (Strongly Suggest / Industry-Standard Tooling / Frontier) with an **explicit warning against shoehorning LLMs into a tabular regression project**. Follow the pattern — don't just list every trendy technique. Match tool to problem.

---

## Verification status

Be precise about what's actually been confirmed, so a fresh session doesn't inherit
assumptions as facts.

| Claim | Status |
|---|---|
| `nfl_data_py` is deprecated in favor of `nflreadpy` | **Verified** — nflverse's own announcement |
| `nflreadpy` function names + signatures used in `src/ingest.py` | **Verified** against the published API reference |
| Local env: Python 3.12.9, `.venv` kernel resolves, `src/` importable | **Verified** in VS Code |
| Package install completed | **Not verified** — last check showed `nflreadpy`, `polars`, `pyarrow` still missing |
| Notebook runs end-to-end | **Not verified** — never executed |
| Sleeper league ID is current | **Resolved** — 2026 league is `1389706592789733376`, chained from 2025 via `previous_league_id`. Sleeper mints a new ID every season for all league types, so this needs updating each August. |
| ID crosswalk actually joins nflverse ↔ Sleeper | **Not verified** — notebook cell 29 asserts on this |
| pandas 3.x compatibility | **Not verified** — `ingest.py` written against 2.x conventions; local env has 3.0.5 |

## What's outstanding

- **Finish the Phase 1 local run** — install packages, run the notebook top to bottom, resolve the two open questions above (league season, crosswalk join)
- Bump the Phase 8 CI workflow from Python 3.11 to 3.12
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

> "Continuing my FanTeasy Stats project. Attached is the current `index.html`, the notebook outline, and the project context doc. Ready to work on Phase 2 of the notebook."

Read this doc, skim `NOTEBOOK_OUTLINE.md` for the current phase, and confirm you understand the project before proposing changes. Don't ask questions this doc already answers.

Check the **Verification status** table before treating any pipeline claim as settled.
