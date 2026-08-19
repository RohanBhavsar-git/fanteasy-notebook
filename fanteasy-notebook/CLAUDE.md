# CLAUDE.md

Guidance for Claude Code working in this repository.

## Read first

- **`PROJECT_CONTEXT.md`** — the "why" behind every decision. Read it before
  proposing changes. Its **Verification status** table separates what has been
  confirmed against live data from what is still a design sketch; do not treat
  a sketch as settled.
- **`NOTEBOOK_OUTLINE.md`** — the 8-phase roadmap. Phases 3-8 were written
  before anything ran, so their code snippets are intent, not tested code.

If a doc and the code disagree, the code is right and the doc needs updating —
say so rather than silently following either.

## What this project is

A Python pipeline that produces `player_advanced_stats.json` for a single-file
HTML fantasy football dashboard. One specific 14-team Sleeper **redraft** league
("Fanteasy Football"), deliberately hard-coded — no league switcher, no accounts.

Current state: Phase 1 (ingestion) and Phase 2a (custom scoring) are complete
and verified. Phase 2b (usage/efficiency features) is next.

## Non-negotiables

1. **No fake, placeholder, seeded, or demo data. Ever.** If something can't be
   computed honestly, the dashboard shows an "Awaiting model output" card.
   Do not generate plausible-looking numbers to make a panel render.
2. **`index.html` is a single file**: vanilla JS, embedded CSS, Chart.js via
   CDN. No build system, no bundler, no framework, no npm. Do not "modernize"
   it, split it into modules, or reformat it wholesale.
3. **Reusable logic goes in `src/`, not in notebooks.** Notebooks import from
   `src/` and stay single-purpose. Exploration goes in a scratch notebook, not
   appended to a working one — that is what broke `01_data_ingestion.ipynb`
   before.
4. **Honest UI copy.** "This league", not "your league". Nothing should imply
   more precision than the data supports.

## The scoring code is load-bearing

`src/features.py::compute_custom_score()` reproduces this league's scoring
**exactly** — 100% match, zero mismatches, across 739 rostered player-weeks
(2025 weeks 5/8/10/12/15, every position including K).

That number was hard-won. Seven non-standard rules were discovered by diffing
against Sleeper's actual results, not by reading the settings dict. They look
like bugs and are not:

- `fum` uses `fumbles_total`, **not** the sum of rushing/receiving/sack fumbles
- `fum` and `fum_lost` **stack** — a lost fumble is −2, a self-recovered one −1
- `fum_rec` (+1) does **not** apply to offensive players
- `fgm_yds_over_30` is computed **per kick**, not from aggregate distance
- `pat_blocked` counts toward `xpmiss`
- `fgmiss` applies **only to misses under 50 yards**
- `pass_int_td` requires `td_team == defteam` — without it, a defender fumbling
  an interception return into his own end zone scores as a pick-six

**If you change anything in `features.py`, re-run the validation in
`notebooks/02_custom_scoring.ipynb` and confirm it still reports 0 mismatches
before committing.** A silent regression here corrupts the Phase 6 target
variable and everything downstream.

Related: **`fantasy_points_ppr` is not a valid target.** It's full PPR; this
league is 0.5. Train on `custom_points`.

## Conventions

- **Python 3.12**, `.venv` inside the project folder. `nflreadpy` (not the
  deprecated `nfl_data_py`) returns Polars; `src/ingest.py` converts to pandas
  at the boundary so everything downstream is pandas.
- **All fetches cache to `data/raw/`.** Never bypass the cache helpers or
  re-download in a loop. `refresh=True` forces a re-fetch when needed.
- **ID columns are strings, always.** `sleeper_id` arrives as float64 (nulls
  force the upcast), turning `"4984"` into `4984.0`; Sleeper's own IDs are
  strings, so a naive join returns zero rows *without raising*. If a merge
  comes back empty, check dtypes before anything else.
- **Filter to `season_type == 'REG'`** before any modeling work. Playoff weeks
  have real stats but usage patterns that don't reflect fantasy-relevant games.
- **Functions that mutate a DataFrame must be idempotent.** Notebook cells get
  re-run constantly; drop an existing column before merging it back, or you get
  `_x`/`_y` suffixes and a confusing KeyError.
- **Fail loudly.** A fetch that returns empty should raise with a message
  naming the likely cause, not return a blank DataFrame that reads as "no data
  published yet."

## Scope boundaries

- **K and DST are out of scope for the projection model.** Kicker output
  depends on how often the offense stalls in FG range — close to noise week to
  week. DST would need a team-defense model layered on an offense model. The
  scorer handles both correctly; the dashboard keeps showing Sleeper's
  projections for K/DST, labeled as Sleeper's.
- **No LLMs, RAG, or agents in the projection model.** This is tabular
  regression. Use gradient boosting.
- **The league ID is hard-coded.** Sleeper mints a new one each season for all
  league types; `DEFAULT_LEAGUE_ID` in `src/ingest.py` needs updating each
  August.

## Working style

- **Name tradeoffs and problems plainly** before implementing. If a request has
  an issue, say so rather than building it and mentioning the caveat after.
- **Small changes over large ones.** Multiple focused edits beat one bundled
  rewrite; issues get caught at each step instead of compounding.
- **After changes, summarize concisely**: what changed, why, how to commit. No
  preamble, no restating the request.
- **Say when something can't be done cleanly** with free public data. Don't
  invent a workaround that produces output nobody should trust.
- **Verify before claiming.** "This should work" and "I ran this and it
  reported 0 mismatches" are different statements. Use the second only when
  it's true.

## Commands

```bash
# activate the environment (Windows)
.venv\Scripts\activate

# run a notebook end-to-end without opening it
jupyter nbconvert --to notebook --execute notebooks/02_custom_scoring.ipynb --inplace

# strip notebook outputs before committing (keeps git diffs readable)
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

`data/raw/`, `data/processed/`, and `.venv/` are gitignored. Never commit
cached data or the environment.
