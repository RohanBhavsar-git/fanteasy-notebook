# FanTeasy Stats — Data Pipeline

Python notebook + module for producing `player_advanced_stats.json`, the data file consumed by the [FanTeasy Stats dashboard](https://rohanbhavsar-git.github.io/fanteasystats).

## What this repo does

Pulls play-by-play, snap counts, projections, and injury data from **nflverse** (via `nflreadpy`) and **Sleeper's public API**, then computes:
- Per-player advanced stats (aDOT, YAC, target share, snap share, etc.)
- Position-based role classification (Pocket Passer, 3-Down Back, Slot WR, ...)
- Field-zone heatmaps (pass/target/rush location frequencies)
- Custom weekly fantasy projections that respect this league's specific scoring rules

Output is a single JSON file the dashboard fetches on load.

## Setup

Requires **Python 3.12** (not 3.13/3.14 — the Phase 6 ML stack lags new releases,
and local should match the CI workflow).

### VS Code (recommended)

1. **File → Open Folder** → select this `fanteasy-notebook` folder.
   Opening the *folder*, not a loose `.ipynb`, is what makes `src/` importable.
2. Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) → **Python: Create Environment**
   → **Venv** → Python 3.12 → tick `requirements.txt`.
3. Open `notebooks/01_data_ingestion.ipynb` and select the `.venv` kernel
   (top-right corner).

### Terminal

```bash
py -3.12 -m venv .venv        # Windows
# python3.12 -m venv .venv    # macOS/Linux

.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python -m ipykernel install --user --name fanteasy --display-name "FanTeasy"
```

### Verify before running anything

Paste into the first notebook cell:

```python
import sys
from pathlib import Path
print("Python :", sys.version.split()[0])
print("Kernel :", sys.executable)
print("CWD    :", Path.cwd())
print("Sees src/ingest.py:", (Path.cwd().parent / "src" / "ingest.py").exists())
for pkg in ("nflreadpy", "polars", "pandas", "pyarrow", "requests"):
    try:
        m = __import__(pkg)
        print(f"  {pkg:<10} {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"  {pkg:<10} MISSING")
```

You need: Python `3.12.x`, a kernel path containing `.venv`, a CWD ending in
`notebooks`, `True` for `src/ingest.py`, and five version numbers.

A `MISSING` package almost always means the terminal and the notebook kernel are
pointed at different environments. `%pip install -r ../requirements.txt` run
*inside a notebook cell* installs into whatever the kernel is using, so it can't
get out of sync.

## Project structure

```
fanteasy-notebook/
├── notebooks/
│   ├── 01_data_ingestion.ipynb       # ← Start here (Phase 1)
│   ├── 02_feature_engineering.ipynb  # Coming next
│   ├── 03_role_classification.ipynb
│   ├── 04_radar_metrics.ipynb
│   ├── 05_heatmap_zones.ipynb
│   ├── 06_projection_model.ipynb
│   └── 07_export_json.ipynb
├── src/
│   ├── ingest.py                     # ← Reusable data-pull functions (Phase 1)
│   ├── features.py                   # Coming next
│   └── ...
├── data/
│   ├── raw/                          # Cached fetches — gitignored
│   ├── processed/                    # Intermediate features
│   └── output/
│       └── player_advanced_stats.json
├── PROJECT_CONTEXT.md                # Why the project is built this way
├── NOTEBOOK_OUTLINE.md               # The 8-phase roadmap
├── .venv/                            # Local environment — gitignored
├── requirements.txt
└── README.md
```

Notebooks are for exploration and visualization. All reusable logic lives in `src/` so it can be imported into any notebook or into a CI job.

## Where the data comes from

| Source | Function in `src/ingest.py` | Cached under |
|---|---|---|
| nflverse play-by-play | `get_pbp()` | `data/raw/pbp_{seasons}.parquet` |
| nflverse weekly stats | `get_weekly_stats()` | `data/raw/weekly_{seasons}.parquet` |
| nflverse snap counts | `get_snap_counts()` | `data/raw/snaps_{seasons}.parquet` |
| Next Gen Stats | `get_ngs_data()` | `data/raw/ngs_*_{seasons}.parquet` |
| Schedule + weather | `get_schedule()` | `data/raw/schedule_{seasons}.parquet` |
| Player ID crosswalk | `get_id_crosswalk()` | `data/raw/id_crosswalk.parquet` |
| Sleeper league config | `get_sleeper_league()` | `data/raw/sleeper_league_{id}.json` |
| Sleeper player DB | `get_sleeper_players()` | `data/raw/sleeper_players.json` |
| Sleeper projections | `get_sleeper_projections()` | `data/raw/sleeper_proj_{yr}_wk{wk}.json` |

All fetches are cached — running any function a second time returns instantly. Pass `refresh=True` to force a re-fetch.

## Two things that will bite you

**1. `nflreadpy` returns Polars, not pandas.** Every function in `src/ingest.py`
converts at the boundary, so what you get back is a pandas DataFrame. If you call
`nflreadpy` directly, remember its API is different (`.filter()`, not `[mask]`).

**2. ID columns must stay strings.** `sleeper_id` arrives as float64 because nulls
force the upcast, turning Sleeper's `"4984"` into `4984.0`. Sleeper's own IDs are
strings, so a naive join returns zero rows *without raising*. `get_id_crosswalk()`
normalizes this. If a merge comes back empty, check dtypes before anything else.

## Current status

- **Phase 1 (Data ingestion)** — code complete, migrated to `nflreadpy`, being run locally for the first time
- **Phase 2 (Feature engineering)** — Not started
- **Phase 3-8** — Not started

Open questions Phase 1 should answer:
- Does the ID crosswalk actually join nflverse ↔ Sleeper? (notebook cell 29 asserts on it)
- Is `DEFAULT_LEAGUE_ID` current? Sleeper mints a new league ID each season.

See `NOTEBOOK_OUTLINE.md` for the full roadmap and `PROJECT_CONTEXT.md` for design decisions.
