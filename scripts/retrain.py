"""
FanTeasy Stats -- Phase 8: retrain.yml's entry point.

workflow_dispatch only, never scheduled (see .github/workflows/retrain.yml).
Fetches ALL historical seasons, rebuilds the full Family 1-6 feature table
from scratch, walk-forward-validates the point model against Sleeper's
projection and the two trailing-average baselines over the most recent
EVAL_SEASONS_BACK seasons, trains FINAL (no-holdout) point + the full
0.10/0.25/0.50/0.75/0.90 quantile set per position on the complete
history, and saves everything weekly_update.py needs for inference-only
prediction into ONE committed artifact (src/artifacts.py::save_model_artifact).
The 3 quantiles beyond q10/q90 (q25/q50/q75) exist for
src/simulate.py's per-player distributions (Phase 8 round 2) -- the
dashboard's own floor/point/ceiling still only ever reads q10/q90.

This is deliberately the expensive job, run by hand -- weekly_update.py
never retrains, so week-to-week output changes come from new data, not a
moving model. Run it locally with:
    .venv\\Scripts\\python.exe scripts\\retrain.py

Deliberately does NOT re-derive the CQR floor/ceiling widening constants
from scratch on every run. Doing that properly needs walk-forward QUANTILE
predictions across the ENTIRE multi-season history (calibration = every
fold before the eval window, not just the eval window itself) -- the single
most expensive computation in this pipeline (see PROJECT_CONTEXT.md's CQR
section: it required its own dedicated analysis pass, not a routine one).
Unlike point-MAE-vs-baselines, that constant does not meaningfully drift
retrain to retrain (no hyperparameter tuning happens anywhere in this
pipeline, so what it's correcting for doesn't change either) -- it's
reused as-is from src/export.py::CQR_WIDEN_BY_10_90 and carried into the
artifact. Recomputing it is a deliberate, separate, manual analysis (rerun
the notebook 07 CQR cells and update that constant by hand), not something
this script attempts -- see its docstring.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import save_model_artifact  # noqa: E402
from src.export import CQR_WIDEN_BY_10_90, CQR_WIDEN_BY_25_75  # noqa: E402
from src.ingest import DEFAULT_LEAGUE_ID, get_id_crosswalk, get_sleeper_league, get_sleeper_projections  # noqa: E402
from src.model import (  # noqa: E402
    FEATURE_COLUMNS_BY_POSITION, POSITIONS, SIMULATION_QUANTILE_ALPHAS, add_sleeper_baseline, add_target_baselines,
    evaluate_position, train_final_models, walk_forward_predict,
)
from src.pipeline import HISTORY_SEED_COLUMNS, build_feature_table  # noqa: E402

# Bump each year once nflreadpy actually publishes the new season's data --
# requesting an unpublished season 404s (see PROJECT_CONTEXT.md's
# Verification status table). Matches every other model in this pipeline's
# default (2018-2025, the Phase 6 data-volume result).
HISTORICAL_SEASONS = list(range(2018, 2026))

# Walk-forward performance is evaluated on the most recent N seasons of
# HISTORICAL_SEASONS (earlier seasons still feed training for those folds --
# see src/model.py::chronological_folds's eval_min_season). 2 matches the
# "2024 Wk5-2025 Wk18" window PROJECT_CONTEXT.md's Phase 6 findings already
# report performance numbers against.
EVAL_SEASONS_BACK = 2

# How many of the most recent COMPLETED seasons get embedded in the
# artifact's history_seed for weekly_update.py to build on top of. 2 covers
# both in-season rolling needs (this season's own weeks) and prev_season_*
# (last season's full average) with one season of slack.
HISTORY_SEED_SEASONS_BACK = 2

BASELINE_COLUMNS = {
    "sleeper_proj": "baseline_sleeper_proj",
    "season_to_date_avg": "baseline_season_to_date_avg",
    "trailing_3wk_avg": "baseline_trailing_3wk_avg",
}


def _git_short_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    return result.stdout.strip() or "unknown"


def _metric(summary, comparison: str, series: str) -> float:
    row = summary[(summary["comparison"] == comparison) & (summary["series"] == series)]
    if row.empty or row["mae"].isna().all():
        raise RuntimeError(f"No MAE computed for comparison={comparison!r} series={series!r}")
    return round(float(row["mae"].iloc[0]), 2)


def main() -> None:
    print(f"[1/5] Building full feature table for seasons {HISTORICAL_SEASONS}...")
    features = build_feature_table(HISTORICAL_SEASONS, DEFAULT_LEAGUE_ID)
    if features.empty:
        raise RuntimeError("build_feature_table returned zero rows -- aborting, nothing to train on.")
    print(f"    {len(features):,} rows x {features.shape[1]} columns")

    features = add_target_baselines(features)

    max_season = max(HISTORICAL_SEASONS)
    eval_min_season = max_season - EVAL_SEASONS_BACK + 1
    print(f"[2/5] Walk-forward validation, eval window: season >= {eval_min_season}...")

    crosswalk = get_id_crosswalk()
    scoring_settings = get_sleeper_league(DEFAULT_LEAGUE_ID)["scoring_settings"]

    performance = {}
    for position in POSITIONS:
        # feature_cols is THIS position's own list (FEATURE_COLUMNS_BY_
        # POSITION), not walk_forward_predict's shared-list default -- the
        # reported performance here must match what train_final_models
        # actually trains below (step 3), or this table would silently
        # describe a model this run doesn't produce. See PROJECT_CONTEXT.md's
        # Team Tendencies findings for why QB's own list differs from the
        # other three.
        wf = walk_forward_predict(
            features, position, feature_cols=FEATURE_COLUMNS_BY_POSITION[position], eval_min_season=eval_min_season
        )
        if wf.empty:
            raise RuntimeError(f"Walk-forward validation produced zero rows for {position} -- aborting.")

        wf = wf.merge(
            features[["player_id", "season", "week", "baseline_season_to_date_avg", "baseline_trailing_3wk_avg"]],
            on=["player_id", "season", "week"], how="left",
        )
        eval_pairs = sorted(set(zip(wf["season"], wf["week"])))
        wf = add_sleeper_baseline(wf, eval_pairs, crosswalk, scoring_settings, get_sleeper_projections)

        summary = evaluate_position(wf, "custom_points", BASELINE_COLUMNS)
        performance[position] = {
            "model_mae": _metric(summary, "model (full sample)", "model_a"),
            "sleeper_mae": _metric(summary, "sleeper_proj", "sleeper_proj"),
            "s2d_mae": _metric(summary, "season_to_date_avg", "season_to_date_avg"),
        }
        print(f"    {position}: {performance[position]} (n={len(wf)})")

    print("[3/5] Training final (no-holdout) models on the complete history...")
    models = train_final_models(
        features, feature_cols=FEATURE_COLUMNS_BY_POSITION, positions=POSITIONS,
        quantile_alphas=SIMULATION_QUANTILE_ALPHAS,
    )
    missing_positions = [p for p in POSITIONS if p not in models]
    if missing_positions:
        raise RuntimeError(f"train_final_models produced no model for {missing_positions} -- aborting.")

    print(f"[4/5] Building history_seed (last {HISTORY_SEED_SEASONS_BACK} seasons)...")
    seed_min_season = max_season - HISTORY_SEED_SEASONS_BACK + 1
    history_seed = features.loc[features["season"] >= seed_min_season, HISTORY_SEED_COLUMNS].reset_index(drop=True)
    history_seed_seasons = sorted(history_seed["season"].unique().tolist())
    if not history_seed_seasons:
        raise RuntimeError("history_seed is empty -- weekly_update.py would have nothing to build on.")
    print(f"    {len(history_seed):,} rows, seasons {history_seed_seasons}")

    print("[5/5] Saving model artifact...")
    model_version = _git_short_sha()
    path = save_model_artifact(
        models=models,
        feature_columns=FEATURE_COLUMNS_BY_POSITION,
        cqr_widen_by_10_90=CQR_WIDEN_BY_10_90,
        cqr_widen_by_25_75=CQR_WIDEN_BY_25_75,
        performance=performance,
        seasons_trained=HISTORICAL_SEASONS,
        history_seed=history_seed,
        history_seed_seasons=history_seed_seasons,
        model_version=model_version,
    )
    size_mb = path.stat().st_size / 1e6
    print(f"Wrote {path} ({size_mb:.2f} MB), model_version={model_version}")
    print("performance:", performance)


if __name__ == "__main__":
    main()
