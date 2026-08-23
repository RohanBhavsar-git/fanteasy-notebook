"""
FanTeasy Stats -- Phase 8: model artifact persistence.

Bundles everything weekly-update.yml's inference-only run needs into ONE
committed file, so it never has to retrain or rebuild the full multi-season
feature table:
  - the per-position final models (src/model.py::train_final_models --
    point regressor + the 5-point 0.10/0.25/0.50/0.75/0.90 quantile set,
    trained on ALL history -- the extra 3 quantiles beyond q10/q90 exist
    for src/simulate.py's per-player distributions, not the dashboard's
    floor/ceiling, which only ever reads q10/q90)
  - feature_columns and cqr_widen_by_10_90/cqr_widen_by_25_75, so
    predictions always match whatever retrain.yml actually trained even
    if src/model.py's or src/export.py's own constants drift later
  - performance: the walk-forward MAE-vs-baselines table (same shape as
    src/export.py's PERFORMANCE_BY_POSITION), refreshed by every retrain
  - history_seed: a trimmed slice of the most recent
    HISTORY_SEED_SEASONS_BACK seasons' raw feature-input columns (see
    src/pipeline.py::HISTORY_SEED_COLUMNS) -- the minimum a not-yet-played
    week's stub row needs for add_rolling_features to compute in-season
    trend AND prev_season_* correctly, without weekly-update.yml re-fetching
    and re-featuring 8 seasons of history just to predict one week. This is
    the piece that makes "fetch current-season data only" possible at all;
    see scripts/weekly_update.py's module docstring for the full mechanism.

retrain.yml is the ONLY workflow that writes this file -- weekly-update.yml
only reads it. Committing it (rather than data/raw or data/processed, which
stay gitignored per CLAUDE.md's "never commit cached data") is a deliberate,
narrow exception; see .gitignore's `models/` block.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "models" / "fanteasy_model.joblib"

# Keys every artifact must have -- checked on load so a malformed or
# partially-written file fails loudly at load time, not with a cryptic
# KeyError deep inside a prediction call.
_REQUIRED_KEYS = (
    "trained_at", "model_version", "seasons_trained", "feature_columns",
    "cqr_widen_by_10_90", "cqr_widen_by_25_75", "performance", "history_seed",
    "history_seed_seasons", "models",
)


def save_model_artifact(
    models: dict,
    feature_columns: list[str],
    cqr_widen_by_10_90: dict[str, float],
    cqr_widen_by_25_75: dict[str, float],
    performance: dict,
    seasons_trained: list[int],
    history_seed,
    history_seed_seasons: list[int],
    model_version: str,
    path: Path = MODEL_ARTIFACT_PATH,
) -> Path:
    """Writes the artifact and returns the path actually written, so a
    caller (scripts/retrain.py) can report its size without guessing."""
    artifact = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "seasons_trained": list(seasons_trained),
        "feature_columns": list(feature_columns),
        "cqr_widen_by_10_90": dict(cqr_widen_by_10_90),
        "cqr_widen_by_25_75": dict(cqr_widen_by_25_75),
        "performance": performance,
        "history_seed": history_seed,
        "history_seed_seasons": list(history_seed_seasons),
        "models": models,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    size_mb = path.stat().st_size / 1e6
    logger.info(f"[artifact write] {path} ({size_mb:.2f} MB)")
    return path


def load_model_artifact(path: Path = MODEL_ARTIFACT_PATH) -> dict:
    """
    Raises rather than returning None -- weekly_update.py has nothing
    useful to fall back to without a trained model, so a missing/corrupt
    artifact should stop the run loudly (matching src/ingest.py's own
    "fail loudly" convention for fetch errors) rather than silently
    skipping prediction or crashing later with a confusing KeyError.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No model artifact at {path}. Run retrain.yml (or `python scripts/retrain.py` "
            "locally) at least once before weekly_update.py can do inference-only prediction "
            "-- there is nothing trained to load yet."
        )
    artifact = joblib.load(path)
    missing = [k for k in _REQUIRED_KEYS if k not in artifact]
    if missing:
        raise KeyError(f"Model artifact at {path} is missing keys {missing} -- re-run retrain.yml.")
    return artifact
