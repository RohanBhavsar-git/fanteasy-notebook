"""
Tests for src/pipeline.py -- Phase 8's shared feature-building orchestration.
"""

import inspect
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (  # noqa: E402
    HISTORY_SEED_COLUMNS, _is_unpublished_season_error, build_feature_table, build_raw_features,
)
from src.usage import ROLLING_SOURCE_COLUMNS  # noqa: E402


def test_history_seed_columns_has_no_duplicates():
    """
    offense_pct is ALSO part of ROLLING_SOURCE_COLUMNS (via
    SNAP_OUTPUT_COLUMNS) -- an earlier version of HISTORY_SEED_COLUMNS
    listed it a second time explicitly, which produced a DataFrame with
    two identically-named columns when selected. That doesn't fail loudly
    at selection time; it fails much later, inside pd.concat, with a
    confusing "Reindexing only valid with uniquely valued Index objects"
    error -- see scripts/weekly_update.py's build_target_week_features
    call. Locking this down directly rather than trusting it by inspection.
    """
    assert len(HISTORY_SEED_COLUMNS) == len(set(HISTORY_SEED_COLUMNS)), (
        f"HISTORY_SEED_COLUMNS has duplicates: {HISTORY_SEED_COLUMNS}"
    )
    # every ROLLING_SOURCE_COLUMNS entry (offense_pct included) is covered
    assert set(ROLLING_SOURCE_COLUMNS) <= set(HISTORY_SEED_COLUMNS)


def test_history_seed_columns_selection_produces_unique_columns():
    """
    End-to-end version of the above: actually select HISTORY_SEED_COLUMNS
    out of a frame and confirm pandas doesn't silently hand back duplicate
    columns (which .columns.duplicated() would catch even if the list
    itself were deduplicated but something else re-introduced a clash).
    """
    df = pd.DataFrame({c: [] for c in HISTORY_SEED_COLUMNS})
    selected = df[HISTORY_SEED_COLUMNS]
    assert not selected.columns.duplicated().any()


def test_is_unpublished_season_error_detects_404_not_other_failures():
    response_404 = requests.Response()
    response_404.status_code = 404
    http_404 = requests.exceptions.HTTPError(response=response_404)
    wrapped_404 = ConnectionError("Failed to download ...")
    wrapped_404.__cause__ = http_404
    assert _is_unpublished_season_error(wrapped_404) is True

    response_500 = requests.Response()
    response_500.status_code = 500
    http_500 = requests.exceptions.HTTPError(response=response_500)
    wrapped_500 = ConnectionError("Failed to download ...")
    wrapped_500.__cause__ = http_500
    assert _is_unpublished_season_error(wrapped_500) is False

    # a connection error with no HTTPError cause at all (e.g. a genuine
    # network timeout) must NOT be treated as "season not published" --
    # that would silently swallow a real fetch failure.
    assert _is_unpublished_season_error(ConnectionError("timed out")) is False


def test_build_raw_features_returns_empty_for_empty_input_without_fetching():
    """
    scripts/weekly_update.py relies on this to short-circuit before a
    not-yet-started season's pbp/snaps/ngs/schedule are ever fetched --
    if this silently tried to fetch anyway, an empty weekly_scored (e.g.
    the offseason) would 404 on pbp too, one level down from where
    build_weekly_scored already handles it.
    """
    result = build_raw_features(pd.DataFrame(), seasons=[2099])
    assert result.empty


_FEATURE_FUNCTION_CALL_PATTERN = re.compile(r"\b(add_\w+_features)\s*\(")


def _feature_function_calls(source: str) -> set[str]:
    return set(_FEATURE_FUNCTION_CALL_PATTERN.findall(source))


def test_notebook_03_feature_chain_matches_build_feature_table():
    """
    notebooks/03_usage_features.ipynb is documented (its own §2 markdown,
    CLAUDE.md, PROJECT_CONTEXT.md) as the exploratory reference that mirrors
    build_feature_table's real production feature chain -- and it silently
    drifted out of sync once already: add_team_tendency_features was added
    to build_raw_features but never backported to this notebook's own
    pipeline cell. data/processed/weekly_features.parquet is gitignored, so
    nothing else catches a stale notebook producing an incomplete table
    (see PROJECT_CONTEXT.md's Context Columns findings for how that gap was
    actually found).

    Statically compares which `add_*_features` functions each side calls,
    by regex over source text -- not a full notebook execution, so this
    stays fast and has no network/data dependency. Fails loudly if a future
    feature family is added to build_raw_features/build_feature_table and
    not backported here, instead of silently producing a stale
    weekly_features.parquet that only surfaces as a confusing downstream
    KeyError or a model trained on fewer columns than the notebook implies.
    """
    production_calls = _feature_function_calls(
        inspect.getsource(build_raw_features) + inspect.getsource(build_feature_table)
    )
    assert production_calls, (
        "regex found zero add_*_features calls in build_raw_features/build_feature_table -- "
        "the pattern is broken, not the code; fix the regex before trusting this test."
    )

    notebook_path = PROJECT_ROOT / "notebooks" / "03_usage_features.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    notebook_calls = _feature_function_calls(notebook_source)

    missing = production_calls - notebook_calls
    assert not missing, (
        f"notebooks/03_usage_features.ipynb is missing {sorted(missing)} -- it no longer "
        "matches src/pipeline.py's real feature-building chain. Add the missing call(s) to "
        "the notebook's pipeline cell (and its imports), then re-run it end to end so "
        "data/processed/weekly_features.parquet stays correct."
    )
