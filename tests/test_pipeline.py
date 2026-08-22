"""
Tests for src/pipeline.py -- Phase 8's shared feature-building orchestration.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (  # noqa: E402
    HISTORY_SEED_COLUMNS, _is_unpublished_season_error, build_raw_features,
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
