"""
Leakage test for Phase 2b usage features (src/usage.py).

Families 1 (volume/share) and 2 (snap share) are NOT rolling/expanding
features -- each player-week's value is computed entirely from that week's
own pbp and snap-count rows, grouped within (team, season, week). There is
no trailing window to leak across yet (that's step 4, add_rolling_features,
which gets its own leakage test).

What this guards against: a bug that accidentally aggregates a "team total"
across more than one week (e.g. season-to-date instead of within-week),
which would make week N's feature quietly depend on weeks after N.

Note this is a narrower claim than PHASE_2B_6_SPEC.md's general point-in-time
rule ("features for week N use only weeks < N"). That rule describes ROLLING
features PREDICTING week N. Families 1-2 instead DESCRIBE week N's own game
-- information that exists once week N has been played -- so the boundary
tested here is "week N does not depend on weeks > N", not "week N does not
depend on week N itself". Truncating the source data at week N-1 (excluding
week N) would make every denominator NaN and the test would fail by
construction; that's not a leak, that's the feature correctly having no data
to work with.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import (  # noqa: E402
    get_id_crosswalk, get_pbp, get_schedule, get_sleeper_league, get_snap_counts,
)
from src.usage import (  # noqa: E402
    CONTEXT_OUTPUT_COLUMNS,
    FANTASY_POSITIONS,
    SITUATIONAL_OUTPUT_COLUMNS,
    SNAP_OUTPUT_COLUMNS,
    VOLUME_OUTPUT_COLUMNS,
    XFP_OUTPUT_COLUMNS,
    add_context_features,
    add_situational_features,
    add_snap_features,
    add_volume_features,
    add_xfp_features,
    _bucket_rate_table,
)

BOUNDARIES = [(2024, 5), (2024, 10), (2025, 5), (2025, 12)]


@pytest.fixture(scope="module")
def weekly_scored() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "weekly_scored.parquet"
    df = pd.read_parquet(path)
    df = df[(df["season_type"] == "REG") & df["position"].isin(FANTASY_POSITIONS)]
    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def pbp() -> pd.DataFrame:
    return get_pbp([2024, 2025])


@pytest.fixture(scope="module")
def snaps() -> pd.DataFrame:
    return get_snap_counts([2024, 2025])


@pytest.fixture(scope="module")
def crosswalk() -> pd.DataFrame:
    return get_id_crosswalk()


@pytest.fixture(scope="module")
def schedule() -> pd.DataFrame:
    return get_schedule([2024, 2025])


@pytest.fixture(scope="module")
def scoring_settings() -> dict:
    return get_sleeper_league()["scoring_settings"]


def _truncate_after(df: pd.DataFrame, season: int, boundary_week: int) -> pd.DataFrame:
    """Drop rows for `season` beyond `boundary_week`; other seasons untouched."""
    drop_mask = (df["season"] == season) & (df["week"] > boundary_week)
    return df[~drop_mask]


@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_volume_features_no_future_leakage(weekly_scored, pbp, season, boundary_week):
    pbp_truncated = _truncate_after(pbp, season, boundary_week)

    full = add_volume_features(weekly_scored, pbp)
    truncated = add_volume_features(weekly_scored, pbp_truncated)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["player_id", "season", "week"] + VOLUME_OUTPUT_COLUMNS
    left = full.loc[mask, cols].reset_index(drop=True)
    right = truncated.loc[mask, cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_snap_features_no_future_leakage(weekly_scored, snaps, crosswalk, season, boundary_week):
    snaps_truncated = _truncate_after(snaps, season, boundary_week)

    full = add_snap_features(weekly_scored, snaps, crosswalk)
    truncated = add_snap_features(weekly_scored, snaps_truncated, crosswalk)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["player_id", "season", "week"] + SNAP_OUTPUT_COLUMNS
    left = full.loc[mask, cols].reset_index(drop=True)
    right = truncated.loc[mask, cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


def test_volume_features_idempotent(weekly_scored, pbp):
    """Calling add_volume_features twice must not create _x/_y suffix columns."""
    once = add_volume_features(weekly_scored, pbp)
    twice = add_volume_features(once, pbp)
    pd.testing.assert_frame_equal(once, twice)


def test_snap_features_idempotent(weekly_scored, snaps, crosswalk):
    once = add_snap_features(weekly_scored, snaps, crosswalk)
    twice = add_snap_features(once, snaps, crosswalk)
    pd.testing.assert_frame_equal(once, twice)


@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_situational_features_no_future_leakage(weekly_scored, pbp, season, boundary_week):
    pbp_truncated = _truncate_after(pbp, season, boundary_week)

    full = add_situational_features(weekly_scored, pbp)
    truncated = add_situational_features(weekly_scored, pbp_truncated)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["player_id", "season", "week"] + SITUATIONAL_OUTPUT_COLUMNS
    left = full.loc[mask, cols].reset_index(drop=True)
    right = truncated.loc[mask, cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_context_features_no_future_leakage(weekly_scored, schedule, season, boundary_week):
    schedule_truncated = _truncate_after(schedule, season, boundary_week)

    full = add_context_features(weekly_scored, schedule)
    truncated = add_context_features(weekly_scored, schedule_truncated)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["player_id", "season", "week"] + CONTEXT_OUTPUT_COLUMNS
    left = full.loc[mask, cols].reset_index(drop=True)
    right = truncated.loc[mask, cols].reset_index(drop=True)

    # check_dtype=False: truncating the schedule leaves rows for weeks
    # AFTER the boundary unmatched, which upcasts int32/bool columns to
    # float64/object for the WHOLE column (a pandas dtype-widening side
    # effect of the left merge, not a leak) even though every row under
    # comparison here (<= boundary_week) still has a real, matched value
    # in both frames. Values are still checked exactly.
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_situational_features_idempotent(weekly_scored, pbp):
    once = add_situational_features(weekly_scored, pbp)
    twice = add_situational_features(once, pbp)
    pd.testing.assert_frame_equal(once, twice)


def test_context_features_idempotent(weekly_scored, schedule):
    once = add_context_features(weekly_scored, schedule)
    twice = add_context_features(once, schedule)
    pd.testing.assert_frame_equal(once, twice)


def test_bucket_rate_table_excludes_its_own_week():
    """
    Precise, white-box check that _bucket_rate_table(plays, season, week)
    excludes week == the cutoff itself, not just weeks after it.

    A "<=" vs "<" bug here CANNOT be caught by the black-box
    truncate-and-compare pattern the other families use (and an earlier,
    now-removed version of this test tried): week N's own plays are also
    the NUMERATOR (the player's actual opportunities that week, which is
    real, already-observed information -- not something point-in-time
    correctness should hide). Removing week N's plays from pbp entirely
    breaks that numerator too, which produced a false failure across
    ~7% of rows unrelated to the rate table itself. The rate table is the
    only thing that has to stay backward-only; this test isolates it.
    """
    plays = pd.DataFrame({
        "player_id": ["A", "B", "C", "outlier"],
        "season": [2024, 2024, 2024, 2024],
        "week": [3, 3, 4, 5],
        "bucket": ["0-9|20-50"] * 4,
        "points": [1.0, 2.0, 1.5, 1000.0],
    })
    rate = _bucket_rate_table(plays, season=2024, week=5)
    assert rate["0-9|20-50"] == pytest.approx((1.0 + 2.0 + 1.5) / 3)


@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_xfp_no_future_leakage(weekly_scored, pbp, scoring_settings, season, boundary_week):
    """
    Removing weeks AFTER boundary_week from pbp must not change xfp for
    any week <= boundary_week -- catches a bug where the rate table's
    "before this week" filter accidentally looks forward, or a future
    week's plays otherwise leak backward into an earlier week's rate.

    This does NOT test whether week N's own plays leak into week N's own
    rate table -- that needs test_bucket_rate_table_excludes_its_own_week
    above, because week N's own plays must stay in pbp here (they're also
    the numerator for week N, not just a candidate rate-table input).
    """
    pbp_truncated = _truncate_after(pbp, season, boundary_week)

    full = add_xfp_features(weekly_scored, pbp, scoring_settings)
    truncated = add_xfp_features(weekly_scored, pbp_truncated, scoring_settings)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["player_id", "season", "week"] + XFP_OUTPUT_COLUMNS
    left = full.loc[mask, cols].reset_index(drop=True)
    right = truncated.loc[mask, cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


def test_xfp_idempotent(weekly_scored, pbp, scoring_settings):
    once = add_xfp_features(weekly_scored, pbp, scoring_settings)
    twice = add_xfp_features(once, pbp, scoring_settings)
    pd.testing.assert_frame_equal(once, twice)
