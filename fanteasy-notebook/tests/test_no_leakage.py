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
    ROLLING_OUTPUT_COLUMNS,
    SITUATIONAL_OUTPUT_COLUMNS,
    SNAP_OUTPUT_COLUMNS,
    VOLUME_OUTPUT_COLUMNS,
    XFP_OUTPUT_COLUMNS,
    add_context_features,
    add_rolling_features,
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


@pytest.fixture(scope="module")
def featured_df(weekly_scored, pbp, snaps, crosswalk, schedule, scoring_settings) -> pd.DataFrame:
    """weekly_scored with Families 1-3 (steps 1-3) already applied -- the
    input add_rolling_features expects."""
    df = add_volume_features(weekly_scored, pbp)
    df = add_snap_features(df, snaps, crosswalk)
    df = add_situational_features(df, pbp)
    df = add_context_features(df, schedule)
    df = add_xfp_features(df, pbp, scoring_settings)
    return df


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


def test_rolling_features_shift_excludes_own_week(featured_df):
    """
    Perturbation test, precise and white-box in spirit: change one real
    player's target_share at a specific week to an extreme outlier and
    confirm that week's OWN target_share_ewm3/_vol/_s2d are UNCHANGED
    (proves the current week's raw value is excluded from its own
    summary), while the FOLLOWING week's target_share_ewm3 DOES change
    (proves the shift makes that value available one week later, rather
    than the column being ignored entirely).

    This is the test the black-box truncate-and-compare pattern below
    cannot provide -- same reasoning as the xFP tests in this file:
    truncating FUTURE weeks never touches week N's own row, so it can't
    tell a correctly-shifted implementation apart from a "forgot to
    shift" bug for week N specifically. This step is exactly where that
    kind of bug would matter most, so it gets the precise test.
    """
    counts = featured_df.groupby(["player_id", "season"]).size()
    pid, season = counts[counts >= 4].index[0]
    weeks = sorted(
        featured_df.loc[
            (featured_df["player_id"] == pid) & (featured_df["season"] == season), "week"
        ].tolist()
    )
    target_week, next_week = weeks[1], weeks[2]  # not the season's first week

    def _row(result, week):
        m = (result["player_id"] == pid) & (result["season"] == season) & (result["week"] == week)
        return result.loc[m].iloc[0]

    baseline = add_rolling_features(featured_df)
    baseline_target = _row(baseline, target_week)
    baseline_next = _row(baseline, next_week)

    perturbed_df = featured_df.copy()
    mask = (
        (perturbed_df["player_id"] == pid) & (perturbed_df["season"] == season)
        & (perturbed_df["week"] == target_week)
    )
    perturbed_df.loc[mask, "target_share"] = 999.0
    perturbed = add_rolling_features(perturbed_df)
    perturbed_target = _row(perturbed, target_week)
    perturbed_next = _row(perturbed, next_week)

    for col in ("target_share_ewm3", "target_share_vol", "target_share_s2d"):
        if pd.notna(baseline_target[col]):
            assert baseline_target[col] == pytest.approx(perturbed_target[col]), col

    assert baseline_next["target_share_ewm3"] != pytest.approx(perturbed_next["target_share_ewm3"])


@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_rolling_features_no_future_leakage(featured_df, season, boundary_week):
    """
    Removing rows for weeks AFTER boundary_week from the input df must
    not change rolling features for any week <= boundary_week. This
    catches a DIFFERENT bug class than the perturbation test above -- a
    future row leaking backward into an earlier week's window (e.g. a
    sort that silently didn't take effect, or an unsorted groupby) --
    not whether week N excludes its own row.
    """
    df_truncated = _truncate_after(featured_df, season, boundary_week)

    full = add_rolling_features(featured_df)
    truncated = add_rolling_features(df_truncated)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["player_id", "season", "week"] + ROLLING_OUTPUT_COLUMNS
    left = full.loc[mask, cols].reset_index(drop=True)
    right = truncated.loc[mask, cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


def test_rolling_features_resets_at_season_boundary(featured_df):
    """
    A player with substantial 2024 history must NOT carry games_played
    or ewm3/vol/s2d state into week 1 of 2025 -- grouping by
    (player_id, season) together, not just player_id, is what enforces
    this. Picks a real heavy-usage player rather than synthesizing one,
    so the check exercises the actual season transition in the data.

    Also checks the deliberate CONTRAST this step adds: the SAME row's
    prev_season_target_share must be POPULATED (not null) and must equal
    that player's actual 2024 average -- confirming the in-season reset
    and the prior-season carryover are two different, correctly
    independent mechanisms on the same row, not a contradiction.
    """
    result = add_rolling_features(featured_df)

    games_2024 = result.loc[result["season"] == 2024].groupby("player_id").size()
    heavy_players = games_2024[games_2024 >= 10].index
    wk1_2025 = result[
        (result["season"] == 2025) & (result["week"] == 1)
        & result["player_id"].isin(heavy_players)
    ]
    assert len(wk1_2025) > 0, "expected at least one heavy-2024 player with a week1-2025 row"

    row = wk1_2025.iloc[0]
    assert row["games_played"] == 0
    assert pd.isna(row["target_share_ewm3"])
    assert pd.isna(row["target_share_vol"])
    assert pd.isna(row["target_share_s2d"])

    manual_2024_avg = result.loc[
        (result["player_id"] == row["player_id"]) & (result["season"] == 2024), "target_share"
    ].mean()
    assert pd.notna(row["prev_season_target_share"])
    assert row["prev_season_target_share"] == pytest.approx(manual_2024_avg)


def test_prev_season_columns(featured_df):
    """
    Prior-season baselines don't need a "no leakage" test in the sense
    the rest of this file uses that phrase -- there's no future data
    anywhere in the computation, only an entire prior season's, which is
    complete by construction before the next season starts. What needs
    checking is correctness: every 2024 row is null (no season exists
    before 2024 in this dataset), every player's value is constant across
    all of their weeks within a season (computed once per player-season,
    not per week), and a player absent from the prior season entirely
    gets null rather than some fallback.
    """
    result = add_rolling_features(featured_df)

    assert result.loc[result["season"] == 2024, "prev_season_target_share"].isna().all()

    both_seasons = (
        result.groupby("player_id")["season"].apply(lambda s: {2024, 2025}.issubset(set(s)))
    )
    pid = both_seasons[both_seasons].index[0]
    values_2025 = result.loc[
        (result["player_id"] == pid) & (result["season"] == 2025), "prev_season_target_share"
    ]
    assert values_2025.nunique() == 1, "prev_season value must be constant across a player's weeks"

    only_2025 = result.groupby("player_id")["season"].apply(lambda s: set(s) == {2025})
    no_prior_season_players = only_2025[only_2025].index
    if len(no_prior_season_players):
        rows = result[result["player_id"].isin(no_prior_season_players)]
        assert rows["prev_season_target_share"].isna().all()


def test_rolling_features_idempotent(featured_df):
    once = add_rolling_features(featured_df)
    twice = add_rolling_features(once)
    pd.testing.assert_frame_equal(once, twice)
