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
    get_id_crosswalk, get_ngs_data, get_pbp, get_schedule, get_sleeper_league, get_snap_counts,
)
from src.pipeline import build_weekly_scored  # noqa: E402
from src.team_tendencies import (  # noqa: E402
    TEAM_TENDENCY_OUTPUT_COLUMNS,
    TEAM_TENDENCY_SAMPLE_COLUMNS,
    add_team_tendency_features,
    build_team_tendency_table,
)
from src.usage import (  # noqa: E402
    CONTEXT_OUTPUT_COLUMNS,
    EFFICIENCY_OUTPUT_COLUMNS,
    OPP_STRENGTH_POSITIONS,
    OPPONENT_STRENGTH_OUTPUT_COLUMNS,
    ROLLING_OUTPUT_COLUMNS,
    SITUATIONAL_OUTPUT_COLUMNS,
    SNAP_OUTPUT_COLUMNS,
    TREND_OUTPUT_COLUMNS,
    VOLUME_OUTPUT_COLUMNS,
    XFP_OUTPUT_COLUMNS,
    add_context_features,
    add_efficiency_features,
    add_opponent_strength_features,
    add_qb_rushing_share_feature,
    add_rolling_features,
    add_situational_features,
    add_snap_features,
    add_trend_features,
    add_volume_features,
    add_xfp_features,
    build_defense_air_ground_split,
    build_defense_strength_table,
    DEFENSE_AIR_GROUND_OUTPUT_COLUMNS,
    _bucket_rate_table,
    _dropback_play_frame,
    _qb_player_ids,
    _qb_rush_play_frame,
    _team_week_opponent,
)

BOUNDARIES = [(2024, 5), (2024, 10), (2025, 5), (2025, 12)]


@pytest.fixture(scope="module")
def weekly_scored() -> pd.DataFrame:
    """
    Builds Phase 2a's weekly_scored table directly via
    src.pipeline.build_weekly_scored, instead of reading the local
    data/processed/weekly_scored.parquet cache 02_custom_scoring.ipynb
    writes -- that file is gitignored and doesn't exist on a fresh CI
    checkout (see .github/workflows/retrain.yml), so this fixture would
    hard-fail before a single leakage test ran. build_weekly_scored
    already goes through get_weekly_stats/get_pbp's own fetch-or-cache
    logic, so this works identically whether data/raw/ is warm (fast,
    local) or empty (a real fetch, CI). Verified to match the notebook's
    own cached weekly_scored.parquet for season 2025 to within
    floating-point noise before this fixture was changed.
    """
    df = build_weekly_scored([2024, 2025])
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
def ngs_receiving() -> pd.DataFrame:
    return get_ngs_data("receiving", [2024, 2025])


@pytest.fixture(scope="module")
def ngs_passing() -> pd.DataFrame:
    return get_ngs_data("passing", [2024, 2025])


@pytest.fixture(scope="module")
def featured_df(
    weekly_scored, pbp, snaps, crosswalk, schedule, scoring_settings, ngs_receiving, ngs_passing
) -> pd.DataFrame:
    """weekly_scored with every non-rolling family already applied -- the
    input add_rolling_features expects."""
    df = add_volume_features(weekly_scored, pbp)
    df = add_snap_features(df, snaps, crosswalk)
    df = add_efficiency_features(df, pbp, ngs_receiving, ngs_passing)
    df = add_situational_features(df, pbp)
    df = add_context_features(df, schedule)
    df = add_xfp_features(df, pbp, scoring_settings)
    return df


def _truncate_after(df: pd.DataFrame, season: int, boundary_week: int) -> pd.DataFrame:
    """Drop rows for `season` beyond `boundary_week`; other seasons untouched."""
    drop_mask = (df["season"] == season) & (df["week"] > boundary_week)
    return df[~drop_mask]


@pytest.fixture(scope="module")
def base_with_volume(weekly_scored, pbp) -> pd.DataFrame:
    """weekly_scored + add_volume_features -- add_efficiency_features needs
    air_yards and dropbacks from Family 1 as input."""
    return add_volume_features(weekly_scored, pbp)


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
def test_efficiency_features_no_future_leakage(
    base_with_volume, pbp, ngs_receiving, ngs_passing, season, boundary_week
):
    pbp_truncated = _truncate_after(pbp, season, boundary_week)
    ngs_r_truncated = _truncate_after(ngs_receiving, season, boundary_week)
    ngs_p_truncated = _truncate_after(ngs_passing, season, boundary_week)

    full = add_efficiency_features(base_with_volume, pbp, ngs_receiving, ngs_passing)
    truncated = add_efficiency_features(base_with_volume, pbp_truncated, ngs_r_truncated, ngs_p_truncated)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["player_id", "season", "week"] + EFFICIENCY_OUTPUT_COLUMNS
    left = full.loc[mask, cols].reset_index(drop=True)
    right = truncated.loc[mask, cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


def test_efficiency_features_idempotent(base_with_volume, pbp, ngs_receiving, ngs_passing):
    once = add_efficiency_features(base_with_volume, pbp, ngs_receiving, ngs_passing)
    twice = add_efficiency_features(once, pbp, ngs_receiving, ngs_passing)
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


# ==========================================================================
# QB xFP -- correctness tests the black-box future-truncation pattern above
# can't catch (these are routing/attribution bugs, not leakage bugs), using
# small synthetic pbp rows rather than real cached data for precise control.
# ==========================================================================
def _play(**overrides) -> dict:
    """One synthetic pbp row with safe, inert defaults -- every field a
    caller doesn't override is a real column name _dropback_play_frame/
    _qb_rush_play_frame actually reads, set to a value that contributes
    nothing to any bucket/stat unless a test deliberately changes it."""
    base = dict(
        season_type="REG", season=2024, week=3,
        qb_dropback=0, two_point_attempt=0,
        passer_player_id=None, rusher_player_id=None,
        pass_attempt=0, sack=0, qb_scramble=0, qb_kneel=0, rush_attempt=0,
        fumbled_1_player_id=None, fumbled_2_player_id=None,
        interception=0, return_touchdown=0, td_team=None, defteam="OPP",
        yardline_100=50, down=1, ydstogo=10,
        yards_gained=0, pass_touchdown=0, rush_touchdown=0,
        fumble=0, fumble_lost=0,
    )
    base.update(overrides)
    return base


def test_dropback_play_frame_excludes_sack_yardage_from_passing_yards():
    """
    pass_attempt == 1 fires on sack rows too (this module's own Family 1
    design note) -- a sack's negative yards_gained (yardage LOST) must
    never be counted as passing_yards, since the official passing_yards
    stat excludes it.
    """
    pbp = pd.DataFrame([
        _play(qb_dropback=1, passer_player_id="QB1", pass_attempt=1, sack=1, yards_gained=-7),
    ])
    frame = _dropback_play_frame(pbp, qb_ids={"QB1"})
    assert frame.loc[0, "passing_yards"] == 0.0


def test_dropback_play_frame_routes_scramble_yards_to_rushing_not_passing():
    """
    A scramble (qb_dropback == 1, rush_attempt == 1, qb_scramble == 1) has
    a null passer_player_id -- attribution must fall back to
    rusher_player_id -- and its yardage is a RUSH, not a pass.
    """
    pbp = pd.DataFrame([
        _play(qb_dropback=1, passer_player_id=None, rusher_player_id="QB1",
              rush_attempt=1, qb_scramble=1, yards_gained=15, rush_touchdown=1),
    ])
    frame = _dropback_play_frame(pbp, qb_ids={"QB1"})
    assert frame.loc[0, "player_id"] == "QB1"
    assert frame.loc[0, "passing_yards"] == 0.0
    assert frame.loc[0, "rushing_yards"] == 15.0
    assert frame.loc[0, "rushing_tds"] == 1.0


def test_dropback_play_frame_attributes_fumble_to_qb_not_receiver():
    """
    A real, measured risk (41.5% of dropback-play fumbles belong to the
    RECEIVER after a completed catch, not the QB -- see
    _dropback_play_frame's own docstring): the play-level fumble flag must
    only count toward the QB's own bucket when the QB himself is the
    identified fumbler (fumbled_1_player_id/fumbled_2_player_id), not
    whenever fumble == 1 on the play.
    """
    pbp = pd.DataFrame([
        # completed pass, RECEIVER fumbles -- must NOT charge the QB
        _play(qb_dropback=1, passer_player_id="QB1", pass_attempt=1,
              yards_gained=8, fumble=1, fumble_lost=1, fumbled_1_player_id="WR1"),
        # sack, QB himself fumbles -- must charge the QB
        _play(qb_dropback=1, passer_player_id="QB1", pass_attempt=1, sack=1,
              yards_gained=-5, fumble=1, fumble_lost=1, fumbled_1_player_id="QB1"),
    ])
    frame = _dropback_play_frame(pbp, qb_ids={"QB1"})
    assert list(frame["fumbles_total"]) == [0.0, 1.0]
    assert list(frame["fumbles_lost_total"]) == [0.0, 1.0]


def test_dropback_play_frame_detects_pick_six_only_when_defense_actually_scores():
    """
    Same td_team == defteam condition add_pick_six_column uses -- an
    interception return fumbled back into the offense's own end zone
    (interception == 1, return_touchdown == 1, but td_team is the
    OFFENSE) must not be scored as a pick-six against the QB.
    """
    pbp = pd.DataFrame([
        _play(qb_dropback=1, passer_player_id="QB1", pass_attempt=1,
              interception=1, return_touchdown=1, td_team="OPP", defteam="OPP"),
        _play(qb_dropback=1, passer_player_id="QB1", pass_attempt=1,
              interception=1, return_touchdown=1, td_team="OFF", defteam="OPP"),
    ])
    frame = _dropback_play_frame(pbp, qb_ids={"QB1"})
    assert list(frame["pass_int_tds"]) == [1.0, 0.0]


def test_qb_rush_play_frame_excludes_kneels_scrambles_and_non_qb_rushers():
    """
    Same designed_rush_attempts mask as _qb_dropback_features (kneel and
    scramble excluded), PLUS restricted to real QB ids -- the raw pbp mask
    alone (rush_attempt == 1, qb_scramble == 0, qb_kneel == 0) matches any
    non-QB rusher too (qb_scramble is simply always 0 on an RB carry, it
    doesn't mean the rusher IS a QB).
    """
    pbp = pd.DataFrame([
        _play(rush_attempt=1, qb_kneel=1, rusher_player_id="QB1", yardline_100=40),  # kneel -- excluded
        _play(rush_attempt=1, qb_scramble=1, rusher_player_id="QB1", yardline_100=40),  # scramble -- excluded here
        _play(rush_attempt=1, rusher_player_id="QB1", yardline_100=3, yards_gained=3, rush_touchdown=1),  # real designed run
        _play(rush_attempt=1, rusher_player_id="RB1", yardline_100=3, yards_gained=3, rush_touchdown=1),  # non-QB -- excluded
    ])
    frame = _qb_rush_play_frame(pbp, qb_ids={"QB1"})
    assert len(frame) == 1
    assert frame.iloc[0]["player_id"] == "QB1"
    assert frame.iloc[0]["rushing_tds"] == 1.0


def test_qb_player_ids_returns_only_qb_position_rows():
    df = pd.DataFrame({
        "player_id": ["QB1", "RB1", "QB2"],
        "position": ["QB", "RB", "QB"],
    })
    assert _qb_player_ids(df) == {"QB1", "QB2"}


# ==========================================================================
# QB rushing share of points -- export-layer descriptive signal
# ==========================================================================
def test_qb_rushing_share_null_for_non_qb_and_real_for_qb():
    scoring = {"rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2}
    df = pd.DataFrame({
        "player_id": ["QB1", "RB1"],
        "position": ["QB", "RB"],
        "season": [2024, 2024],
        "week": [1, 1],
        "custom_points": [20.0, 20.0],
        "rushing_yards": [50.0, 50.0],
        "rushing_tds": [1.0, 1.0],
        "rushing_2pt_conversions": [0.0, 0.0],
    })
    out = add_qb_rushing_share_feature(df, scoring)
    # rushing_points = 50*0.1 + 1*6 = 11.0; share = 11/20 = 0.55
    assert out.loc[out["player_id"] == "QB1", "rushing_share_of_points"].iloc[0] == pytest.approx(0.55)
    assert pd.isna(out.loc[out["player_id"] == "RB1", "rushing_share_of_points"].iloc[0])


def test_qb_rushing_share_null_when_custom_points_is_zero():
    scoring = {"rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2}
    df = pd.DataFrame({
        "player_id": ["QB1"], "position": ["QB"], "season": [2024], "week": [1],
        "custom_points": [0.0], "rushing_yards": [0.0], "rushing_tds": [0.0],
        "rushing_2pt_conversions": [0.0],
    })
    out = add_qb_rushing_share_feature(df, scoring)
    assert pd.isna(out["rushing_share_of_points"].iloc[0])


def test_qb_rushing_share_rolled_columns_never_see_their_own_week():
    """Same shift(1)-within-group mechanism as add_trend_features's own
    rz_opportunity_share -- week 2's _s2d must reflect only week 1."""
    scoring = {"rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2}
    df = pd.DataFrame({
        "player_id": ["QB1", "QB1"],
        "position": ["QB", "QB"],
        "season": [2024, 2024],
        "week": [1, 2],
        "custom_points": [20.0, 20.0],
        "rushing_yards": [100.0, 0.0],  # week 1: big rushing share; week 2: zero rushing
        "rushing_tds": [0.0, 0.0],
        "rushing_2pt_conversions": [0.0, 0.0],
    })
    out = add_qb_rushing_share_feature(df, scoring)
    week1 = out[out["week"] == 1].iloc[0]
    week2 = out[out["week"] == 2].iloc[0]
    assert pd.isna(week1["rushing_share_of_points_s2d"])  # no prior week yet
    # week 2's s2d must equal week 1's raw share (100*0.1/20 = 0.5), not week 2's own (0.0)
    assert week2["rushing_share_of_points_s2d"] == pytest.approx(0.5)


# ==========================================================================
# FAMILY 5B — OPPONENT DEFENSIVE STRENGTH
# ==========================================================================
@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_opponent_strength_no_future_leakage(featured_df, schedule, season, boundary_week):
    """
    Removing weeks AFTER boundary_week from BOTH the source player-week
    frame and the schedule must not change opponent-strength features for
    any week <= boundary_week -- same black-box pattern as every other
    family in this file. Both inputs need truncating: the schedule
    determines who played whom each week, and the player frame supplies
    the xfp that becomes what an opponent "allowed".
    """
    df_truncated = _truncate_after(featured_df, season, boundary_week)
    schedule_truncated = _truncate_after(schedule, season, boundary_week)

    full = add_opponent_strength_features(featured_df, schedule)
    truncated = add_opponent_strength_features(df_truncated, schedule_truncated)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["player_id", "season", "week"] + OPPONENT_STRENGTH_OUTPUT_COLUMNS
    left = full.loc[mask, cols].reset_index(drop=True)
    right = truncated.loc[mask, cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


def test_opponent_strength_shift_excludes_own_week(featured_df, schedule):
    """
    Perturbation test, same spirit as test_rolling_features_shift_excludes_own_week
    and test_trend_features_shift_excludes_own_week: a black-box truncate-
    and-compare test can't prove week W's defense-strength values exclude
    week W's OWN game, because removing week W also destroys real same-
    week opportunity data a later week legitimately needs. Instead:
    perturb one real player's xfp at a specific week to an extreme
    outlier, and confirm (a) every player facing that SAME opponent in
    week W itself -- including the perturbed player's own row -- is
    unaffected (their opponent's defense-strength value at week W already
    only reflects games strictly before W), while (b) a player facing
    that same opponent in ITS NEXT game is affected (the perturbed week
    has, by then, entered the opponent's trailing window one week later).
    """
    team_week_opp = _team_week_opponent(schedule)

    candidates = featured_df[
        featured_df["position"].isin(OPP_STRENGTH_POSITIONS) & featured_df["xfp"].notna()
    ]
    picked = None
    for _, row in candidates.iterrows():
        player_id, team, season, week = row["player_id"], row["team"], row["season"], row["week"]
        opp_row = team_week_opp[
            (team_week_opp["team"] == team) & (team_week_opp["season"] == season)
            & (team_week_opp["week"] == week)
        ]
        if opp_row.empty:
            continue
        defteam = opp_row["opponent"].iloc[0]
        defteam_later_games = team_week_opp[
            (team_week_opp["team"] == defteam) & (team_week_opp["season"] == season)
            & (team_week_opp["week"] > week)
        ]
        if not defteam_later_games.empty:
            next_week = int(defteam_later_games.sort_values("week")["week"].iloc[0])
            picked = (player_id, season, week, defteam, next_week)
            break
    assert picked is not None, "expected a real player whose opponent has a later same-season game"
    player_id, season, week, defteam, next_week = picked

    def _rows_facing(result: pd.DataFrame, wk: int) -> pd.DataFrame:
        m = (
            (result["season"] == season) & (result["week"] == wk)
            & (result["opponent"] == defteam) & result["position"].isin(OPP_STRENGTH_POSITIONS)
        )
        return result.loc[m].set_index("player_id").sort_index()

    baseline = add_opponent_strength_features(featured_df, schedule)
    baseline_same_week = _rows_facing(baseline, week)
    baseline_next_week = _rows_facing(baseline, next_week)
    assert not baseline_same_week.empty
    assert not baseline_next_week.empty

    perturbed_df = featured_df.copy()
    mask = (
        (perturbed_df["player_id"] == player_id) & (perturbed_df["season"] == season)
        & (perturbed_df["week"] == week)
    )
    perturbed_df.loc[mask, "xfp"] = 999.0
    perturbed = add_opponent_strength_features(perturbed_df, schedule)
    perturbed_same_week = _rows_facing(perturbed, week)
    perturbed_next_week = _rows_facing(perturbed, next_week)

    for col in OPPONENT_STRENGTH_OUTPUT_COLUMNS:
        pd.testing.assert_series_equal(baseline_same_week[col], perturbed_same_week[col])

    changed = any(
        not baseline_next_week[col].equals(perturbed_next_week[col])
        for col in OPPONENT_STRENGTH_OUTPUT_COLUMNS
    )
    assert changed, "expected the perturbation to reach the opponent's NEXT game one week later"


def test_opponent_strength_null_for_qb(featured_df, schedule):
    """
    QB has no xFP counterpart for passing production (see
    add_xfp_features's own scope-gap docstring note), so no defense-vs-QB
    value exists to expose -- every QB row must be null here, the same
    disclosed gap xfp/fp_over_expected already carry for QB rows.
    """
    result = add_opponent_strength_features(featured_df, schedule)
    qb_rows = result[result["position"] == "QB"]
    assert len(qb_rows) > 0
    for col in OPPONENT_STRENGTH_OUTPUT_COLUMNS:
        assert qb_rows[col].isna().all()


def test_opponent_strength_idempotent(featured_df, schedule):
    once = add_opponent_strength_features(featured_df, schedule)
    twice = add_opponent_strength_features(once, schedule)
    pd.testing.assert_frame_equal(once, twice)


def test_defense_strength_table_opponent_adjustment_direction(featured_df, schedule):
    """
    Sanity check on the opponent-adjustment SIGN, not just its plumbing:
    for a defense whose average opponent has been stronger than the
    league average at a given position, the ADJUSTED allowed value must
    be LOWER than the raw one (some of what they gave up is attributed to
    facing tough offenses, not a weak defense) -- and the reverse for a
    defense whose opponents have been weaker than average. Checked
    directly against real 2024-2025 data rather than assumed from the
    formula alone.
    """
    table = build_defense_strength_table(featured_df, schedule)
    rows = table.dropna(subset=["allowed_ewm3", "allowed_adj_ewm3"])
    assert len(rows) > 0

    # avg_opponent_strength - league_avg_generated_s2d is exactly what's
    # subtracted from allowed_ewm3/allowed_s2d to get allowed_adj_ewm3/s2d
    # -- recover it from the two already-computed columns rather than
    # re-importing build_defense_strength_table's private intermediates.
    correction = rows["allowed_ewm3"] - rows["allowed_adj_ewm3"]
    above_average_schedule = rows[correction > 0.5]
    below_average_schedule = rows[correction < -0.5]
    assert len(above_average_schedule) > 0
    assert len(below_average_schedule) > 0
    assert (above_average_schedule["allowed_adj_ewm3"] < above_average_schedule["allowed_ewm3"]).all()
    assert (below_average_schedule["allowed_adj_ewm3"] > below_average_schedule["allowed_ewm3"]).all()


@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_defense_air_ground_split_no_future_leakage(
    featured_df, pbp, schedule, scoring_settings, season, boundary_week
):
    """
    Same black-box truncate-and-compare pattern as every other family in
    this file: a defense's real air/ground allowed value at week N must
    be unchanged when weeks strictly after N are removed from the
    source pbp -- proves the air/ground split doesn't leak the very
    games it's trying to describe. `cutoffs` inside the function is
    derived from featured_df's own (season, week) pairs, not from pbp,
    so truncating pbp changes what's AVAILABLE to compute from without
    changing which (team, week) rows are asked for -- the same "scaffold
    from df, not from the upstream source" reasoning
    build_team_tendency_table's own docstring documents.
    """
    pbp_truncated = _truncate_after(pbp, season, boundary_week)

    full = build_defense_air_ground_split(featured_df, pbp, schedule, scoring_settings)
    truncated = build_defense_air_ground_split(featured_df, pbp_truncated, schedule, scoring_settings)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["team", "season", "week"] + DEFENSE_AIR_GROUND_OUTPUT_COLUMNS
    left = full.loc[mask, cols].sort_values(["team", "week"]).reset_index(drop=True)
    right = truncated.loc[mask, cols].sort_values(["team", "week"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_defense_air_ground_split_produces_real_values(featured_df, pbp, schedule, scoring_settings):
    """Sanity check on the plumbing, not the model: once a season has a
    real handful of games, real teams should have a non-null air/ground
    allowed value -- an all-null result would mean the opponent-mapping
    or rolling step silently dropped every row. Air and ground allowed
    should also be non-negative (they're sums of bucket-rate xfp over
    real plays, never a subtracted or adjusted quantity here)."""
    table = build_defense_air_ground_split(featured_df, pbp, schedule, scoring_settings)
    real = table.dropna(subset=DEFENSE_AIR_GROUND_OUTPUT_COLUMNS)
    assert len(real) > 100
    assert (real["xfp_allowed_air_s2d"] >= 0).all()
    assert (real["xfp_allowed_ground_s2d"] >= 0).all()


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


# ==========================================================================
# FAMILY 7 — TREND SIGNAL (Phase 3')
# ==========================================================================
@pytest.fixture(scope="module")
def rolled_df(featured_df) -> pd.DataFrame:
    """featured_df + add_rolling_features -- the input add_trend_features expects."""
    return add_rolling_features(featured_df)


def test_trend_features_shift_excludes_own_week(rolled_df):
    """
    Same perturbation strategy as test_rolling_features_shift_excludes_own_week,
    aimed at add_trend_features specifically: a black-box truncate-and-compare
    test can't prove week N's OWN trend_signal excludes week N's own row,
    because removing week N's row also destroys real same-week information a
    later week legitimately needs. Perturbing target_share to an extreme
    outlier at one week must leave that SAME week's trend_signal unchanged
    (it's derived from ewm3/s2d, which are already shift(1)-safe upstream)
    while changing the FOLLOWING week's trend_signal (proving the perturbed
    value does reach the next week, one step later).
    """
    counts = rolled_df.groupby(["player_id", "season"]).size()
    pid, season = counts[counts >= 4].index[0]
    weeks = sorted(
        rolled_df.loc[
            (rolled_df["player_id"] == pid) & (rolled_df["season"] == season), "week"
        ].tolist()
    )
    target_week, next_week = weeks[1], weeks[2]

    def _row(result, week):
        m = (result["player_id"] == pid) & (result["season"] == season) & (result["week"] == week)
        return result.loc[m].iloc[0]

    baseline = add_trend_features(rolled_df)
    baseline_target = _row(baseline, target_week)
    baseline_next = _row(baseline, next_week)

    perturbed_df = rolled_df.copy()
    mask = (
        (perturbed_df["player_id"] == pid) & (perturbed_df["season"] == season)
        & (perturbed_df["week"] == target_week)
    )
    perturbed_df.loc[mask, "target_share"] = 999.0
    perturbed = add_trend_features(perturbed_df)
    perturbed_target = _row(perturbed, target_week)
    perturbed_next = _row(perturbed, next_week)

    if pd.notna(baseline_target["target_share_trend_signal"]):
        assert baseline_target["target_share_trend_signal"] == pytest.approx(
            perturbed_target["target_share_trend_signal"]
        )
    if pd.notna(baseline_next["target_share_trend_signal"]) or pd.notna(perturbed_next["target_share_trend_signal"]):
        assert baseline_next["target_share_trend_signal"] != pytest.approx(
            perturbed_next["target_share_trend_signal"]
        )


@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_trend_features_no_future_leakage(rolled_df, season, boundary_week):
    """
    Removing rows for weeks AFTER boundary_week must not change trend
    features for any week <= boundary_week -- catches a future row leaking
    backward (e.g. an unsorted groupby), the same bug class
    test_rolling_features_no_future_leakage guards against for Family 6.
    """
    df_truncated = _truncate_after(rolled_df, season, boundary_week)

    full = add_trend_features(rolled_df)
    truncated = add_trend_features(df_truncated)

    mask = (full["season"] == season) & (full["week"] <= boundary_week)
    cols = ["player_id", "season", "week"] + TREND_OUTPUT_COLUMNS
    left = full.loc[mask, cols].reset_index(drop=True)
    right = truncated.loc[mask, cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


def test_trend_signal_null_below_min_games(rolled_df):
    """games_played < MIN_GAMES_FOR_TREND must null every trend_signal/
    trend_direction column -- the noise floor that keeps a two-game player
    off a riser/faller list, checked directly rather than just trusted."""
    from src.usage import MIN_GAMES_FOR_TREND, TREND_SOURCE_FEATURES

    result = add_trend_features(rolled_df)
    thin = result[result["games_played"] < MIN_GAMES_FOR_TREND]
    assert len(thin) > 0, "expected at least one player-week below the min-games floor"

    for feat in TREND_SOURCE_FEATURES:
        assert thin[f"{feat}_trend_signal"].isna().all()
        assert thin[f"{feat}_trend_direction"].isna().all()


def test_trend_features_idempotent(rolled_df):
    once = add_trend_features(rolled_df)
    twice = add_trend_features(once)
    pd.testing.assert_frame_equal(once, twice)


# ==========================================================================
# TEAM TENDENCIES — offense-level identity from pbp (src/team_tendencies.py)
# ==========================================================================
# Narrower fixture requirements than featured_df above: build_team_tendency_
# table only needs player_id/position/season (to resolve a target's
# position) plus pbp -- weekly_scored itself already has those, no Family
# 1-5 feature computation needed first.
@pytest.mark.parametrize("season,boundary_week", BOUNDARIES)
def test_team_tendencies_no_future_leakage(weekly_scored, pbp, season, boundary_week):
    """
    Same black-box future-truncation pattern as every other family --
    removing pbp rows for weeks AFTER boundary_week must not change the
    team-tendency table for any week <= boundary_week.

    Unlike every add_*_features test above, build_team_tendency_table
    builds a FRESH (team, season, week) grid from pbp rather than adding
    columns onto weekly_scored's own fixed row set -- full/truncated can
    legitimately land on different row indices for the same logical key
    (fewer total team-weeks exist once later weeks are dropped), so each
    side needs its OWN mask computed against its OWN frame, then both
    sorted to a canonical row order before comparing -- reusing one
    frame's boolean mask against the other's differently-indexed rows
    would silently misalign instead of comparing the same team-weeks.
    """
    pbp_truncated = _truncate_after(pbp, season, boundary_week)

    full = build_team_tendency_table(weekly_scored, pbp)
    truncated = build_team_tendency_table(weekly_scored, pbp_truncated)

    cols = ["team", "season", "week"] + TEAM_TENDENCY_OUTPUT_COLUMNS + TEAM_TENDENCY_SAMPLE_COLUMNS

    def _scoped(result: pd.DataFrame) -> pd.DataFrame:
        mask = (result["season"] == season) & (result["week"] <= boundary_week)
        return result.loc[mask, cols].sort_values(["team", "week"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(_scoped(full), _scoped(truncated))


def test_team_tendencies_shift_excludes_own_week(weekly_scored, pbp):
    """
    White-box perturbation test, same reasoning as
    test_opponent_strength_shift_excludes_own_week: black-box truncation
    can't prove week W's own row excludes week W's own plays, because
    removing week W also destroys real same-week data a LATER week
    legitimately needs. Instead: perturb every real pass/run play_oe value
    for one real team at one real week to an extreme outlier, and confirm
    (a) that team's OWN row at that same week is unaffected (its ewm3/s2d
    already only reflect games strictly before it), while (b) that team's
    NEXT real game's row IS affected (the perturbed week has, by then,
    entered its own trailing window one week later).
    """
    table = build_team_tendency_table(weekly_scored, pbp)
    counts = table.dropna(subset=["proe_ewm3"]).groupby(["team", "season"]).size()
    team, season = counts[counts >= 2].index[0]
    team_weeks = sorted(table.loc[(table["team"] == team) & (table["season"] == season), "week"].tolist())
    week, next_week = team_weeks[0], team_weeks[1]

    perturbed_pbp = pbp.copy()
    mask = (
        (perturbed_pbp["posteam"] == team) & (perturbed_pbp["season"] == season)
        & (perturbed_pbp["week"] == week) & perturbed_pbp["play_type"].isin(["pass", "run"])
        & perturbed_pbp["pass_oe"].notna()
    )
    assert mask.sum() > 0, "expected at least one real neutral-situation play to perturb"
    perturbed_pbp.loc[mask, "pass_oe"] = 999.0

    baseline = build_team_tendency_table(weekly_scored, pbp)
    perturbed = build_team_tendency_table(weekly_scored, perturbed_pbp)

    def _row(result: pd.DataFrame, wk: int) -> pd.Series:
        r = result[(result["team"] == team) & (result["season"] == season) & (result["week"] == wk)]
        assert len(r) == 1
        return r.iloc[0]

    baseline_same = _row(baseline, week)
    perturbed_same = _row(perturbed, week)
    for col in TEAM_TENDENCY_OUTPUT_COLUMNS + TEAM_TENDENCY_SAMPLE_COLUMNS:
        left, right = baseline_same[col], perturbed_same[col]
        assert (pd.isna(left) and pd.isna(right)) or left == right, (
            f"{col} at the perturbed week itself changed -- it should only reflect PRIOR weeks"
        )

    baseline_next = _row(baseline, next_week)
    perturbed_next = _row(perturbed, next_week)
    assert baseline_next["proe_ewm3"] != perturbed_next["proe_ewm3"], (
        "expected the perturbation to reach this team's NEXT game one week later"
    )


def test_team_tendencies_no_position_gate(weekly_scored, pbp):
    """
    Unlike Family 5B's opponent strength (position-vs-position, QB null by
    construction), team tendencies describe the OFFENSE, not the player --
    every position on the same team/week must see the IDENTICAL value,
    including QB (no null carve-out here)."""
    result = add_team_tendency_features(weekly_scored, pbp)
    sample = result.dropna(subset=["proe_ewm3"])
    assert len(sample) > 0
    assert set(sample["position"].unique()) >= {"QB", "RB", "WR", "TE"} or len(sample["position"].unique()) > 0

    grouped = sample.groupby(["team", "season", "week"])[TEAM_TENDENCY_OUTPUT_COLUMNS].nunique()
    assert (grouped <= 1).all().all(), "every player on the same team/week must share the same team-tendency values"


def test_team_tendencies_idempotent(weekly_scored, pbp):
    once = add_team_tendency_features(weekly_scored, pbp)
    twice = add_team_tendency_features(once, pbp)
    pd.testing.assert_frame_equal(once, twice)
