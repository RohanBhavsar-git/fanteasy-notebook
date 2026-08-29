"""
FanTeasy Stats — Usage and efficiency features (Phase 2b)

Steps 1-4 of PHASE_2B_6_SPEC.md's Order of Work, plus Family 3 (efficiency --
built out of chronological order, added when steps 1-2 turned out to have
skipped it): volume/share, snap-share, efficiency, situational, game-context,
xFP, and rolling-aggregate feature families. Keep this separate from
features.py, which owns scoring.

Design note on denominators:
    Team-level denominators (team targets, team air yards, team carries,
    team receptions) are computed from the FULL pbp -- every offensive
    play, not just QB/RB/WR/TE rows -- then joined onto the (already
    position-filtered) input frame by (team, season, week). Computing
    denominators from the filtered frame itself would silently undercount
    them (a target to a FB or a trick-play completion to a lineman would
    vanish from the team total), inflating every share in that game.

Design note on target_share's denominator:
    The spec text says "targets / team pass attempts". Taken literally,
    "team pass attempts" includes sacks, spikes, and throwaways/broken-up
    passes with no eligible receiver -- none of which can be anyone's
    target. Using that as the denominator makes target_share sum to
    *less* than 1.0 per team-week, which contradicts this phase's own
    acceptance criteria ("target shares sum to roughly 1.0 per
    team-week"). This module uses the standard fantasy-analytics
    definition instead: targets / team TARGETS (pass attempts with a
    recorded receiver_player_id). Verified this sums to exactly 1.0 per
    team-week -- see 03_usage_features.ipynb.

Design note on "attempts" vs pbp's pass_attempt column:
    In this nflverse pbp snapshot, pass_attempt == 1 on sack rows too, so
    it does NOT match the official "attempts" stat on its own. Verified
    against Josh Allen, 2024 Wk 1: weekly attempts == 23, and
    (pass_attempt == 1) & (sack == 0) also == 23, while raw pass_attempt
    == 1 alone gives 25. Team pass-attempt-shaped aggregates in this
    module use that (pass_attempt == 1) & (sack == 0) definition; player-
    level numerators are pulled from weekly_scored's own attempts/carries/
    targets/receptions columns rather than re-derived, since those are
    already the validated official-stat columns.

Design note on QB dropback/scramble attribution:
    On scramble rows, pbp leaves passer_player_id null and only populates
    rusher_player_id -- verified 2,433/2,433 scramble rows have a null
    passer_player_id. So a QB's dropback total has to be attributed via
    passer_player_id OR (when that's null) rusher_player_id, restricted
    to qb_dropback == 1 rows. designed_rush_attempts / dropbacks /
    scramble_rate are only meaningful for QBs; they're left null for
    RB/WR/TE rows rather than computed (a WR has no "dropbacks").

    designed_rush_attempts excludes qb_kneel plays -- verified 2024 Wk1
    Kyle Trask took 2 clock-killing kneels that set rush_attempt == 1 in
    this pbp snapshot, which would otherwise count as 2 "designed" rushes
    for a QB who never ran a real called play.

    Null vs. zero for dropbacks/designed_rush_attempts (QB rows only):
    a QB row in weekly_scored means the player appeared that week, so a
    QB who matches none of the three underlying event types (e.g. a
    kneel-only relief appearance) gets dropbacks = 0 and
    designed_rush_attempts = 0, not null -- the value is known, it's zero.
    scramble_rate is computed AFTER that zero-fill, so a true 0-dropback
    week divides 0/0 -> null: a rate has no defined value with a zero
    denominator. Null is reserved for RB/WR/TE rows, where the concept
    doesn't apply at all.

Design note on situational counts (add_situational_features):
    Same null-vs-zero reasoning as the QB dropback fields above, applied
    to every position in scope: rz_targets, rz_carries, inside_10_touches,
    and inside_5_carries are 0 (not null) for any QB/RB/WR/TE row, because
    a row existing in df already means the player appeared that week.
    Only the SHARE columns can be genuinely null, and only when the team
    recorded zero of that situation all week (e.g. never threw during a
    two-minute situation) -- a real absence of data, not a bug.

Design note on spread_line's sign (add_context_features):
    Verified empirically against real results, not assumed: nflverse's
    spread_line is POSITIVE when the HOME team is favored (opposite of
    the sportsbook "-7 favorite" notation). See _team_week_context's
    docstring for the two games used to confirm this.

Design note on xFP's rate table (add_xfp_features):
    See that function's own docstring for the full reasoning. Short
    version: the bucket rate for week N is an EXPANDING, GLOBAL average
    over every target/carry (or, for QB, dropback/designed-rush) strictly
    before week N -- crossing season boundaries deliberately, since two
    seasons of pbp isn't enough to reset the clock every September. QB
    xfp covers dropbacks (the full realized value of a dropback --
    completion, incompletion, interception, sack, or scramble) and
    designed rush attempts, using the exact same bucket-rate machinery as
    RB/WR/TE's target/carry buckets, kept as a SEPARATE rate table (a
    dropback and a WR target aren't the same opportunity type).

Design note on the season boundary in rolling aggregates (add_rolling_features):
    See that function's own docstring for the full reasoning. Short
    version: ewm3/vol/s2d/games_played group by (player_id, season)
    together, so they reset at every season boundary -- deliberately the
    OPPOSITE of xFP's rate table, because these describe a player's OWN
    trend (which really does reset with a new team/role/coordinator)
    while xFP's rate table estimates a league-wide constant (which
    doesn't).

Design note on excluding two_point_attempt == 1 from every pbp-built
team-week denominator (Family 1 and Family 4):
    Found in Family 1 via a full audit of every team-week's target_share
    remainder (not just the worst case): weekly_scored's own official
    targets/carries columns -- the numerator for every Family 1 share --
    already exclude two-point-conversion plays, but _team_week_totals's
    team-level denominators didn't, silently deflating every share on any
    team-week with a 2pt attempt (160/1088 team-weeks were affected
    before the fix). Family 4's situational totals had the same 2pt plays
    included on BOTH numerator and denominator (both built from pbp,
    neither from weekly_scored's official columns), so it was never a
    live numerator/denominator mismatch bug there -- but it was accidental
    consistency, not a deliberate choice, and editing just one side later
    (as Family 1's numerator vs. denominator drifted) would have silently
    reintroduced exactly this bug. Fixed in both places on the same
    principle: every pbp-built team-week denominator in this module
    excludes two_point_attempt == 1, full stop.

Usage:
    from src.usage import (
        add_volume_features, add_snap_features, add_efficiency_features,
        add_situational_features, add_context_features,
        add_xfp_features, add_rolling_features, add_trend_features,
        get_usage_trend_leaders,
    )
    df = add_volume_features(weekly_scored, pbp)
    df = add_snap_features(df, snaps, crosswalk)
    df = add_efficiency_features(df, pbp, ngs_receiving, ngs_passing)
    df = add_situational_features(df, pbp)
    df = add_context_features(df, schedule)
    df = add_xfp_features(df, pbp, scoring_settings)
    df = add_rolling_features(df)
    df = add_trend_features(df)  # Phase 3' -- usage trend signal, not a model feature
    risers, fallers = get_usage_trend_leaders(df, season, week, feature="target_share")
"""

from __future__ import annotations

import logging

import pandas as pd

from src.features import compute_custom_score

logger = logging.getLogger(__name__)

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")

VOLUME_OUTPUT_COLUMNS = [
    "air_yards", "target_share", "air_yards_share", "wopr",
    "carry_share", "touches", "touch_share",
    "pass_attempts", "dropbacks", "designed_rush_attempts", "scramble_rate",
]
SNAP_OUTPUT_COLUMNS = ["offense_snaps", "offense_pct"]

# QB-only fields within add_volume_features's output -- null by design for
# every other position, since the concepts (dropback, scramble) don't apply.
_QB_ONLY_COLUMNS = ["dropbacks", "designed_rush_attempts", "scramble_rate"]


# ==========================================================================
# FAMILY 1 — VOLUME AND SHARE
# ==========================================================================
def _team_week_totals(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per (team, season, week): team targets, team air yards (on targeted
    passes only), team carries, team receptions. REG season only.

    Built from the full pbp regardless of position, so shares computed
    against these totals are correct even for teams whose stray targets
    went to a FB or a trick-play passer.

    Excludes two_point_attempt == 1 plays. Found via the full team-week
    target_share audit (every targeted play's remainder should equal
    exactly its out-of-scope-position targets; 160/1088 team-weeks didn't
    reconcile before this fix): weekly_scored's own official `targets`
    and `carries` columns -- the NUMERATOR for every share in this
    module -- already exclude 2-point conversion plays (verified: a
    receiver with 13 raw pbp targets including one 2pt try showed
    `targets == 12` in weekly_scored; a rusher with 17 raw carries
    including one 2pt try showed `carries == 16`). Before this fix, the
    DENOMINATOR here counted 2pt plays that the numerator never counted,
    silently deflating every share on any team-week with a 2pt attempt.
    """
    reg = pbp[(pbp["season_type"] == "REG") & (pbp["two_point_attempt"] == 0)]

    targeted = reg[(reg["pass_attempt"] == 1) & reg["receiver_player_id"].notna()]
    team_targets = (
        targeted.groupby(["posteam", "season", "week"])
        .agg(team_targets=("receiver_player_id", "size"),
             team_air_yards=("air_yards", "sum"))
        .reset_index()
    )

    team_receptions = (
        targeted[targeted["complete_pass"] == 1]
        .groupby(["posteam", "season", "week"])
        .size()
        .reset_index(name="team_receptions")
    )

    team_carries = (
        reg[reg["rush_attempt"] == 1]
        .groupby(["posteam", "season", "week"])
        .size()
        .reset_index(name="team_carries")
    )

    totals = team_targets.merge(
        team_receptions, on=["posteam", "season", "week"], how="outer"
    ).merge(
        team_carries, on=["posteam", "season", "week"], how="outer"
    )
    return totals.rename(columns={"posteam": "team"})


def _qb_dropback_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per (gsis player_id, season, week): dropbacks, designed_rush_attempts,
    scramble_rate. Attribution uses passer_player_id, falling back to
    rusher_player_id on scramble rows where passer_player_id is null.
    """
    reg = pbp[pbp["season_type"] == "REG"].copy()
    reg["qb_id"] = reg["passer_player_id"].where(
        reg["passer_player_id"].notna(), reg["rusher_player_id"]
    )

    dropback_rows = reg[(reg["qb_dropback"] == 1) & reg["qb_id"].notna()]
    dropbacks = (
        dropback_rows.groupby(["qb_id", "season", "week"])
        .size()
        .reset_index(name="dropbacks")
    )
    scrambles = (
        dropback_rows[dropback_rows["qb_scramble"] == 1]
        .groupby(["qb_id", "season", "week"])
        .size()
        .reset_index(name="scrambles")
    )
    # Excludes qb_kneel plays. Verified 2024 Wk1 Kyle Trask: 2 clock-killing
    # kneels set rush_attempt == 1 and qb_scramble == 0 in this pbp
    # snapshot, which would otherwise count as 2 "designed" rush attempts
    # for a QB who never ran a real called play.
    designed = (
        reg[(reg["rush_attempt"] == 1) & (reg["qb_scramble"] == 0)
            & (reg["qb_kneel"] == 0) & reg["rusher_player_id"].notna()]
        .groupby(["rusher_player_id", "season", "week"])
        .size()
        .reset_index(name="designed_rush_attempts")
        .rename(columns={"rusher_player_id": "qb_id"})
    )

    # Outer join, not chained off `dropbacks` as a mandatory base -- a QB
    # who only ran designed plays with zero real dropbacks (rare, but
    # possible for a wildcat-only appearance) still needs a row here.
    # Any (qb_id, season, week) present in this frame means the QB
    # recorded at least one of these three event types -- absence from
    # the frame does NOT mean "unknown", it means "zero of all three".
    # Callers (add_volume_features) are responsible for filling 0 rather
    # than null for QB rows that don't match here at all (e.g. a
    # kneel-only appearance, which now matches none of the three).
    out = dropbacks.merge(
        scrambles, on=["qb_id", "season", "week"], how="outer"
    ).merge(
        designed, on=["qb_id", "season", "week"], how="outer"
    )
    for col in ("dropbacks", "scrambles", "designed_rush_attempts"):
        out[col] = out[col].fillna(0)
    return out.rename(columns={"qb_id": "player_id"})


def add_volume_features(df: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Add target/air-yards/carry share, WOPR, touches, and QB dropback/
    scramble features. One row in, one row out -- no row-count change.

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with at least player_id, position, team,
            season, week, targets, receiving_air_yards, carries,
            receptions, attempts (e.g. weekly_scored.parquet).
        pbp: play-by-play frame from get_pbp(), used only for the
            team-level denominators and QB dropback/scramble attribution.

    Returns:
        Copy of df with VOLUME_OUTPUT_COLUMNS added.
    """
    required = ["player_id", "position", "team", "season", "week",
                "targets", "receiving_air_yards", "carries", "receptions",
                "attempts"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_volume_features: df is missing columns {missing}")

    out = df.drop(columns=[c for c in VOLUME_OUTPUT_COLUMNS if c in df.columns])

    totals = _team_week_totals(pbp)
    out = out.merge(totals, on=["team", "season", "week"], how="left")

    out["air_yards"] = out["receiving_air_yards"]
    out["target_share"] = out["targets"] / out["team_targets"]
    out["air_yards_share"] = out["air_yards"] / out["team_air_yards"]
    out["wopr"] = 1.5 * out["target_share"] + 0.7 * out["air_yards_share"]
    out["carry_share"] = out["carries"] / out["team_carries"]
    out["touches"] = out["carries"] + out["receptions"]
    out["touch_share"] = out["touches"] / (out["team_carries"] + out["team_receptions"])
    out["pass_attempts"] = out["attempts"]

    qb_feats = _qb_dropback_features(pbp)
    out = out.merge(qb_feats, on=["player_id", "season", "week"], how="left")

    # weekly_scored only carries rows for players who recorded activity, so
    # any QB row here already means the player appeared. A QB absent from
    # qb_feats (e.g. a kneel-only appearance) genuinely had zero dropbacks
    # and zero designed rush attempts -- that's known, not unknown -- so
    # fill 0 rather than leaving it null. scramble_rate is computed AFTER
    # this fill so a true 0-dropback week divides 0/0 -> null (a rate has
    # no defined value with a zero denominator), not 0.
    is_qb = out["position"] == "QB"
    zero_fill = ["dropbacks", "scrambles", "designed_rush_attempts"]
    out.loc[is_qb, zero_fill] = out.loc[is_qb, zero_fill].fillna(0)
    out["scramble_rate"] = out["scrambles"] / out["dropbacks"]

    non_qb = ~is_qb
    out.loc[non_qb, _QB_ONLY_COLUMNS] = pd.NA

    return out.drop(columns=["team_targets", "team_air_yards",
                              "team_carries", "team_receptions", "scrambles"])


# ==========================================================================
# FAMILY 2 — SNAP SHARE
# ==========================================================================
def add_snap_features(
    df: pd.DataFrame, snaps: pd.DataFrame, crosswalk: pd.DataFrame
) -> pd.DataFrame:
    """
    Add offense_snaps and offense_pct via the pfr_player_id -> gsis_id
    crosswalk hop. Verified 99.67% match for QB/RB/WR/TE (see
    PROJECT_CONTEXT.md Verification status); the miss is fringe players
    absent from the crosswalk entirely, not a join-key format bug.

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with player_id, season, week.
        snaps: from get_snap_counts() -- pfr_player_id-keyed, per game.
        crosswalk: from get_id_crosswalk() -- has pfr_id and gsis_id.

    Returns:
        Copy of df with SNAP_OUTPUT_COLUMNS added (null where the crosswalk
        hop didn't resolve a gsis_id, e.g. o-line/long-snapper rows, which
        this scope never touches, or the ~0.3% fringe-player miss).
    """
    required = ["player_id", "season", "week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_snap_features: df is missing columns {missing}")
    if "pfr_id" not in crosswalk.columns or "gsis_id" not in crosswalk.columns:
        raise KeyError("add_snap_features: crosswalk is missing pfr_id/gsis_id")

    out = df.drop(columns=[c for c in SNAP_OUTPUT_COLUMNS if c in df.columns])

    cw = crosswalk.dropna(subset=["pfr_id"]).drop_duplicates(subset=["pfr_id"])
    snaps_reg = snaps[snaps["game_type"] == "REG"]

    # A player can appear on TWO teams' snap logs in the same week from a
    # genuine mid-week waiver claim (verified on the 2018-2025 range: Jalen
    # Davis, pfr_player_id DaviJa06, appears for both ARI and MIA in 2019
    # Wk16 -- one real player, not a crosswalk error). Deduplicate on the
    # PFR side, before the crosswalk join, keeping the team he played more
    # snaps for that week -- this has to happen before the collision check
    # below, which exists to catch a DIFFERENT problem (two DISTINCT
    # players mapped to the same gsis_id) and should stay strict for that.
    snaps_reg = snaps_reg.sort_values("offense_snaps", ascending=False).drop_duplicates(
        subset=["pfr_player_id", "season", "week"], keep="first"
    )

    mapped = snaps_reg.merge(
        cw[["pfr_id", "gsis_id"]],
        left_on="pfr_player_id", right_on="pfr_id", how="inner",
    )

    dupes = mapped.duplicated(subset=["gsis_id", "season", "week"], keep=False)
    if dupes.any():
        raise ValueError(
            f"{dupes.sum()} snap rows collide on (gsis_id, season, week) after "
            "the pfr_player_id -> gsis_id join -- two DIFFERENT PFR players "
            "mapped to the same gsis_id (the multi-team-in-one-week case is "
            "already deduplicated above, so this is a real crosswalk problem). "
            "Investigate before trusting snap_share."
        )

    snap_cols = (
        mapped[["gsis_id", "season", "week", "offense_snaps", "offense_pct"]]
        .rename(columns={"gsis_id": "player_id"})
    )

    return out.merge(snap_cols, on=["player_id", "season", "week"], how="left")


# ==========================================================================
# FAMILY 3 — EFFICIENCY
# ==========================================================================
# Skipped in steps 1-2 (an omission, not a scope decision) -- built now,
# out of chronological order but in the spec's own family order (3, before
# situational/context).
EFFICIENCY_OUTPUT_COLUMNS = [
    "adot", "yac_per_reception", "catch_rate", "yards_per_target", "yards_per_carry",
    "cpoe", "epa_per_dropback",
    "avg_separation", "avg_cushion", "time_to_throw",
]


def _qb_efficiency_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per (gsis player_id, season, week): total qb_epa on dropback plays
    (epa_per_dropback's numerator -- the caller divides by df's own
    `dropbacks` column from add_volume_features, rather than recomputing
    a second, potentially-drifting dropback count here) and mean cpoe on
    true pass attempts (a rate already, no further division needed).

    Attribution mirrors _qb_dropback_features: passer_player_id, falling
    back to rusher_player_id on scramble rows where it's null.
    """
    reg = pbp[pbp["season_type"] == "REG"].copy()
    reg["qb_id"] = reg["passer_player_id"].where(
        reg["passer_player_id"].notna(), reg["rusher_player_id"]
    )

    dropback_rows = reg[(reg["qb_dropback"] == 1) & reg["qb_id"].notna()]
    epa_sum = (
        dropback_rows.groupby(["qb_id", "season", "week"])["qb_epa"]
        .sum()
        .reset_index(name="qb_epa_sum")
    )

    cpoe_rows = reg[
        (reg["pass_attempt"] == 1) & reg["passer_player_id"].notna() & reg["cpoe"].notna()
    ]
    cpoe_avg = (
        cpoe_rows.groupby(["passer_player_id", "season", "week"])["cpoe"]
        .mean()
        .reset_index()
        .rename(columns={"passer_player_id": "qb_id"})
    )

    out = epa_sum.merge(cpoe_avg, on=["qb_id", "season", "week"], how="outer")
    return out.rename(columns={"qb_id": "player_id"})


def _ngs_receiving_lookup(ngs_receiving: pd.DataFrame) -> pd.DataFrame:
    """
    Per (gsis player_id, season, week): avg_separation, avg_cushion.
    Excludes week == 0 (nflverse's season-aggregate row, not a real week)
    and non-REG rows. NGS receiving only covers WR/TE -- verified directly
    (player_position has exactly two values, {'TE', 'WR'}) -- so RB rows
    get null here from the join never matching, not from a bug.
    """
    reg = ngs_receiving[
        (ngs_receiving["season_type"] == "REG") & (ngs_receiving["week"] > 0)
    ]
    return reg[["player_gsis_id", "season", "week", "avg_separation", "avg_cushion"]].rename(
        columns={"player_gsis_id": "player_id"}
    )


def _ngs_passing_lookup(ngs_passing: pd.DataFrame) -> pd.DataFrame:
    """
    Per (gsis player_id, season, week): time_to_throw. Excludes week == 0
    and non-REG rows, same as _ngs_receiving_lookup. NGS passing only
    covers QB (verified: player_position == {'QB'} exactly), so this
    never needs an explicit position mask.
    """
    reg = ngs_passing[
        (ngs_passing["season_type"] == "REG") & (ngs_passing["week"] > 0)
    ]
    return reg[["player_gsis_id", "season", "week", "avg_time_to_throw"]].rename(
        columns={"player_gsis_id": "player_id", "avg_time_to_throw": "time_to_throw"}
    )


def add_efficiency_features(
    df: pd.DataFrame, pbp: pd.DataFrame, ngs_receiving: pd.DataFrame, ngs_passing: pd.DataFrame
) -> pd.DataFrame:
    """
    Add per-opportunity efficiency: adot, yac_per_reception, catch_rate,
    yards_per_target, yards_per_carry (all from weekly_scored's own
    official receiving/rushing columns, already present in df -- no need
    to re-derive them from pbp), cpoe and epa_per_dropback for QBs (from
    pbp), and avg_separation/avg_cushion/time_to_throw from NGS.

    Ratio null semantics: every ratio's denominator (targets, receptions,
    carries, dropbacks) is a count column ALREADY in df from an earlier
    step -- callers needing to distinguish "0 opportunities" from "some
    opportunities, ratio genuinely undefined" (there's no such case here;
    see below) can check that column directly, so no companion team_*-
    style count column is added. Every numerator here is derived from the
    exact same opportunities as its denominator (e.g. receiving_yards
    only accumulates from real receptions), so a 0 denominator always
    pairs with a 0 numerator -- plain division gives 0/0 = null, never a
    spurious 0 or inf.

    cpoe/epa_per_dropback are null for every non-QB row (masked
    explicitly, same pattern as Family 1's dropbacks/scramble_rate) --
    not because the ratio is undefined, but because the concept doesn't
    apply to a non-passer.

    NGS coverage: 2016+ only (irrelevant for this project's 2024-2025
    scope, but noted for whenever SEASONS extends further back) and
    sparse even within that range -- avg_separation/avg_cushion cover
    WR/TE only (RB is entirely absent from NGS receiving, verified), and
    a real gap in null rates should be expected and is reported, not
    silently backfilled.

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with player_id, position, season, week,
            targets, receptions, receiving_yards, receiving_yards_after_catch,
            carries, rushing_yards, dropbacks (i.e. df after
            add_volume_features has already run -- dropbacks is required).
        pbp: play-by-play frame from get_pbp().
        ngs_receiving: from get_ngs_data('receiving', seasons).
        ngs_passing: from get_ngs_data('passing', seasons).

    Returns:
        Copy of df with EFFICIENCY_OUTPUT_COLUMNS added.
    """
    required = [
        "player_id", "position", "season", "week", "targets", "receptions",
        "receiving_yards", "receiving_yards_after_catch", "carries",
        "rushing_yards", "dropbacks",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_efficiency_features: df is missing columns {missing}")

    out = df.drop(columns=[c for c in EFFICIENCY_OUTPUT_COLUMNS if c in df.columns])

    out["adot"] = out["air_yards"] / out["targets"]
    out["yac_per_reception"] = out["receiving_yards_after_catch"] / out["receptions"]
    out["catch_rate"] = out["receptions"] / out["targets"]
    out["yards_per_target"] = out["receiving_yards"] / out["targets"]
    out["yards_per_carry"] = out["rushing_yards"] / out["carries"]

    qb_feats = _qb_efficiency_from_pbp(pbp)
    out = out.merge(qb_feats, on=["player_id", "season", "week"], how="left")
    out["epa_per_dropback"] = out["qb_epa_sum"] / out["dropbacks"]
    out = out.drop(columns=["qb_epa_sum"])
    out.loc[out["position"] != "QB", ["cpoe", "epa_per_dropback"]] = pd.NA

    out = out.merge(
        _ngs_receiving_lookup(ngs_receiving), on=["player_id", "season", "week"], how="left"
    )
    out = out.merge(
        _ngs_passing_lookup(ngs_passing), on=["player_id", "season", "week"], how="left"
    )

    return out


# ==========================================================================
# FAMILY 4 — SITUATIONAL
# ==========================================================================
# "Red zone" / "inside 10" / "goal line" are all evaluated at the play's
# field position AT SNAP (yardline_100), not where the play ended -- the
# standard usage-stat convention, and the only choice under which a
# touchdown-scoring target or carry still counts toward the bucket it
# started in.
RZ_YARDLINE = 20
INSIDE_10_YARDLINE = 10
GOAL_LINE_YARDLINE = 5
TWO_MINUTE_SECONDS = 120

SITUATIONAL_OUTPUT_COLUMNS = [
    "rz_targets", "rz_target_share", "team_rz_targets",
    "rz_carries", "rz_carry_share", "team_rz_carries",
    "inside_10_touches",
    "inside_5_carries", "goal_line_carry_share", "team_inside_5_carries",
    "third_down_target_share",
    "two_minute_target_share", "team_two_minute_targets",
]


def _situational_team_totals(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per (team, season, week): team-level denominators for every
    situational share below. Built from the full pbp, not just
    QB/RB/WR/TE targets, for the same reason as _team_week_totals -- a
    red-zone target to a FB still has to count toward the team total.

    Excludes two_point_attempt == 1, matching _team_week_totals and
    xFP's play frames. This family's numerator and denominator were both
    already built from pbp directly (neither reused weekly_scored's
    official columns the way Family 1's numerator did), so including 2pt
    plays consistently on both sides was never a numerator/denominator
    mismatch -- but it was accidental consistency, not a deliberate
    choice, and the exact bug fixed in Family 1 would reappear the moment
    only one side of this pair got edited. One convention, applied
    everywhere a team-week denominator is built from pbp.
    """
    reg = pbp[(pbp["season_type"] == "REG") & (pbp["two_point_attempt"] == 0)]
    targeted = reg[(reg["pass_attempt"] == 1) & reg["receiver_player_id"].notna()]
    rushes = reg[reg["rush_attempt"] == 1]

    def _count(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        return (
            frame.groupby(["posteam", "season", "week"])
            .size()
            .reset_index(name=name)
        )

    totals = _count(targeted[targeted["yardline_100"] <= RZ_YARDLINE], "team_rz_targets")
    totals = totals.merge(
        _count(rushes[rushes["yardline_100"] <= RZ_YARDLINE], "team_rz_carries"),
        on=["posteam", "season", "week"], how="outer",
    ).merge(
        _count(rushes[rushes["yardline_100"] <= GOAL_LINE_YARDLINE], "team_inside_5_carries"),
        on=["posteam", "season", "week"], how="outer",
    ).merge(
        _count(targeted[targeted["down"] == 3], "team_third_down_targets"),
        on=["posteam", "season", "week"], how="outer",
    ).merge(
        _count(targeted[targeted["half_seconds_remaining"] <= TWO_MINUTE_SECONDS],
               "team_two_minute_targets"),
        on=["posteam", "season", "week"], how="outer",
    )
    return totals.rename(columns={"posteam": "team"})


def _situational_player_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per (gsis player_id, season, week): raw situational event counts.
    Absence from this frame means zero of every event type here, not
    unknown -- add_situational_features fills 0 for any QB/RB/WR/TE row,
    since weekly_scored only carries rows for players who appeared.

    Excludes two_point_attempt == 1 -- see _situational_team_totals for
    why. Must stay in sync with that function's exclusion; both changed
    together here.
    """
    reg = pbp[(pbp["season_type"] == "REG") & (pbp["two_point_attempt"] == 0)]
    targeted = reg[(reg["pass_attempt"] == 1) & reg["receiver_player_id"].notna()]
    caught = targeted[targeted["complete_pass"] == 1]
    rushes = reg[(reg["rush_attempt"] == 1) & reg["rusher_player_id"].notna()]

    def _count(frame: pd.DataFrame, id_col: str, name: str) -> pd.DataFrame:
        return (
            frame.groupby([id_col, "season", "week"])
            .size()
            .reset_index(name=name)
            .rename(columns={id_col: "player_id"})
        )

    frames = [
        _count(targeted[targeted["yardline_100"] <= RZ_YARDLINE],
               "receiver_player_id", "rz_targets"),
        _count(rushes[rushes["yardline_100"] <= RZ_YARDLINE],
               "rusher_player_id", "rz_carries"),
        _count(rushes[rushes["yardline_100"] <= INSIDE_10_YARDLINE],
               "rusher_player_id", "inside_10_rush"),
        _count(caught[caught["yardline_100"] <= INSIDE_10_YARDLINE],
               "receiver_player_id", "inside_10_rec"),
        _count(rushes[rushes["yardline_100"] <= GOAL_LINE_YARDLINE],
               "rusher_player_id", "inside_5_carries"),
        _count(targeted[targeted["down"] == 3],
               "receiver_player_id", "third_down_targets"),
        _count(targeted[targeted["half_seconds_remaining"] <= TWO_MINUTE_SECONDS],
               "receiver_player_id", "two_minute_targets"),
    ]
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["player_id", "season", "week"], how="outer")

    count_cols = ["rz_targets", "rz_carries", "inside_10_rush", "inside_10_rec",
                  "inside_5_carries", "third_down_targets", "two_minute_targets"]
    out[count_cols] = out[count_cols].fillna(0)
    out["inside_10_touches"] = out["inside_10_rush"] + out["inside_10_rec"]
    return out.drop(columns=["inside_10_rush", "inside_10_rec"])


def add_situational_features(df: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Add red-zone/inside-10/goal-line volume plus third-down and
    two-minute target share -- the "where touchdowns actually come from"
    features that platforms don't surface.

    goal_line_carry_share reuses the inside_5_carries cutoff (<=5 yards)
    as "goal line" -- share of the team's own goal-line carries, not a
    separately-defined bucket.

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with player_id, position, team, season, week.
        pbp: play-by-play frame from get_pbp().

    Returns:
        Copy of df with SITUATIONAL_OUTPUT_COLUMNS added. Raw counts
        (rz_targets, rz_carries, inside_10_touches, inside_5_carries) are
        0, never null, for any row in df.

        rz_target_share, rz_carry_share, goal_line_carry_share, and
        two_minute_target_share are null when the team recorded zero of
        that situation all week (e.g. never faced a two-minute passing
        situation) -- a share genuinely has no value with a 0/0
        denominator. Each of those four ships with a companion team_*
        count column (team_rz_targets, team_rz_carries,
        team_inside_5_carries, team_two_minute_targets, always 0 not
        null) so a model -- or a person reading the table -- can tell
        "team had zero of these all week" (team_* == 0) apart from
        "player got none of the team's share" (team_* > 0 but the
        player's own count is 0). Null alone can't distinguish those two
        situations. third_down_target_share has no companion column: it
        was never observed null in 2024-2025 (every team-week had at
        least one third-down target), so the ambiguity doesn't arise in
        practice.
    """
    required = ["player_id", "position", "team", "season", "week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_situational_features: df is missing columns {missing}")

    out = df.drop(columns=[c for c in SITUATIONAL_OUTPUT_COLUMNS if c in df.columns])

    totals = _situational_team_totals(pbp)
    out = out.merge(totals, on=["team", "season", "week"], how="left")
    team_count_cols = ["team_rz_targets", "team_rz_carries", "team_inside_5_carries",
                        "team_third_down_targets", "team_two_minute_targets"]
    out[team_count_cols] = out[team_count_cols].fillna(0)

    player_counts = _situational_player_counts(pbp)
    out = out.merge(player_counts, on=["player_id", "season", "week"], how="left")

    count_cols = ["rz_targets", "rz_carries", "inside_10_touches",
                  "inside_5_carries", "third_down_targets", "two_minute_targets"]
    out[count_cols] = out[count_cols].fillna(0)

    # x/0 where x is guaranteed 0 too (a player can't out-count their own
    # team) divides to NaN, not inf -- that's the null we want.
    out["rz_target_share"] = out["rz_targets"] / out["team_rz_targets"]
    out["rz_carry_share"] = out["rz_carries"] / out["team_rz_carries"]
    out["goal_line_carry_share"] = out["inside_5_carries"] / out["team_inside_5_carries"]
    out["third_down_target_share"] = out["third_down_targets"] / out["team_third_down_targets"]
    out["two_minute_target_share"] = out["two_minute_targets"] / out["team_two_minute_targets"]

    return out.drop(columns=["team_third_down_targets", "third_down_targets",
                              "two_minute_targets"])


# ==========================================================================
# FAMILY 5 — GAME CONTEXT
# ==========================================================================
CONTEXT_OUTPUT_COLUMNS = [
    "is_home", "days_rest", "spread", "game_total", "team_implied_total",
    "roof", "surface", "temp", "wind",
]

# Two-way partition of CONTEXT_OUTPUT_COLUMNS for src/model.py's per-
# position feature selection -- add_context_features() itself is
# unaffected and still computes every CONTEXT_OUTPUT_COLUMNS column
# regardless of which subset a given position's model actually trains on.
# Split out because a family-level walk-forward ablation of the whole
# block masked a real, opposite-signed pair of effects at RB (Vegas
# helped, weather hurt, and the two nearly canceled into a false "no
# signal" at the block level) -- see PROJECT_CONTEXT.md's Context Columns
# findings for the full per-position numbers.
VEGAS_SCHEDULE_OUTPUT_COLUMNS = ["is_home", "days_rest", "spread", "game_total", "team_implied_total"]
WEATHER_OUTPUT_COLUMNS = ["roof", "surface", "temp", "wind"]


def _team_week_context(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Per (team, season, week): pregame game-context columns, built by
    stacking the home and away perspective of each schedule row.

    Sign convention verified empirically against schedule's own `result`
    (home_score - away_score), not assumed: `spread_line` is POSITIVE
    when the HOME team is favored. 2024 Wk18 BAL (home) carried
    spread_line = +19.5 vs. CLE and won by 25; 2024 Wk15 NYG (home)
    carried spread_line = -16.5 vs. the actual favorite BAL and lost by
    21. That's nflverse's own convention, not the sportsbook "-7
    favorite" notation -- easy to invert if not checked against real
    results first.

    `spread` in the output is re-signed to each team's OWN perspective
    (positive = this team favored), not always the home team's
    spread_line, so a favorite reads the same whether home or away.
    team_implied_total = (game_total + own_spread) / 2.
    """
    sched = schedule[schedule["game_type"] == "REG"]

    shared = ["season", "week", "total_line", "spread_line",
              "roof", "surface", "temp", "wind"]

    home = sched[["home_team", "home_rest"] + shared].rename(
        columns={"home_team": "team", "home_rest": "days_rest"}
    )
    home["is_home"] = True
    home["spread"] = home["spread_line"]

    away = sched[["away_team", "away_rest"] + shared].rename(
        columns={"away_team": "team", "away_rest": "days_rest"}
    )
    away["is_home"] = False
    away["spread"] = -away["spread_line"]

    out = pd.concat([home, away], ignore_index=True)
    out["game_total"] = out["total_line"]
    out["team_implied_total"] = (out["game_total"] + out["spread"]) / 2
    # Nullable "boolean" (not numpy bool) so a left merge that leaves some
    # rows unmatched fills pd.NA instead of silently upcasting the whole
    # column to generic "object" dtype.
    out["is_home"] = out["is_home"].astype("boolean")

    return out.drop(columns=["total_line", "spread_line"])


def add_context_features(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Add pregame game-context features: is_home, days_rest, spread (this
    team's own signed spread -- positive means favored), game_total,
    team_implied_total, roof, surface, temp, wind.

    Deliberately excludes opponent defensive strength -- that needs
    prior-weeks-only computation and belongs with the rolling features in
    step 4, not here.

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with team, season, week.
        schedule: from get_schedule().

    Returns:
        Copy of df with CONTEXT_OUTPUT_COLUMNS added. temp/wind are null
        for dome/closed-roof games by design (weather doesn't apply
        indoors) plus a small ~2.5% gap on outdoor games where nflverse
        itself has no recorded weather.
    """
    required = ["team", "season", "week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_context_features: df is missing columns {missing}")

    out = df.drop(columns=[c for c in CONTEXT_OUTPUT_COLUMNS if c in df.columns])

    context = _team_week_context(schedule)
    return out.merge(context, on=["team", "season", "week"], how="left")


# ==========================================================================
# EXPECTED FANTASY POINTS (xFP) — step 3
# ==========================================================================
# Bucket boundaries. yardline_100 is distance to the opponent's end zone,
# so smaller = closer to scoring. Evaluated at the play's field position
# at snap, matching the situational-family convention.
TARGET_AIR_YARDS_BINS = [-float("inf"), 0, 10, 20, float("inf")]
TARGET_AIR_YARDS_LABELS = ["behind_los", "0-9", "10-19", "20+"]
TARGET_FIELD_POS_BINS = [-float("inf"), 10, 20, 50, float("inf")]
TARGET_FIELD_POS_LABELS = ["inside_10", "10-20", "20-50", "beyond_50"]
CARRY_FIELD_POS_BINS = [-float("inf"), 5, 10, 20, 50, float("inf")]
CARRY_FIELD_POS_LABELS = ["inside_5", "5-10", "10-20", "20-50", "beyond_50"]

# QB dropback buckets reuse TARGET_FIELD_POS_BINS/LABELS exactly (same
# boundaries -- a dropback's own field position compresses scoring value
# the same way a target's does). The inside_10 band alone is further split
# by an "obvious passing down" flag -- checked empirically against full
# 2018-2025 play counts before deciding, not assumed: mean points per
# dropback differ by ~0.35 between the two there (1.49 standard vs. 1.14
# obvious, both well populated at n=8,832/790), while the identical split
# inside the other three field-position bands moves the mean by at most
# ~0.03 -- noise, not signal. Splitting everywhere the counts merely ALLOW
# it would add bucket sparsity for zero benefit outside the goal line, so
# this only splits where the rate itself demonstrably differs.
QB_OBVIOUS_PASSING_DOWNS = (3, 4)
QB_OBVIOUS_PASSING_DISTANCE = 7

# QB designed-rush buckets reuse CARRY_FIELD_POS_BINS/LABELS exactly (finer
# near the goal line than the dropback bins) -- verified this granularity
# matters, not assumed: mean points per QB designed rush go 0.23
# (beyond_50) -> 0.33 (20-50) -> 0.66 (10-20) -> 1.30 (5-10) -> 3.05
# (inside_5), a real, monotonic, goal-line-driven jump at every step,
# including between the two thinnest bands (5-10 at n=332, inside_5 at
# n=751 across the full 2018-2025 history) -- merging those would blend
# two genuinely different rates, not just tidy up a thin sample.

# Merges decided from full 2024-2025 play counts (see the step-3 report --
# every count and mean was printed before this decision, per the request
# not to silently trust a thin bucket):
#   20+ air yards x inside_10 field position:  0 plays -- structurally
#       near-impossible; you can't throw 20+ air yards starting inside
#       your opponent's 10, there isn't room on the field.
#   10-19 air yards x inside_10:               78 plays -- thin.
#   20+ air yards x 10-20:                     49 plays -- thin.
# All three merge toward midfield, where the deep-ball sample is large,
# giving 10-19|<=20 (913 plays) and 20+|<=50 (1,559 plays). Every other
# target bucket cleared 266 plays on its own. No carry bucket needed
# merging -- the thinnest (inside_5) still had 1,535 plays.
_TARGET_BUCKET_MERGES = {
    "10-19|inside_10": "10-19|<=20",
    "10-19|10-20": "10-19|<=20",
    "20+|inside_10": "20+|<=50",
    "20+|10-20": "20+|<=50",
    "20+|20-50": "20+|<=50",
}

XFP_OUTPUT_COLUMNS = ["xfp", "fp_over_expected"]


def _target_play_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per pass attempt with a recorded receiver (a "target"), with
    ONLY the stat columns compute_custom_score needs to score that single
    play under this league's rules.

    Built as a fresh, minimal DataFrame rather than a copy of the pbp
    slice -- pbp already carries its own play-level passing_yards and
    rushing_yards columns (the PASSER's stat line on that same play), and
    copying the whole slice would hand compute_custom_score a
    "passing_yards" column it never asked to build here. Since pass_yd
    (0.04) is a real active rule, that silently added 0.04 x yards_gained
    to every target's score. Caught by checking a specific player-week's
    per-play sum against their real custom_points -- Nico Collins 2025
    Wk10 scored 22.54 by the buggy version against a real 19.10, and the
    excess divided by yards_gained was exactly 0.04 on every row.

    Excludes two-point conversion tries (278 across 2024-2025) -- always
    thrown from the 2-yard line, which would badly distort the inside-10
    buckets for a play type that isn't a normal-down "opportunity" in the
    same sense.
    """
    mask = (
        (pbp["season_type"] == "REG") & (pbp["pass_attempt"] == 1)
        & pbp["receiver_player_id"].notna() & (pbp["two_point_attempt"] == 0)
    )
    src = pbp.loc[mask]

    air_yards_band = pd.cut(src["air_yards"], TARGET_AIR_YARDS_BINS,
                             labels=TARGET_AIR_YARDS_LABELS, right=False)
    field_pos_band = pd.cut(src["yardline_100"], TARGET_FIELD_POS_BINS,
                             labels=TARGET_FIELD_POS_LABELS, right=True)
    raw_bucket = air_yards_band.astype(str) + "|" + field_pos_band.astype(str)
    bucket = raw_bucket.map(_TARGET_BUCKET_MERGES).fillna(raw_bucket)

    return pd.DataFrame({
        "player_id": src["receiver_player_id"].to_numpy(),
        "season": src["season"].to_numpy(),
        "week": src["week"].to_numpy(),
        "bucket": bucket.to_numpy(),
        "receptions": src["complete_pass"].fillna(0).to_numpy(),
        "receiving_yards": src["yards_gained"].fillna(0).to_numpy(),
        "receiving_tds": src["pass_touchdown"].fillna(0).to_numpy(),
        "receiving_2pt_conversions": 0.0,
        "fumbles_total": src["fumble"].fillna(0).to_numpy(),
        "fumbles_lost_total": src["fumble_lost"].fillna(0).to_numpy(),
    })


def _carry_play_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per rush attempt with a recorded rusher (a "carry"), with
    ONLY the stat columns compute_custom_score needs to score that single
    play. Built as a fresh, minimal DataFrame for the same reason as
    _target_play_frame -- pbp's own receiving_yards/passing_yards columns
    must never reach compute_custom_score for a carry either.

    Excludes qb_kneel plays -- a clock-killing kneel is not a scoring
    opportunity (same reasoning as designed_rush_attempts's kneel
    exclusion above) -- and two-point tries, for the same reason as
    targets.
    """
    mask = (
        (pbp["season_type"] == "REG") & (pbp["rush_attempt"] == 1)
        & pbp["rusher_player_id"].notna() & (pbp["two_point_attempt"] == 0)
        & (pbp["qb_kneel"] == 0)
    )
    src = pbp.loc[mask]

    field_pos_band = pd.cut(src["yardline_100"], CARRY_FIELD_POS_BINS,
                             labels=CARRY_FIELD_POS_LABELS, right=True)

    return pd.DataFrame({
        "player_id": src["rusher_player_id"].to_numpy(),
        "season": src["season"].to_numpy(),
        "week": src["week"].to_numpy(),
        "bucket": field_pos_band.astype(str).to_numpy(),
        "carries": 1.0,
        "rushing_yards": src["yards_gained"].fillna(0).to_numpy(),
        "rushing_tds": src["rush_touchdown"].fillna(0).to_numpy(),
        "rushing_2pt_conversions": 0.0,
        "fumbles_total": src["fumble"].fillna(0).to_numpy(),
        "fumbles_lost_total": src["fumble_lost"].fillna(0).to_numpy(),
    })


def _scored_play_universe(pbp: pd.DataFrame, scoring_settings: dict) -> pd.DataFrame:
    """
    Every target and carry across the full pbp, scored per-play with this
    league's rules and labeled with its (merged) bucket. Long format:
    player_id, season, week, bucket, points -- one row per play.

    This covers targets and carries only, not a QB's own passing
    production or designed rushes -- see _qb_scored_play_universe for
    the QB-specific analogue (a separate rate table, not pooled with this
    one). NOTE: _carry_play_frame's own mask isn't QB-gated, so a QB
    scramble (rush_attempt == 1) can still appear in THIS universe too --
    add_xfp_features drops any QB player_id from this table's contribution
    before merging, so it never competes with the QB-specific value below,
    but the scramble play itself still contributes to the RB/WR/TE-wide
    carry bucket RATE it's pooled into. Not fixed here (a pre-existing
    imprecision, flagged not chased -- see PROJECT_CONTEXT.md's QB xFP
    findings).
    """
    targets = _target_play_frame(pbp)
    targets["points"] = compute_custom_score(targets, scoring_settings, warn=False)

    carries = _carry_play_frame(pbp)
    carries["points"] = compute_custom_score(carries, scoring_settings, warn=False)

    cols = ["player_id", "season", "week", "bucket", "points"]
    return pd.concat([targets[cols], carries[cols]], ignore_index=True)


def _qb_player_ids(df: pd.DataFrame) -> set:
    """
    Distinct player_id values tagged QB anywhere in df. Needed to restrict
    the pbp-derived dropback/designed-rush populations to real QBs: the
    raw pbp masks alone are NOT QB-specific (qb_scramble is a QB-only
    flag, but it being 0 on a row doesn't mean the rusher IS a QB -- it's
    simply always 0 on a non-QB run too), the same reason
    add_volume_features's own designed_rush_attempts column has to be
    nulled for non-QB rows AFTER its merge rather than computed QB-only
    from the start.
    """
    return set(df.loc[df["position"] == "QB", "player_id"].unique())


def _dropback_play_frame(pbp: pd.DataFrame, qb_ids: set) -> pd.DataFrame:
    """
    One row per QB dropback (qb_dropback == 1), restricted to real QBs
    and excluding two-point tries, with ONLY the stat columns
    compute_custom_score needs to score that single play.

    Covers every realized outcome of a dropback -- verified directly
    against full 2018-2025 pbp, dropbacks split cleanly into exactly
    three mutually-exclusive subtypes (plus a negligible 4-row anomaly):
    a real pass attempt (completion, incompletion, or interception --
    pass_attempt == 1, sack == 0), a sack (pass_attempt == 1, sack == 1,
    scores 0 unless the QB himself fumbles), or a scramble (rush_attempt
    == 1, qb_scramble == 1, scores as a rush -- attributed via
    rusher_player_id, since passer_player_id is null on every scramble
    row, verified 7,441/7,441). Sack YARDAGE LOST is never counted as
    passing_yards -- the official stat excludes it, and pass_attempt == 1
    fires on sack rows too (see this module's own Family 1 design note),
    so passing_yards is explicitly gated to real-pass rows only.

    Fumble attribution requires checking WHO fumbled
    (fumbled_1_player_id/fumbled_2_player_id), not just the play-level
    fumble flag: verified 1,097 of 2,641 real fumbles on dropback plays
    (41.5%) belong to the RECEIVER after a completed catch, not the QB --
    attributing the play-level flag blindly (as _target_play_frame/
    _carry_play_frame already safely do, since a target/carry row's
    "owner" IS almost always who fumbled) would incorrectly charge the
    QB for fumbles already being counted against the receiver's own
    bucket. 1,321/1,326 sack fumbles and 100/100 scramble fumbles ARE the
    QB's own, confirming the gate isn't just theoretical caution.

    Bucket: TARGET_FIELD_POS_BINS/LABELS on yardline_100, with inside_10
    further split by QB_OBVIOUS_PASSING_DOWNS/DISTANCE -- see that
    constant's own comment for why only inside_10 gets this split.
    """
    mask = (
        (pbp["season_type"] == "REG") & (pbp["qb_dropback"] == 1)
        & (pbp["two_point_attempt"] == 0)
    )
    src = pbp.loc[mask].copy()
    src["qb_id"] = src["passer_player_id"].where(
        src["passer_player_id"].notna(), src["rusher_player_id"]
    )
    src = src[src["qb_id"].isin(qb_ids)]

    is_real_pass = (src["pass_attempt"] == 1) & (src["sack"] == 0)
    is_scramble = src["qb_scramble"] == 1
    qb_is_fumbler = (
        (src["fumbled_1_player_id"] == src["qb_id"])
        | (src["fumbled_2_player_id"] == src["qb_id"])
    )
    pick_six = (
        (src["interception"] == 1) & (src["return_touchdown"] == 1)
        & (src["td_team"] == src["defteam"])
    )

    field_pos_band = pd.cut(src["yardline_100"], TARGET_FIELD_POS_BINS,
                             labels=TARGET_FIELD_POS_LABELS, right=True)
    obvious = (
        src["down"].isin(QB_OBVIOUS_PASSING_DOWNS)
        & (src["ydstogo"] >= QB_OBVIOUS_PASSING_DISTANCE)
    )
    is_inside_10 = field_pos_band.astype(str) == "inside_10"
    inside_10_split = obvious.map({True: "inside_10|obvious", False: "inside_10|standard"})
    bucket = "dropback:" + field_pos_band.astype(str).where(~is_inside_10, inside_10_split)

    return pd.DataFrame({
        "player_id": src["qb_id"].to_numpy(),
        "season": src["season"].to_numpy(),
        "week": src["week"].to_numpy(),
        "bucket": bucket.to_numpy(),
        "passing_yards": src["yards_gained"].where(is_real_pass, 0.0).fillna(0).to_numpy(),
        "passing_tds": src["pass_touchdown"].fillna(0).to_numpy(),
        "passing_interceptions": src["interception"].fillna(0).to_numpy(),
        "pass_int_tds": pick_six.astype(float).to_numpy(),
        "rushing_yards": src["yards_gained"].where(is_scramble, 0.0).fillna(0).to_numpy(),
        "rushing_tds": src["rush_touchdown"].where(is_scramble, 0.0).fillna(0).to_numpy(),
        "fumbles_total": src["fumble"].where(qb_is_fumbler, 0.0).fillna(0).to_numpy(),
        "fumbles_lost_total": src["fumble_lost"].where(qb_is_fumbler, 0.0).fillna(0).to_numpy(),
    })


def _qb_rush_play_frame(pbp: pd.DataFrame, qb_ids: set) -> pd.DataFrame:
    """
    One row per QB DESIGNED rush attempt -- same mask as
    _qb_dropback_features's own designed_rush_attempts (rush_attempt ==
    1, qb_scramble == 0, qb_kneel == 0), restricted to real QBs and
    excluding two-point tries. A scramble is NOT a designed run (see
    _dropback_play_frame, where it's bucketed instead) and a kneel is not
    a real scoring opportunity at all -- same exclusions as elsewhere in
    this pipeline.

    Field-position bins reuse CARRY_FIELD_POS_BINS/LABELS -- see that
    constant's own comment for the real, goal-line-driven rate jump that
    justifies the finer resolution near the goal.
    """
    mask = (
        (pbp["season_type"] == "REG") & (pbp["rush_attempt"] == 1)
        & (pbp["qb_scramble"] == 0) & (pbp["qb_kneel"] == 0)
        & pbp["rusher_player_id"].notna() & (pbp["two_point_attempt"] == 0)
    )
    src = pbp.loc[mask]
    src = src[src["rusher_player_id"].isin(qb_ids)]

    qb_is_fumbler = (
        (src["fumbled_1_player_id"] == src["rusher_player_id"])
        | (src["fumbled_2_player_id"] == src["rusher_player_id"])
    )
    field_pos_band = pd.cut(src["yardline_100"], CARRY_FIELD_POS_BINS,
                             labels=CARRY_FIELD_POS_LABELS, right=True)

    return pd.DataFrame({
        "player_id": src["rusher_player_id"].to_numpy(),
        "season": src["season"].to_numpy(),
        "week": src["week"].to_numpy(),
        "bucket": ("qb_rush:" + field_pos_band.astype(str)).to_numpy(),
        "rushing_yards": src["yards_gained"].fillna(0).to_numpy(),
        "rushing_tds": src["rush_touchdown"].fillna(0).to_numpy(),
        "fumbles_total": src["fumble"].where(qb_is_fumbler, 0.0).fillna(0).to_numpy(),
        "fumbles_lost_total": src["fumble_lost"].where(qb_is_fumbler, 0.0).fillna(0).to_numpy(),
    })


def _qb_scored_play_universe(df: pd.DataFrame, pbp: pd.DataFrame, scoring_settings: dict) -> pd.DataFrame:
    """
    Every real QB dropback and designed rush attempt, scored per-play with
    this league's rules and labeled with its bucket. Long format:
    player_id, season, week, bucket, points -- the QB-only analogue of
    _scored_play_universe, kept as a SEPARATE rate table rather than
    pooled with the RB/WR/TE target/carry universe: a QB dropback and a
    WR target aren't the same opportunity type, and a QB rush near the
    goal line scores at a materially different rate than an RB carry
    there (see QB_OBVIOUS_PASSING_DOWNS/CARRY_FIELD_POS_BINS's own
    comments) -- blending them would let one leak into (and dilute) the
    other's rate.
    """
    qb_ids = _qb_player_ids(df)

    dropbacks = _dropback_play_frame(pbp, qb_ids)
    dropbacks["points"] = compute_custom_score(dropbacks, scoring_settings, warn=False)

    rushes = _qb_rush_play_frame(pbp, qb_ids)
    rushes["points"] = compute_custom_score(rushes, scoring_settings, warn=False)

    cols = ["player_id", "season", "week", "bucket", "points"]
    return pd.concat([dropbacks[cols], rushes[cols]], ignore_index=True)


def _bucket_rate_table(plays: pd.DataFrame, season: int, week: int) -> pd.Series:
    """
    League-average points per bucket, using ONLY plays strictly before
    (season, week) in global chronological order -- i.e. all of an
    earlier season, or an earlier week of the same season. This is what
    makes xfp for week N safe to use as a week-N feature: it never sees
    week N or later.
    """
    before = plays[
        (plays["season"] < season)
        | ((plays["season"] == season) & (plays["week"] < week))
    ]
    return before.groupby("bucket")["points"].mean()


def _xfp_table_for_universe(plays: pd.DataFrame, cutoffs: list) -> pd.DataFrame:
    """
    Shared expanding-window, point-in-time-safe xfp computation over any
    long-format (player_id, season, week, bucket, points) play universe --
    factored out so the skill-position (target+carry) and QB (dropback+
    designed-rush) universes can each run through the identical mechanism
    without duplicating it. See add_xfp_features's own docstring for the
    full point-in-time reasoning; this is the per-week loop body that used
    to live inline there.
    """
    xfp_parts = []
    for season, week in cutoffs:
        this_week = plays[(plays["season"] == season) & (plays["week"] == week)]
        if this_week.empty:
            continue
        rate_table = _bucket_rate_table(plays, season, week)
        rates = this_week["bucket"].map(rate_table)
        grouped = rates.groupby(this_week["player_id"])
        total = grouped.sum(min_count=1)
        any_null = grouped.apply(lambda s: s.isna().any())
        total = total.where(~any_null)
        part = total.reset_index(name="xfp")
        part["season"] = season
        part["week"] = week
        xfp_parts.append(part)

    if not xfp_parts:
        return pd.DataFrame(columns=["player_id", "season", "week", "xfp"])
    return pd.concat(xfp_parts, ignore_index=True)


def add_xfp_features(
    df: pd.DataFrame, pbp: pd.DataFrame, scoring_settings: dict
) -> pd.DataFrame:
    """
    Add xfp (expected fantasy points from opportunity alone) and
    fp_over_expected = custom_points - xfp, for every position including
    QB.

    For each RB/WR/TE player-week, xfp sums the bucket rate for every
    target and carry the player actually had that week. For each QB
    player-week, xfp sums the bucket rate for every dropback and
    designed rush attempt they actually had -- the same bucket-rate
    mechanism, applied to the play types Family 1 (step 3) originally
    skipped (see _qb_scored_play_universe). Either way, each bucket's
    rate is the league-average points per play in that bucket computed
    from every play STRICTLY BEFORE that week (see _bucket_rate_table) --
    an expanding, point-in-time-safe window that crosses season
    boundaries (unlike the in-season-only rule for Family 6's rolling
    aggregates -- see the point-in-time note below).

    The two rate tables (skill-position and QB) are computed and applied
    completely separately, then concatenated before the single merge onto
    df -- a QB dropback and a WR target are different opportunity types
    and must not share a bucket-rate estimate (see
    _qb_scored_play_universe's own comment). Any QB player_id is
    explicitly dropped from the skill-position table's contribution
    first: _carry_play_frame's own mask isn't QB-gated (a QB scramble is
    still `rush_attempt == 1`), so a scrambling QB could otherwise pick up
    a partial, WRONG xfp here (their scrambles alone, via the RB/WR/TE
    carry rate) that would compete with their real, complete QB xfp below.

    Point-in-time note: the rate table is expanding and GLOBAL, not
    reset at each season boundary. Week 1 of 2025 draws on the entirety
    of 2024 rather than starting from zero. This was a deliberate choice,
    not an oversight: with only two seasons of pbp, resetting at the
    season boundary would leave 2025's early weeks (and, worse, ALL of
    2024's early weeks, which have no prior season at all) with barely
    any data behind the rarer buckets. Excluding week N and everything
    after it -- not just requiring week < N -- is exactly what
    tests/test_no_leakage.py checks: it proves week N's xfp is unchanged
    when week N's own plays are removed from the source pbp, which a
    naive "<=" cutoff bug would fail.

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with player_id, position, team, season,
            week, custom_points (e.g. weekly_scored.parquet, already
            scored by compute_custom_score).
        pbp: play-by-play frame from get_pbp().
        scoring_settings: from get_sleeper_league()["scoring_settings"] --
            plays are scored with THIS league's actual rules, not generic
            PPR, via the same compute_custom_score() used everywhere else.

    Returns:
        Copy of df with XFP_OUTPUT_COLUMNS added.
        xfp is 0 (not null) for a player-week with zero qualifying plays
        (zero targets/carries for RB/WR/TE, zero dropbacks/designed
        rushes for QB) -- a known value, since summing zero opportunities
        is trivially zero regardless of whether any rate table exists
        yet.
        xfp is null when the player had at least one qualifying play that
        week but at least one of those plays fell in a bucket with NO
        historical plays before that week -- most common in the first
        few weeks of 2018, before the rarer merged buckets have
        accumulated data. A single unresolvable play nulls the whole
        week rather than silently under-counting it (no fake data).
    """
    required = ["player_id", "position", "team", "season", "week", "custom_points"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_xfp_features: df is missing columns {missing}")

    out = df.drop(columns=[c for c in XFP_OUTPUT_COLUMNS if c in df.columns])
    cutoffs = sorted(set(zip(out["season"], out["week"])))

    skill_plays = _scored_play_universe(pbp, scoring_settings)
    qb_plays = _qb_scored_play_universe(out, pbp, scoring_settings)
    qb_ids = _qb_player_ids(out)

    skill_xfp = _xfp_table_for_universe(skill_plays, cutoffs)
    skill_xfp = skill_xfp[~skill_xfp["player_id"].isin(qb_ids)]
    qb_xfp = _xfp_table_for_universe(qb_plays, cutoffs)

    xfp_table = pd.concat([skill_xfp, qb_xfp], ignore_index=True)
    out = out.merge(xfp_table, on=["player_id", "season", "week"], how="left")

    all_play_keys = (
        set(zip(skill_plays["player_id"], skill_plays["season"], skill_plays["week"]))
        | set(zip(qb_plays["player_id"], qb_plays["season"], qb_plays["week"]))
    )
    had_any_play = pd.Series(
        list(zip(out["player_id"], out["season"], out["week"]))
    ).isin(all_play_keys)
    zero_opportunity = (~had_any_play.to_numpy()) & out["xfp"].isna()
    out.loc[zero_opportunity, "xfp"] = 0.0

    out["fp_over_expected"] = out["custom_points"] - out["xfp"]

    return out


# ==========================================================================
# FAMILY 5B — OPPONENT DEFENSIVE STRENGTH (Family 5 addendum)
# ==========================================================================
# PHASE_2B_6_SPEC.md's Family 5 named this and deferred it: "Opponent
# defensive strength by position, computed on prior weeks only -- fantasy
# points allowed to RB/WR/TE, opponent-adjusted if practical." QB is out of
# scope here -- NOT for the reason xFP itself once was (xFP now has a real
# QB bucket-rate model too, see _qb_scored_play_universe/add_xfp_features),
# but because extending THIS metric to QB would need its own defense-side
# aggregation of QB xfp allowed (bucketed by down/distance/field position
# the way dropbacks/designed rushes are), which hasn't been built. That's a
# real, buildable follow-up now that the QB xfp it would reuse exists --
# out of scope for this pass, not silently skipped: no QB column below is
# ever populated, disclosed rather than hidden.
#
# The metric: for each defense, in each week, the xFP (NOT raw actual
# points) that the OPPOSING team's RB/WR/TE group generated that week.
# Reuses `xfp` exactly as add_xfp_features already computes it (a
# league-average bucket rate applied to real opportunities), rather than
# raw custom_points allowed -- for the same reason xfp exists at all: a
# defense that gave up one lucky 70-yard broken-tackle score looks worse
# than the opportunity it actually conceded, and a defense that forced an
# incompletion on a wide-open deep shot looks better than the dangerous
# look it actually allowed. xFP measures the opportunity, not the bounce.
#
# Point-in-time mechanics: identical recipe to add_rolling_features (window
# the raw per-team-week value with pandas' native ewm/expanding, then
# shift(1) the whole result within the group) but grouped by (team,
# position, season) instead of (player_id, season) -- a defense is a
# scheme, not an individual, but the same "role changes across an
# offseason, so reset at the season boundary" reasoning from Family 6
# applies here too (new coordinator, new personnel). Row-based like every
# other _ewm3 in this module, not calendar-based -- a bye week is simply a
# missing row, not a zero.
#
# Opponent adjustment: a single-pass schedule-strength correction, not a
# full iterative simultaneous-rating solve (e.g. Massey/Colley-style). An
# iterative method needs many games to converge and is hard to keep
# point-in-time-safe with as few as 1-3 prior games in an early-season
# fold; a single pass is transparent, cheap, and tractable at this data
# volume -- "opponent-adjusted if practical" from the spec, not "if
# perfect". For each defense, at each week: average the TRAILING offensive
# strength (each opponent's own generated_s2d, evaluated AS OF the week
# that game was played -- already point-in-time-safe on its own) of every
# team it has faced so far this season, compare that average to the
# league-wide average offensive strength at that same point in time, and
# subtract the difference from the defense's raw allowed number. A defense
# that has faced unusually strong offenses gets its raw "allowed" number
# reduced by exactly how much stronger those offenses were than average; a
# defense that has faced unusually weak ones gets it increased.
OPP_STRENGTH_POSITIONS = ("RB", "WR", "TE")

OPPONENT_STRENGTH_OUTPUT_COLUMNS = [
    "opp_def_xfp_allowed_ewm3", "opp_def_xfp_allowed_s2d",
    "opp_def_xfp_allowed_adj_ewm3", "opp_def_xfp_allowed_adj_s2d",
]


def _team_week_opponent(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Per (team, season, week): the opponent that team faced that week, built
    by stacking the home/away perspective of each schedule row (same
    stacking idiom as _team_week_context). No row for a team's bye week --
    the team simply doesn't appear for that (season, week).
    """
    sched = schedule[schedule["game_type"] == "REG"]
    shared = ["season", "week"]
    home = sched[["home_team", "away_team"] + shared].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = sched[["home_team", "away_team"] + shared].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    return pd.concat([home, away], ignore_index=True)


def _rolling_team_position(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Adds `{value_col}_ewm3`/`{value_col}_s2d` to long_df -- the same
    window-then-shift(1) recipe as add_rolling_features, generalized from
    (player_id, season) groups to (team, position, season) groups.
    long_df must have team, position, season, week, and value_col, with one
    row per (team, position, season, week) -- not one row per player.
    """
    sorted_df = long_df.sort_values(["team", "position", "season", "week"])
    group_keys = [sorted_df["team"], sorted_df["position"], sorted_df["season"]]
    grouped = sorted_df.groupby(["team", "position", "season"], sort=False)

    def _shift_within_group(s: pd.Series) -> pd.Series:
        return s.groupby(group_keys, sort=False).shift(1)

    ewm_raw = grouped[value_col].ewm(halflife=EWM_HALFLIFE, min_periods=1).mean().droplevel([0, 1, 2])
    s2d_raw = grouped[value_col].expanding().mean().droplevel([0, 1, 2])

    new_cols = pd.DataFrame({
        f"{value_col}_ewm3": _shift_within_group(ewm_raw).reindex(long_df.index),
        f"{value_col}_s2d": _shift_within_group(s2d_raw).reindex(long_df.index),
    }, index=long_df.index)
    return pd.concat([long_df, new_cols], axis=1)


def build_defense_strength_table(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Per (team, position, season, week) for position in OPP_STRENGTH_POSITIONS:
    this team's own trailing OFFENSIVE xFP output at that position
    (generated_ewm3/generated_s2d -- also the input to the opponent
    adjustment below), this team's trailing DEFENSIVE xFP allowed to that
    position (allowed_ewm3/allowed_s2d -- the raw "how good is this
    defense" metric), and the opponent-adjusted version of the allowed
    metric (allowed_adj_ewm3/allowed_adj_s2d). Public (not
    underscore-prefixed): src/export.py calls this directly to build the
    "which defenses are favorable this week" panel, a per-TEAM ranking
    that add_opponent_strength_features's per-PLAYER join below doesn't
    expose on its own.

    Args:
        df: player-week frame with player_id, position, team, season,
            week, xfp (i.e. df after add_xfp_features has already run).
        schedule: from get_schedule() -- used only to know who played whom
            each week, not for any Vegas/weather column.

    Returns:
        team, position, season, week, generated_ewm3, generated_s2d,
        allowed_ewm3, allowed_s2d, allowed_adj_ewm3, allowed_adj_s2d.
        Null wherever the team has fewer than the required prior in-season
        games at that position -- same season-boundary reset and null
        semantics as add_rolling_features.
    """
    required = ["player_id", "position", "team", "season", "week", "xfp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"build_defense_strength_table: df is missing columns {missing}")

    pos_df = df[df["position"].isin(OPP_STRENGTH_POSITIONS)]
    generated = (
        pos_df.groupby(["team", "position", "season", "week"])["xfp"]
        .sum(min_count=1)
        .reset_index(name="generated")
    )
    generated = _rolling_team_position(generated, "generated")

    team_week_opp = _team_week_opponent(schedule)

    # This team's own opponent-that-week's generated total becomes what
    # THIS team's defense allowed that week -- reusing the already-
    # point-in-time-safe generated_ewm3/generated_s2d (below) as the
    # offense's own "how good are they" read at the time of that specific
    # game, before this function applies its OWN shift(1) again for the
    # defense's trailing "allowed" trend.
    allowed_raw = generated.merge(
        team_week_opp, on=["team", "season", "week"], how="left"
    )[["opponent", "position", "season", "week", "generated"]].rename(
        columns={"opponent": "team", "generated": "allowed"}
    )
    allowed_raw = allowed_raw.dropna(subset=["team"])
    allowed = _rolling_team_position(allowed_raw, "allowed")

    # Opponent adjustment: for each of a defense's own games so far this
    # season, look up that week's OPPONENT's own generated_s2d (their
    # offensive strength as of that game -- already point-in-time-safe,
    # since generated_s2d itself never sees that team's own future).
    # Average those across the defense's games so far (shift(1)-safe
    # expanding mean, same _rolling_team_position recipe as everything
    # else here) to get "how strong were the offenses this defense has
    # faced, on average, as of this week" -- compared below against the
    # league-wide average at that same point in time.
    positions_df = pd.DataFrame({"position": list(OPP_STRENGTH_POSITIONS)})
    team_week_opp_pos = team_week_opp.merge(positions_df, how="cross")
    opp_strength_per_game = team_week_opp_pos.merge(
        generated[["team", "position", "season", "week", "generated_s2d"]]
        .rename(columns={"team": "opponent", "generated_s2d": "opp_offense_strength"}),
        on=["opponent", "position", "season", "week"], how="left",
    )
    opp_strength_per_game = _rolling_team_position(opp_strength_per_game, "opp_offense_strength")
    avg_opponent_strength = opp_strength_per_game[
        ["team", "position", "season", "week", "opp_offense_strength_s2d"]
    ].rename(columns={"opp_offense_strength_s2d": "avg_opponent_strength"})

    league_avg = (
        generated.groupby(["position", "season", "week"])["generated_s2d"]
        .mean()
        .reset_index(name="league_avg_generated_s2d")
    )

    out = allowed.merge(
        generated[["team", "position", "season", "week", "generated_ewm3", "generated_s2d"]],
        on=["team", "position", "season", "week"], how="outer",
    )
    out = out.merge(avg_opponent_strength, on=["team", "position", "season", "week"], how="left")
    out = out.merge(league_avg, on=["position", "season", "week"], how="left")

    sos_correction = out["avg_opponent_strength"] - out["league_avg_generated_s2d"]
    out["allowed_adj_ewm3"] = out["allowed_ewm3"] - sos_correction
    out["allowed_adj_s2d"] = out["allowed_s2d"] - sos_correction

    return out[["team", "position", "season", "week",
                "generated_ewm3", "generated_s2d",
                "allowed_ewm3", "allowed_s2d",
                "allowed_adj_ewm3", "allowed_adj_s2d"]]


DEFENSE_AIR_GROUND_OUTPUT_COLUMNS = [
    "xfp_allowed_air_ewm3", "xfp_allowed_air_s2d",
    "xfp_allowed_ground_ewm3", "xfp_allowed_ground_s2d",
]


def build_defense_air_ground_split(
    df: pd.DataFrame, pbp: pd.DataFrame, schedule: pd.DataFrame, scoring_settings: dict
) -> pd.DataFrame:
    """
    Per (team, season, week): this team's DEFENSIVE xFP allowed to
    RB/WR/TE, split by how it was conceded -- through the air (targets)
    or on the ground (carries). The SAME data and bucket-rate mechanism
    as build_defense_strength_table's own allowed_ewm3/allowed_s2d --
    the only difference is summing target-derived and carry-derived
    per-player-week xfp SEPARATELY instead of pooled. A single blended
    "xFP allowed" hides WHY a defense allows what it allows: a
    run-funneling defense and a pass-funneling defense can post the
    identical total while favoring completely different fantasy
    players. _target_play_frame/_carry_play_frame's bucket strings never
    collide (air_yards|field_pos vs. a bare field_pos label), so the
    league-average bucket RATE each side draws on is already exactly
    what it would be in the pooled computation -- this doesn't refit
    anything, it just keeps the two sums apart instead of adding them
    together before this function ever sees them.

    Deliberately NOT opponent-adjusted (unlike allowed_adj_* above) and
    NOT split by position (unlike the rest of this table) -- this
    describes one thing about a defense as a whole (where its real
    fantasy-point damage comes from), the same single-number-per-team
    framing PROE/pace/red-zone-split/target-distribution already use in
    src/team_tendencies.py, not a second per-position ranking next to
    build_defense_rankings' existing one.

    Args:
        df: player-week frame with player_id, position, team, season,
            week (weekly_scored or later is fine).
        pbp: from get_pbp().
        schedule: from get_schedule().
        scoring_settings: this league's real scoring rules, from
            get_sleeper_league()["scoring_settings"].

    Returns:
        team, season, week, xfp_allowed_air_ewm3, xfp_allowed_air_s2d,
        xfp_allowed_ground_ewm3, xfp_allowed_ground_s2d. Null wherever
        the team has fewer than the required prior in-season games,
        same convention as build_defense_strength_table.
    """
    required = ["player_id", "position", "team", "season", "week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"build_defense_air_ground_split: df is missing columns {missing}")

    cutoffs = sorted(set(zip(df["season"], df["week"])))

    targets = _target_play_frame(pbp)
    targets["points"] = compute_custom_score(targets, scoring_settings, warn=False)
    carries = _carry_play_frame(pbp)
    carries["points"] = compute_custom_score(carries, scoring_settings, warn=False)

    play_cols = ["player_id", "season", "week", "bucket", "points"]
    air_xfp = _xfp_table_for_universe(targets[play_cols], cutoffs).rename(columns={"xfp": "xfp_air"})
    ground_xfp = _xfp_table_for_universe(carries[play_cols], cutoffs).rename(columns={"xfp": "xfp_ground"})

    pos_df = (
        df[df["position"].isin(OPP_STRENGTH_POSITIONS)][["player_id", "team", "season", "week"]]
        .drop_duplicates()
        .merge(air_xfp, on=["player_id", "season", "week"], how="left")
        .merge(ground_xfp, on=["player_id", "season", "week"], how="left")
    )
    pos_df["xfp_air"] = pos_df["xfp_air"].fillna(0)
    pos_df["xfp_ground"] = pos_df["xfp_ground"].fillna(0)

    generated = (
        pos_df.groupby(["team", "season", "week"])[["xfp_air", "xfp_ground"]]
        .sum(min_count=1)
        .reset_index()
    )
    # No position axis here (see docstring) -- _rolling_team_position still
    # needs a `position` column to group by, so a constant fills that slot
    # without changing what the groupby actually partitions on.
    generated["position"] = "ALL"

    team_week_opp = _team_week_opponent(schedule)
    allowed_raw = generated.merge(
        team_week_opp, on=["team", "season", "week"], how="left"
    )[["opponent", "position", "season", "week", "xfp_air", "xfp_ground"]].rename(
        columns={"opponent": "team"}
    )
    allowed_raw = allowed_raw.dropna(subset=["team"])

    allowed_air = _rolling_team_position(
        allowed_raw[["team", "position", "season", "week", "xfp_air"]], "xfp_air"
    )
    allowed_ground = _rolling_team_position(
        allowed_raw[["team", "position", "season", "week", "xfp_ground"]], "xfp_ground"
    )

    out = allowed_air[["team", "position", "season", "week", "xfp_air_ewm3", "xfp_air_s2d"]].merge(
        allowed_ground[["team", "position", "season", "week", "xfp_ground_ewm3", "xfp_ground_s2d"]],
        on=["team", "position", "season", "week"],
    )
    return out.rename(columns={
        "xfp_air_ewm3": "xfp_allowed_air_ewm3", "xfp_air_s2d": "xfp_allowed_air_s2d",
        "xfp_ground_ewm3": "xfp_allowed_ground_ewm3", "xfp_ground_s2d": "xfp_allowed_ground_s2d",
    })[["team", "season", "week"] + DEFENSE_AIR_GROUND_OUTPUT_COLUMNS]


def add_opponent_strength_features(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Add OPPONENT_STRENGTH_OUTPUT_COLUMNS: how strong THIS WEEK's opponent's
    defense has trailingly been against players at THIS PLAYER'S OWN
    position, from build_defense_strength_table. Also adds `opponent`
    (that week's opposing team code) -- descriptive metadata for the
    export layer, deliberately NOT a model feature (raw team identity, not
    a computed strength signal) and so not part of
    OPPONENT_STRENGTH_OUTPUT_COLUMNS.

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with player_id, position, team, season,
            week, xfp (i.e. df after add_xfp_features has already run).
        schedule: from get_schedule().

    Returns:
        Copy of df with `opponent` and OPPONENT_STRENGTH_OUTPUT_COLUMNS
        added. Null for every QB row (scope gap -- see this section's
        module-level comment) and for a team's bye week (no opponent that
        week), the same "real absence of data, not a bug" convention as
        everywhere else in this pipeline.
    """
    required = ["player_id", "position", "team", "season", "week", "xfp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_opponent_strength_features: df is missing columns {missing}")

    out = df.drop(columns=[c for c in OPPONENT_STRENGTH_OUTPUT_COLUMNS + ["opponent"] if c in df.columns])

    team_week_opp = _team_week_opponent(schedule)
    out = out.merge(team_week_opp, on=["team", "season", "week"], how="left")

    defense_table = build_defense_strength_table(df, schedule).rename(columns={
        "team": "opponent",
        "allowed_ewm3": "opp_def_xfp_allowed_ewm3",
        "allowed_s2d": "opp_def_xfp_allowed_s2d",
        "allowed_adj_ewm3": "opp_def_xfp_allowed_adj_ewm3",
        "allowed_adj_s2d": "opp_def_xfp_allowed_adj_s2d",
    })[["opponent", "position", "season", "week"] + OPPONENT_STRENGTH_OUTPUT_COLUMNS]

    out = out.merge(defense_table, on=["opponent", "position", "season", "week"], how="left")
    return out


# ==========================================================================
# FIELD HEATMAP ZONES (Phase 5) — display-only, never a model feature
# ==========================================================================
# Deliberately SEPARATE bin definitions from the xFP buckets above, built
# from the same two raw ingredients (air_yards depth, yardline_100 field
# position) but shaped for a human reading a chart at a glance rather than
# for a league-average RATE ESTIMATE that needs enough plays per bucket to
# be statistically reliable across every player at once. xFP's
# merge-thin-buckets step (_TARGET_BUCKET_MERGES) solves a different
# problem -- a league-wide rate table's reliability; a heatmap zone's
# reliability is about ONE player's own sample, handled per-player by the
# `sparse` flag in src/export.py::build_heatmap_snapshot instead of a
# global bucket merge.
HEATMAP_DEPTH_BINS = [-float("inf"), 0, 10, 20, float("inf")]
HEATMAP_DEPTH_LABELS = ["behind_los", "short", "intermediate", "deep"]
HEATMAP_FIELD_POS_BINS = [-float("inf"), 20, 50, float("inf")]
HEATMAP_FIELD_POS_LABELS = ["red_zone", "midfield", "backfield"]

HEATMAP_ZONE_LABELS = {
    "behind_los": "Behind LOS", "short": "Short", "intermediate": "Intermediate", "deep": "Deep",
    "red_zone": "Red Zone", "midfield": "Midfield", "backfield": "Backfield",
    "left": "Left", "middle": "Middle", "right": "Right",
}

# Which zone kinds apply to which position -- QB gets pass location x
# depth only (no rushing zones, even though QBs do carry -- matches
# getHeatmapTitle()'s already-live "Pass Distribution" title, which
# doesn't promise a rushing view). RB gets both real usage types it
# actually has, matching getHeatmapTitle()'s "Rushing Direction &
# Receiving".
HEATMAP_POSITION_KINDS = {
    "QB": ["passing"],
    "RB": ["rushing", "receiving"],
    "WR": ["receiving"],
    "TE": ["receiving"],
}


def _real_target_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Every real target (pass_attempt==1, a recorded receiver, not a 2pt
    try) -- the SAME population _target_play_frame builds for xFP, kept
    as a separate copy here rather than imported so a change to xFP's
    scoring-oriented frame can't silently change what a heatmap zone
    counts. air_yards/yardline_100 are both ~100% populated on this real
    population (verified directly against 2025 pbp: 0% null on either
    column for real targets) -- the notna() filter below is defensive,
    not expected to drop real rows.
    """
    mask = (
        (pbp["season_type"] == "REG") & (pbp["pass_attempt"] == 1)
        & pbp["receiver_player_id"].notna() & (pbp["two_point_attempt"] == 0)
    )
    src = pbp.loc[mask]
    return src[src["air_yards"].notna() & src["yardline_100"].notna()]


def receiving_zone_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per real target, zoned by depth x field position (both
    HEATMAP_* bins, not xFP's). zone_a=depth, zone_b=field position."""
    src = _real_target_plays(pbp)
    depth = pd.cut(src["air_yards"], HEATMAP_DEPTH_BINS, labels=HEATMAP_DEPTH_LABELS, right=False).astype(str)
    field_pos = pd.cut(src["yardline_100"], HEATMAP_FIELD_POS_BINS, labels=HEATMAP_FIELD_POS_LABELS, right=True).astype(str)
    return pd.DataFrame({
        "player_id": src["receiver_player_id"].to_numpy(),
        "season": src["season"].to_numpy(),
        "week": src["week"].to_numpy(),
        "zone_a": depth.to_numpy(),
        "zone_b": field_pos.to_numpy(),
    })


def passing_zone_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per real target, zoned by pass_location x depth, attributed
    to the PASSER rather than the receiver -- the same real-target
    population as receiving_zone_plays, just grouped by whoever threw it.
    zone_a=location, zone_b=depth (reversed order from receiving_zone_plays
    on purpose -- "pass location and depth" is how a QB's own chart reads,
    left-to-right then how deep).

    pass_attempt==1 also fires on sack rows in this pbp snapshot (see
    PROJECT_CONTEXT.md's "ID columns are strings" sibling note on
    pass-attempt-shaped denominators) -- _real_target_plays already
    excludes those by requiring a real receiver_player_id, which a sack
    never has, so no separate sack filter is needed here.
    """
    src = _real_target_plays(pbp)
    location = src["pass_location"].astype(str)
    depth = pd.cut(src["air_yards"], HEATMAP_DEPTH_BINS, labels=HEATMAP_DEPTH_LABELS, right=False).astype(str)
    valid = src["pass_location"].notna()
    return pd.DataFrame({
        "player_id": src.loc[valid, "passer_player_id"].to_numpy(),
        "season": src.loc[valid, "season"].to_numpy(),
        "week": src.loc[valid, "week"].to_numpy(),
        "zone_a": location[valid].to_numpy(),
        "zone_b": depth[valid].to_numpy(),
    })


def rushing_zone_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per real carry (rush_attempt==1, a recorded rusher, not a 2pt
    try or a kneel -- the SAME population _carry_play_frame builds for
    xFP), zoned by run_location x field position. 99.3% of real 2025
    carries have a known run_location (verified directly); the ~0.7%
    without one are dropped, an honest small gap rather than a guessed
    direction. zone_a=direction, zone_b=field position.
    """
    mask = (
        (pbp["season_type"] == "REG") & (pbp["rush_attempt"] == 1)
        & pbp["rusher_player_id"].notna() & (pbp["two_point_attempt"] == 0)
        & (pbp["qb_kneel"] == 0)
    )
    src = pbp.loc[mask]
    src = src[src["run_location"].notna() & src["yardline_100"].notna()]
    field_pos = pd.cut(src["yardline_100"], HEATMAP_FIELD_POS_BINS, labels=HEATMAP_FIELD_POS_LABELS, right=True).astype(str)
    return pd.DataFrame({
        "player_id": src["rusher_player_id"].to_numpy(),
        "season": src["season"].to_numpy(),
        "week": src["week"].to_numpy(),
        "zone_a": src["run_location"].to_numpy(),
        "zone_b": field_pos.to_numpy(),
    })


# ==========================================================================
# FAMILY 6 — ROLLING AGGREGATES (step 4)
# ==========================================================================
# Every continuous PLAYER feature from Families 1-4 and xFP (volume/share,
# snap share, efficiency, situational, xFP -- NOT Family 5's game context,
# see below), rolled into three point-in-time-safe trailing summaries:
#   <feat>_ewm3  -- exponentially weighted mean, ~3-week half-life
#   <feat>_vol   -- season-to-date expanding STANDARD DEVIATION (volatility)
#   <feat>_s2d   -- season-to-date expanding MEAN
#
# _vol/_s2d naming history: the first version of this module built one
# column named "_std" for the expanding standard deviation, reasoning
# that PHASE_2B_6_SPEC.md's prose ("expanding mean (stability)") must
# have been a typo, since a mean doesn't measure stability. Corrected:
# it was a naming error, not a typo -- "season-to-date" was intended, and
# BOTH the mean and the standard deviation are wanted, as two separate
# columns. Kept the already-built standard deviation (useful for Phase
# 6.5's floor/ceiling work) under the honest name _vol, and added _s2d
# as the expanding mean this section always should have had alongside it.
#
# Family 5 (game context) is excluded from rolling treatment entirely --
# not just the categorical/boolean columns (is_home, roof, surface, where
# "no meaningful 3-week average of a stadium surface" was always obvious),
# but the continuous ones too (days_rest, spread, game_total,
# team_implied_total, temp, wind). These describe THIS WEEK'S game, not
# the player -- a trailing average of wind speed isn't a meaningful
# quantity, and rolling them was a mistake in the original spec, not a
# deliberate design choice. Confirmed harmful in practice, not just
# theoretically wrong: SHAP diagnostics on the Phase 6 model (see
# PROJECT_CONTEXT.md) showed context columns and their rolled variants
# occupying up to 8 of the top 20 features by importance at some
# positions -- weight the model was spending on noise instead of the
# opportunity signals that actually predict points. Current-week context
# values are still used directly as model features (Family 5's own
# add_context_features output, untouched) -- only the derived
# _ewm3/_vol/_s2d/prev_season_ variants are gone.
ROLLING_SOURCE_COLUMNS = (
    list(VOLUME_OUTPUT_COLUMNS)
    + list(SNAP_OUTPUT_COLUMNS)
    + list(EFFICIENCY_OUTPUT_COLUMNS)
    + list(SITUATIONAL_OUTPUT_COLUMNS)
    + list(XFP_OUTPUT_COLUMNS)
)

EWM_HALFLIFE = 3

# "Core" continuous features for the prior-season baseline -- everything in
# ROLLING_SOURCE_COLUMNS except team-level counts (team_rz_targets,
# team_rz_carries, team_inside_5_carries, team_two_minute_targets), which
# describe the player's TEAM that week, not the player -- a player's own
# "prior-season average of their team's red-zone trips" isn't a player
# attribute. (Game-context columns don't need a separate exclusion here
# any more -- they're not in ROLLING_SOURCE_COLUMNS to begin with.) What's
# left is volume, share, snap, efficiency, and situational-share/xFP
# columns -- the opportunity/role/skill signals a manager would actually
# want going into week 1, before any of this season's games have set the
# in-season rolling windows.
_PREV_SEASON_EXCLUDED_COLUMNS = {
    "team_rz_targets", "team_rz_carries", "team_inside_5_carries", "team_two_minute_targets",
}
PREV_SEASON_SOURCE_COLUMNS = [
    c for c in ROLLING_SOURCE_COLUMNS if c not in _PREV_SEASON_EXCLUDED_COLUMNS
]

ROLLING_OUTPUT_COLUMNS = (
    [f"{c}_ewm3" for c in ROLLING_SOURCE_COLUMNS]
    + [f"{c}_vol" for c in ROLLING_SOURCE_COLUMNS]
    + [f"{c}_s2d" for c in ROLLING_SOURCE_COLUMNS]
    + ["games_played", "snap_share_delta_3wk"]
    + [f"prev_season_{c}" for c in PREV_SEASON_SOURCE_COLUMNS]
)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trailing summaries of every continuous PLAYER feature from Families
    1-4 and xFP (Family 5's game context is deliberately excluded -- see
    ROLLING_SOURCE_COLUMNS's comment), plus games_played,
    snap_share_delta_3wk (deferred from step 1), and prior-season
    baselines for a core subset of those features.

    Point-in-time mechanics (ewm3/vol/s2d): df is sorted by
    ['player_id', 'season', 'week'] and grouped by ('player_id',
    'season'). The ewm/expanding window is computed on the RAW column
    first (pandas' native, C-level GroupBy.ewm()/.expanding() -- fast),
    then the WHOLE resulting series is `.shift(1)`-ed within each group.
    This lands on exactly the same numbers as shifting first and
    windowing second would (both ewm and expanding are pure
    left-to-right recursions -- the value at row N-1 never depends on
    row N or later, so "windowed then shift" and "shift then windowed"
    agree at every row; verified directly against a per-group
    Python-loop implementation before switching, since the loop version
    took ~22s on this dataset and the vectorized version takes under a
    second). Either way, week N's _ewm3/_vol/_s2d can never see week N's
    own row. Verified with x = [10, 20, NaN, 40, 50] across weeks 1-5:
    week 2's ewm3 is exactly week 1's raw value (10.0), and a NaN week
    (3) leaves the running ewm3 unchanged rather than resetting or
    zeroing it -- pandas' ewm/expanding skip NaN, they don't propagate
    it.

    Season boundary (ewm3/vol/s2d/games_played): grouping by
    ('player_id', 'season') together, not just 'player_id', means week 1
    of a new season always starts with an empty group -- ewm3/vol/s2d
    are null and games_played is 0, regardless of how much history the
    player has from the prior season. This is a DIFFERENT rule than
    xFP's rate table (add_xfp_features), which deliberately DOES cross
    season boundaries, AND a different rule than prev_season_* below,
    which deliberately DOES carry data across the boundary -- all three
    are intentional, not inconsistent:
      - ewm3/vol/s2d estimate THIS player's own in-season trend, and
        blending two seasons together would average a role change across
        an offseason (new team, new role, new coordinator) as if nothing
        happened -- that's the kind of leakage this family exists to
        prevent. Resetting to null/0 at the boundary is correct, not a
        gap to fix.
      - xFP's rate table estimates a LEAGUE-WIDE constant (the value of a
        20-yard target near the goal line, roughly stable year to year),
        not this player's own trajectory, and with only two seasons
        cached, resetting it every September would starve the rarer
        buckets of data for the first several weeks of 2024 AND all of
        2025.
      - prev_season_* (below) is a genuinely different quantity from
        ewm3/vol/s2d: not "this player's trend so far this season" (which
        must reset) but "what was this player's role LAST season"
        (which by definition is fixed and cannot leak -- an entire prior
        season is fully in the past by the time week 1 of the next one
        kicks off). It gets its OWN columns rather than blending into the
        in-season window, per PHASE_2B_6_SPEC.md's point-in-time rule
        ("prior-season aggregates go in their own columns with their own
        missingness").

    prev_season_<feat> (PREV_SEASON_SOURCE_COLUMNS only -- see that
    constant's comment for what's excluded and why): each player's
    FULL-SEASON mean of <feat> in season S, relabeled to season S+1 and
    left-merged onto every week of that player's S+1 rows. Structurally
    incapable of leaking -- there's no season-S+1 data anywhere in the
    computation, only season S's, which is entirely complete by the time
    S+1 exists. Null for a player with no row in season S at all
    (rookies, or anyone whose first year in this dataset is the current
    one) -- correctly "unknown", not "zero". The same value repeats
    across all of a player's weeks within a season (it's fixed once per
    player-season, not re-derived per week).

    snap_share_delta_3wk: offense_pct.shift(1) - offense_pct.shift(4) --
    the last known share minus the share from exactly 3 weeks before
    that. Both shifts are already point-in-time safe on their own (shift
    is monotonic), so no additional shift is applied to the difference.

    games_played: `.cumcount()` at row N is the count of PRIOR rows in
    the (player, season) group -- already shift(1)-shaped by
    construction (a player's first row of a season gets 0, not 1).

    Simplification: the ewm3 half-life is ROW-based (3 recorded games),
    not calendar-based. A player coming off a bye week or a missed game
    has a 2-week gap between rows that this treats as one step, slightly
    stretching the effective half-life. Not corrected here -- this is an
    auxiliary "let the model learn the weighting" feature, not something
    precision-critical enough to justify pandas' more fragile
    datetime-based `times=` EWM path.

    Idempotent: existing output columns are dropped before recomputing.
    Row order of the input is preserved in the output (internal
    computation sorts a copy, then re-aligns to the original index).

    Args:
        df: player-week frame with player_id, season, week, and every
            column in ROLLING_SOURCE_COLUMNS plus offense_pct (i.e. df
            after add_volume_features, add_snap_features,
            add_situational_features, add_context_features, and
            add_xfp_features have all already run).

    Returns:
        Copy of df with ROLLING_OUTPUT_COLUMNS added. Null wherever the
        player has fewer than the required prior in-season observations
        (heaviest in week 1 of each season, by construction), wherever
        the underlying source column is itself null for that position
        (e.g. dropbacks_ewm3 is null for every RB/WR/TE row, same as
        dropbacks itself -- xfp_ewm3 is NOT in this category any more,
        now that add_xfp_features populates real xfp for QB rows too), or
        -- for prev_season_* only -- wherever the player has no row at
        all in the immediately prior season.
    """
    required = ["player_id", "season", "week", "offense_pct"] + ROLLING_SOURCE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_rolling_features: df is missing columns {missing}")

    out = df.drop(columns=[c for c in ROLLING_OUTPUT_COLUMNS if c in df.columns])

    sorted_df = out.sort_values(["player_id", "season", "week"])
    group_keys = [sorted_df["player_id"], sorted_df["season"]]
    grouped = sorted_df.groupby(["player_id", "season"], sort=False)

    def _shift_within_group(s: pd.Series) -> pd.Series:
        return s.groupby(group_keys, sort=False).shift(1)

    # Computed on the RAW (unshifted) column via pandas' native, C-level
    # GroupBy.ewm()/.expanding() -- then the WHOLE result is shifted by 1
    # within each group. This is mathematically identical to shifting
    # first and windowing second (both ewm and expanding are pure
    # left-to-right recursions: the value at row N-1 never depends on row
    # N or later, so "windowed[N-1]" and "shift(windowed)[N]" are the
    # same number) -- verified directly against the per-group Python-loop
    # version this replaced. It matters because the per-group
    # .transform(lambda ...) version took ~22s on 11,869 rows x 34
    # source columns (a Python callback per (player, season) group, per
    # column); this vectorized version is under a second.
    results: dict[str, pd.Series] = {}
    for col in ROLLING_SOURCE_COLUMNS:
        ewm_raw = grouped[col].ewm(halflife=EWM_HALFLIFE, min_periods=1).mean().droplevel([0, 1])
        results[f"{col}_ewm3"] = _shift_within_group(ewm_raw)

        vol_raw = grouped[col].expanding().std().droplevel([0, 1])
        results[f"{col}_vol"] = _shift_within_group(vol_raw)

        s2d_raw = grouped[col].expanding().mean().droplevel([0, 1])
        results[f"{col}_s2d"] = _shift_within_group(s2d_raw)

    results["games_played"] = grouped.cumcount()
    results["snap_share_delta_3wk"] = (
        _shift_within_group(sorted_df["offense_pct"])
        - grouped["offense_pct"].shift(4)
    )

    # Single concat rather than 100+ sequential `out[name] = ...`
    # assignments -- the latter re-fragments the frame on every insert
    # (pandas warns about exactly this) and was ~20s on this dataset;
    # concat once is under a second.
    new_cols = pd.DataFrame(
        {name: series.reindex(out.index) for name, series in results.items()},
        index=out.index,
    )
    out = pd.concat([out, new_cols], axis=1)

    # Prior-season baselines: each player's full-season mean in season S
    # relabeled to season S+1, then left-merged onto every week of that
    # player's S+1 rows. No shift/expanding mechanics needed -- an entire
    # prior season is fully in the past regardless of which week of the
    # CURRENT season a row is on, so the same value applies to all of them.
    season_avg = (
        out.groupby(["player_id", "season"])[PREV_SEASON_SOURCE_COLUMNS]
        .mean()
        .reset_index()
    )
    season_avg["season"] = season_avg["season"] + 1
    season_avg = season_avg.rename(
        columns={c: f"prev_season_{c}" for c in PREV_SEASON_SOURCE_COLUMNS}
    )
    out = out.merge(season_avg, on=["player_id", "season"], how="left")

    return out


# ==========================================================================
# FAMILY 7 — TREND SIGNAL (Phase 3')
# ==========================================================================
# Replaces NOTEBOOK_OUTLINE.md's old Phase 3 ("role classification" --
# Pocket Passer / 3-Down Back / Slot WR, etc.). A role label is a category:
# it forces a player into one bucket of a fixed set using thresholds picked
# by eye, and says nothing about whether that role is CHANGING right now,
# which is the thing a manager actually needs to know week to week. A trend
# signal is a continuous, honestly-uncertain read on exactly that, with no
# bucket to half-fit a player into.
#
# Window choice, settled empirically rather than assumed -- see
# notebooks/04_usage_trends.ipynb section 1 for the full study and
# PROJECT_CONTEXT.md's "Phase 3' findings" for the summary. The question:
# does a usage rise measured over a 3-week half-life actually predict the
# NEXT game's usage staying elevated ("holds") rather than falling back to
# the season baseline ("reverts"), compared to a 4- or 5-week half-life?
# 3 won for every one of target_share/carry_share/offense_pct/
# rz_opportunity_share, on both hold-rate and correlation with next-game
# usage, monotonically (3 > 4 > 5 in every single case, out of 21k+ real
# player-weeks per feature). So this reuses add_rolling_features's EXISTING
# <feat>_ewm3/<feat>_s2d columns for target_share/carry_share/offense_pct
# directly -- no second, competing half-life constant.
#
# rz_opportunity_share is new: Family 4 has rz_target_share and
# rz_carry_share separately, with DIFFERENT denominators (team RZ targets
# vs. team RZ carries), so they can't just be added together into one
# honest combined share the way their raw counts can. Computed here with
# the SAME shift(1)-after-window mechanics as add_rolling_features, at the
# same validated halflife=3 -- but deliberately NOT added to
# ROLLING_SOURCE_COLUMNS. Doing so would silently turn it into a new
# src/model.py training feature (FEATURE_COLUMNS is derived FROM
# ROLLING_OUTPUT_COLUMNS) and retroactively change the already-published,
# already-validated Phase 6 model without anyone asking for that. This
# family is a display/export-layer derived feature, downstream of the
# model, not a new model input.
#
# Comparability across players: the raw ewm3-minus-s2d GAP isn't
# comparable across players on its own -- a bell-cow RB's season-long
# target_share naturally swings more, in raw percentage points, than a
# committee back's. Dividing by the player's own season-to-date volatility
# (the already-existing <feat>_vol column -- expanding std) turned out to
# matter empirically, not just conceptually: at a fixed
# z = gap / vol threshold, hold-rate clearly improves over a raw
# top-quartile-gap threshold at a comparable sample size (see the
# notebook). The z > 0.25 boundary for "rising"/"falling" below was picked
# by checking 0.0/0.25/0.5/0.75/1.0 against real hold-rates: 0.25 keeps a
# workable ~13-20% of eligible weeks flagged per direction while clearly
# beating a 0.0 (any positive gap) cutoff; higher thresholds cut the
# flagged sample down to near-nothing (well under 2% of weeks) for a
# marginal further hold-rate gain, too thin a list to be useful. 0/0 (a
# player with zero within-season variance so far, e.g. exactly one game)
# divides to null, not a spurious signal.
#
# Honesty note on rz_opportunity_share specifically: even at the validated
# window and threshold, its hold-rate stays below 50% in the historical
# check (a "rise" reverts more often than it holds), clearly weaker than
# the other three -- red-zone opportunities are a low-volume, high-variance
# event category, and this signal is real but noisier. Shipped anyway
# because it was explicitly asked for, not suppressed -- but callers
# (the Phase 7 export, any future dashboard panel) should not present it
# with the same confidence as the other three without repeating that
# caveat.
TREND_SOURCE_FEATURES = ["target_share", "carry_share", "offense_pct", "rz_opportunity_share"]
TREND_DIRECTION_THRESHOLD = 0.25

# Below this many prior in-season games, the season-to-date mean/vol this
# signal divides by are themselves too thin to trust -- validated directly:
# raising the eligibility floor from 2 to 8 games monotonically improved
# the signal's correlation with next-game usage (e.g. target_share: 0.085
# at >=2 games vs. 0.123 at >=8), but 8 would exclude nearly half of every
# season from ever appearing on a riser/faller list. 5 is the floor this
# module's own validation study was run under, not a separate, unjustified
# number chosen after the fact.
MIN_GAMES_FOR_TREND = 5

TREND_OUTPUT_COLUMNS = (
    ["rz_opportunity_share", "rz_opportunity_share_ewm3",
     "rz_opportunity_share_s2d", "rz_opportunity_share_vol"]
    + [f"{c}_trend_signal" for c in TREND_SOURCE_FEATURES]
    + [f"{c}_trend_direction" for c in TREND_SOURCE_FEATURES]
)


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a normalized usage-trend signal and a rising/falling/stable
    direction label for snap share (offense_pct), target_share,
    carry_share, and the new rz_opportunity_share -- see this section's
    module-level comment for the window/threshold choices and why they're
    each backed by a real historical check, not a guess.

    rz_opportunity_share = (rz_targets + rz_carries) / (team_rz_targets +
    team_rz_carries) -- a single share spanning both a rusher's and a
    receiver's red-zone touches. Null when the team recorded zero RZ plays
    all week (0/0), the same convention as every other share in this
    module -- a real absence of data, not this player getting shut out.

    trend_signal = (<feat>_ewm3 - <feat>_s2d) / <feat>_vol -- how many of
    the player's OWN season-to-date standard deviations above/below
    baseline their recent (3-game half-life) usage is running. Null before
    MIN_GAMES_FOR_TREND prior in-season games, or wherever the underlying
    ewm3/s2d/vol inputs are themselves null (e.g. every rolling column for
    a QB's carry_share-adjacent fields where the concept doesn't apply --
    inherited, not re-derived here).

    trend_direction is "rising" (signal > TREND_DIRECTION_THRESHOLD),
    "falling" (signal < -TREND_DIRECTION_THRESHOLD), "stable" (between), or
    null (signal itself null).

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with player_id, season, week, games_played,
            rz_targets, team_rz_targets, rz_carries, team_rz_carries, and
            every <feat>/<feat>_ewm3/<feat>_s2d/<feat>_vol column for
            target_share/carry_share/offense_pct (i.e. df after
            add_situational_features and add_rolling_features have both
            already run).

    Returns:
        Copy of df with TREND_OUTPUT_COLUMNS added.
    """
    required = (
        ["player_id", "season", "week", "games_played",
         "rz_targets", "team_rz_targets", "rz_carries", "team_rz_carries"]
        + [f"{c}{suffix}" for c in ("target_share", "carry_share", "offense_pct")
           for suffix in ("", "_ewm3", "_s2d", "_vol")]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_trend_features: df is missing columns {missing}")

    out = df.drop(columns=[c for c in TREND_OUTPUT_COLUMNS if c in df.columns])

    rz_opportunity_share = (
        (out["rz_targets"] + out["rz_carries"])
        / (out["team_rz_targets"] + out["team_rz_carries"])
    ).rename("rz_opportunity_share")
    out = pd.concat([out, rz_opportunity_share], axis=1)

    sorted_df = out.sort_values(["player_id", "season", "week"])
    group_keys = [sorted_df["player_id"], sorted_df["season"]]
    grouped = sorted_df.groupby(["player_id", "season"], sort=False)

    def _shift_within_group(s: pd.Series) -> pd.Series:
        return s.groupby(group_keys, sort=False).shift(1)

    # Same recipe as add_rolling_features: window the RAW column, then
    # shift(1) the whole result within each (player, season) group -- week
    # N's rz_opportunity_share_ewm3/_s2d/_vol never see week N's own row.
    ewm_raw = grouped["rz_opportunity_share"].ewm(halflife=EWM_HALFLIFE, min_periods=1).mean().droplevel([0, 1])
    s2d_raw = grouped["rz_opportunity_share"].expanding().mean().droplevel([0, 1])
    vol_raw = grouped["rz_opportunity_share"].expanding().std().droplevel([0, 1])

    rz_cols = pd.DataFrame({
        "rz_opportunity_share_ewm3": _shift_within_group(ewm_raw),
        "rz_opportunity_share_s2d": _shift_within_group(s2d_raw),
        "rz_opportunity_share_vol": _shift_within_group(vol_raw),
    }).reindex(out.index)
    out = pd.concat([out, rz_cols], axis=1)

    # Single concat rather than 8 sequential out[name] = ... assignments --
    # same reasoning as add_rolling_features's own concat (the latter
    # re-fragments the frame on every insert; pandas warns about exactly
    # this).
    eligible = out["games_played"] >= MIN_GAMES_FOR_TREND
    signal_cols: dict[str, pd.Series] = {}
    for feat in TREND_SOURCE_FEATURES:
        gap = out[f"{feat}_ewm3"] - out[f"{feat}_s2d"]
        signal = (gap / out[f"{feat}_vol"]).where(eligible)
        signal_cols[f"{feat}_trend_signal"] = signal

        direction = pd.Series("stable", index=out.index, dtype="string")
        direction = direction.mask(signal > TREND_DIRECTION_THRESHOLD, "rising")
        direction = direction.mask(signal < -TREND_DIRECTION_THRESHOLD, "falling")
        direction = direction.mask(signal.isna())
        signal_cols[f"{feat}_trend_direction"] = direction

    out = pd.concat([out, pd.DataFrame(signal_cols, index=out.index)], axis=1)
    return out


# ==========================================================================
# QB RUSHING SHARE OF POINTS — export-layer descriptive signal
# ==========================================================================
# What share of a QB's own custom_points came from rushing (rush_yd +
# rush_td + rush_2pt) rather than passing. This league pays 0.1/rushing
# yard and 6/rushing TD against 0.04/passing yard and 4/passing TD, so a
# rushing QB banks points before he throws a single pass -- the single
# most decision-relevant number the old Opportunity Shares panel never
# showed for QB (snap %/target share/carry share/red-zone share are all
# meaningless for a starting quarterback, who takes every snap and has no
# target share at all).
#
# Not a model feature, same reasoning as Family 7's rz_opportunity_share:
# an export-layer descriptive signal, downstream of the model's own
# feature set (FEATURE_COLUMNS_BY_POSITION), not folded in without a
# measured reason -- see PROJECT_CONTEXT.md's QB xFP as a Model Feature
# findings for what "folded in without measuring" already cost once.
QB_RUSHING_SHARE_OUTPUT_COLUMNS = [
    "rushing_share_of_points", "rushing_share_of_points_ewm3", "rushing_share_of_points_s2d",
]


def add_qb_rushing_share_feature(df: pd.DataFrame, scoring_settings: dict) -> pd.DataFrame:
    """
    Add rushing_share_of_points (QB-only) and its point-in-time-safe
    _ewm3/_s2d rolled versions -- same recipe as add_trend_features's own
    rz_opportunity_share: window the RAW per-week ratio, then shift(1)
    within (player_id, season), so week N's rolled values never see week
    N's own row.

    rushing_share_of_points = (rushing_yards*rush_yd + rushing_tds*rush_td
    + rushing_2pt_conversions*rush_2pt) / custom_points, using THIS
    league's real scoring weights (never hardcoded -- a league that
    reweights rushing vs. passing would silently change what "banks
    points" means). Null wherever custom_points is exactly 0 that week (a
    share with a zero denominator has no defined value, same convention
    as scramble_rate), and null for every non-QB row -- the concept
    doesn't apply to a player who doesn't throw.

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with player_id, position, season, week,
            custom_points, rushing_yards, rushing_tds,
            rushing_2pt_conversions (i.e. weekly_scored's own raw columns,
            still present at this point in the pipeline).
        scoring_settings: from get_sleeper_league()["scoring_settings"].

    Returns:
        Copy of df with QB_RUSHING_SHARE_OUTPUT_COLUMNS added.
    """
    required = ["player_id", "position", "season", "week", "custom_points",
                "rushing_yards", "rushing_tds", "rushing_2pt_conversions"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_qb_rushing_share_feature: df is missing columns {missing}")

    out = df.drop(columns=[c for c in QB_RUSHING_SHARE_OUTPUT_COLUMNS if c in df.columns])

    rush_yd_wt = float(scoring_settings.get("rush_yd") or 0)
    rush_td_wt = float(scoring_settings.get("rush_td") or 0)
    rush_2pt_wt = float(scoring_settings.get("rush_2pt") or 0)
    rushing_points = (
        out["rushing_yards"].fillna(0) * rush_yd_wt
        + out["rushing_tds"].fillna(0) * rush_td_wt
        + out["rushing_2pt_conversions"].fillna(0) * rush_2pt_wt
    )
    share = (rushing_points / out["custom_points"]).where(out["custom_points"] != 0)
    rushing_share_of_points = share.where(out["position"] == "QB").rename("rushing_share_of_points")

    sorted_share = rushing_share_of_points.reindex(out.index).to_frame().join(out[["player_id", "season", "week"]])
    sorted_share = sorted_share.sort_values(["player_id", "season", "week"])
    group_keys = [sorted_share["player_id"], sorted_share["season"]]
    grouped = sorted_share.groupby(["player_id", "season"], sort=False)

    def _shift_within_group(s: pd.Series) -> pd.Series:
        return s.groupby(group_keys, sort=False).shift(1)

    ewm_raw = grouped["rushing_share_of_points"].ewm(halflife=EWM_HALFLIFE, min_periods=1).mean().droplevel([0, 1])
    s2d_raw = grouped["rushing_share_of_points"].expanding().mean().droplevel([0, 1])

    new_cols = pd.DataFrame({
        "rushing_share_of_points": rushing_share_of_points,
        "rushing_share_of_points_ewm3": _shift_within_group(ewm_raw),
        "rushing_share_of_points_s2d": _shift_within_group(s2d_raw),
    }).reindex(out.index)
    return pd.concat([out, new_cols], axis=1)


def get_usage_trend_leaders(
    df: pd.DataFrame,
    season: int,
    week: int,
    feature: str,
    position: str | None = None,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Riser/faller lists: the top `top_n` players by `<feature>_trend_signal`
    for one (season, week), highest first (risers) and lowest first
    (fallers), optionally restricted to one position.

    Only rows with a non-null signal are eligible -- add_trend_features
    already nulls the signal below MIN_GAMES_FOR_TREND prior games, so this
    is what keeps a two-game small-sample player from topping the list on
    noise (a null wouldn't reliably sort to either extreme on its own, so
    it's filtered explicitly here rather than relying on sort order).

    Args:
        df: player-week frame after add_trend_features has run.
        season, week: the week to rank.
        feature: one of TREND_SOURCE_FEATURES.
        position: restrict to one position (QB/RB/WR/TE), or None for all.
        top_n: how many players per list.

    Returns:
        (risers, fallers) -- each a DataFrame with player_id,
        player_display_name, position, team, the feature's raw value,
        <feature>_ewm3, <feature>_trend_signal, and <feature>_trend_direction,
        sorted by signal (risers descending, fallers ascending).
    """
    if feature not in TREND_SOURCE_FEATURES:
        raise ValueError(
            f"get_usage_trend_leaders: unknown feature {feature!r}, "
            f"expected one of {TREND_SOURCE_FEATURES}"
        )
    signal_col = f"{feature}_trend_signal"

    week_df = df[
        (df["season"] == season) & (df["week"] == week) & df[signal_col].notna()
    ]
    if position is not None:
        week_df = week_df[week_df["position"] == position]

    id_cols = [c for c in ["player_id", "player_display_name", "position", "team"] if c in week_df.columns]
    cols = id_cols + [feature, f"{feature}_ewm3", signal_col, f"{feature}_trend_direction"]

    risers = week_df.sort_values(signal_col, ascending=False).head(top_n)[cols].reset_index(drop=True)
    fallers = week_df.sort_values(signal_col, ascending=True).head(top_n)[cols].reset_index(drop=True)
    return risers, fallers
