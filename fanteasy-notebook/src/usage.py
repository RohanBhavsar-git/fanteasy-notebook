"""
FanTeasy Stats — Usage and efficiency features (Phase 2b)

Steps 1-2 of PHASE_2B_6_SPEC.md's Order of Work: volume/share, snap-share,
situational, and game-context feature families. Keep this separate from
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
    over every target/carry strictly before week N -- crossing season
    boundaries deliberately, since two seasons of pbp isn't enough to
    reset the clock every September. QB xfp only covers rushing
    opportunities, not passing -- QB fp_over_expected is not a
    meaningful luck signal under this design.

Design note on the season boundary in rolling aggregates (add_rolling_features):
    See that function's own docstring for the full reasoning. Short
    version: ewm3/std/games_played group by (player_id, season) together,
    so they reset at every season boundary -- deliberately the OPPOSITE
    of xFP's rate table, because these describe a player's OWN trend
    (which really does reset with a new team/role/coordinator) while
    xFP's rate table estimates a league-wide constant (which doesn't).

Usage:
    from src.usage import (
        add_volume_features, add_snap_features,
        add_situational_features, add_context_features,
        add_xfp_features, add_rolling_features,
    )
    df = add_volume_features(weekly_scored, pbp)
    df = add_snap_features(df, snaps, crosswalk)
    df = add_situational_features(df, pbp)
    df = add_context_features(df, schedule)
    df = add_xfp_features(df, pbp, scoring_settings)
    df = add_rolling_features(df)
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
    """
    reg = pbp[pbp["season_type"] == "REG"]

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
    mapped = snaps_reg.merge(
        cw[["pfr_id", "gsis_id"]],
        left_on="pfr_player_id", right_on="pfr_id", how="inner",
    )

    dupes = mapped.duplicated(subset=["gsis_id", "season", "week"], keep=False)
    if dupes.any():
        raise ValueError(
            f"{dupes.sum()} snap rows collide on (gsis_id, season, week) after "
            "the pfr_player_id -> gsis_id join -- two PFR players mapped to the "
            "same gsis_id. Investigate before trusting snap_share."
        )

    snap_cols = (
        mapped[["gsis_id", "season", "week", "offense_snaps", "offense_pct"]]
        .rename(columns={"gsis_id": "player_id"})
    )

    return out.merge(snap_cols, on=["player_id", "season", "week"], how="left")


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
    """
    reg = pbp[pbp["season_type"] == "REG"]
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
    """
    reg = pbp[pbp["season_type"] == "REG"]
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

    This covers targets and carries only, not a QB's own pass attempts.
    A QB's rushing opportunities are represented (their carries score
    like anyone else's); their passing production has no xFP counterpart
    at all under this design. See add_xfp_features's docstring.
    """
    targets = _target_play_frame(pbp)
    targets["points"] = compute_custom_score(targets, scoring_settings, warn=False)

    carries = _carry_play_frame(pbp)
    carries["points"] = compute_custom_score(carries, scoring_settings, warn=False)

    cols = ["player_id", "season", "week", "bucket", "points"]
    return pd.concat([targets[cols], carries[cols]], ignore_index=True)


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


def add_xfp_features(
    df: pd.DataFrame, pbp: pd.DataFrame, scoring_settings: dict
) -> pd.DataFrame:
    """
    Add xfp (expected fantasy points from opportunity alone) and
    fp_over_expected = custom_points - xfp.

    For each player-week, xfp sums the bucket rate for every target and
    carry the player actually had that week, where each bucket's rate is
    the league-average points per play in that bucket computed from
    every play STRICTLY BEFORE that week (see _bucket_rate_table) --
    an expanding, point-in-time-safe window that crosses season
    boundaries (unlike the in-season-only rule for Family 6's rolling
    aggregates -- see the point-in-time note below).

    IMPORTANT SCOPE GAP: this covers targets and carries only, per the
    spec's Family 1(step 3) bucket list -- there is no bucket for a QB's
    own pass attempts. A QB's xfp would only reflect their rushing
    opportunities; their (much larger) passing production has no
    counterpart in xfp at all, so QB fp_over_expected would read as
    strongly "positive" for essentially every passing QB regardless of
    luck -- not a real signal, an artifact of what wasn't modeled. Rather
    than ship a column that looks like a luck metric but isn't, xfp and
    fp_over_expected are explicitly forced to null for QB rows below.
    This must never reach a dashboard panel as if it meant something for
    QBs -- fix by adding a passing-yardage bucket family, not by
    interpreting the null as "no opinion."

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
        xfp and fp_over_expected are null for every QB row -- see the
        SCOPE GAP note above. Not "unknown", deliberately not modeled.
        xfp is 0 (not null) for an RB/WR/TE player-week with zero
        qualifying plays (zero targets and zero non-kneel carries) -- a
        known value, since summing zero opportunities is trivially zero
        regardless of whether any rate table exists yet.
        xfp is null when the player had at least one target/carry that
        week but at least one of those plays fell in a bucket with NO
        historical plays before that week -- most common in the first
        few weeks of 2024, before the rarer merged buckets have
        accumulated data. A single unresolvable play nulls the whole
        week rather than silently under-counting it (no fake data).
    """
    required = ["player_id", "position", "team", "season", "week", "custom_points"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_xfp_features: df is missing columns {missing}")

    out = df.drop(columns=[c for c in XFP_OUTPUT_COLUMNS if c in df.columns])

    plays = _scored_play_universe(pbp, scoring_settings)

    cutoffs = sorted(set(zip(out["season"], out["week"])))
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

    xfp_table = (
        pd.concat(xfp_parts, ignore_index=True) if xfp_parts
        else pd.DataFrame(columns=["player_id", "season", "week", "xfp"])
    )

    out = out.merge(xfp_table, on=["player_id", "season", "week"], how="left")

    had_any_play = pd.Series(
        list(zip(out["player_id"], out["season"], out["week"]))
    ).isin(set(zip(plays["player_id"], plays["season"], plays["week"])))
    zero_opportunity = (~had_any_play.to_numpy()) & out["xfp"].isna()
    out.loc[zero_opportunity, "xfp"] = 0.0

    out["fp_over_expected"] = out["custom_points"] - out["xfp"]

    # xFP covers targets and carries only -- a QB's passing production
    # (the bulk of their real points) has no bucket and no counterpart
    # here. Leaving xfp/fp_over_expected computed for QB rows would show
    # every passing QB as wildly "over expected", which isn't a luck
    # signal, it's just points that were never modeled. Null them
    # outright rather than let a QB row masquerade as a real estimate --
    # this must never reach a dashboard panel as if it meant something.
    out.loc[out["position"] == "QB", ["xfp", "fp_over_expected"]] = pd.NA

    return out


# ==========================================================================
# FAMILY 6 — ROLLING AGGREGATES (step 4)
# ==========================================================================
# Every continuous feature from Families 1-5 (volume/share, snap share,
# situational, game context, xFP), rolled into three point-in-time-safe
# trailing summaries:
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
# Categorical/boolean context columns (is_home, roof, surface) are
# excluded -- there's no meaningful "3-week average" of a stadium surface.
_NON_CONTINUOUS_CONTEXT_COLUMNS = {"is_home", "roof", "surface"}

ROLLING_SOURCE_COLUMNS = (
    list(VOLUME_OUTPUT_COLUMNS)
    + list(SNAP_OUTPUT_COLUMNS)
    + list(SITUATIONAL_OUTPUT_COLUMNS)
    + [c for c in CONTEXT_OUTPUT_COLUMNS if c not in _NON_CONTINUOUS_CONTEXT_COLUMNS]
    + list(XFP_OUTPUT_COLUMNS)
)

EWM_HALFLIFE = 3

# "Core" continuous features for the prior-season baseline -- ROLLING_SOURCE_COLUMNS
# minus two groups that don't describe a PLAYER's own persistent role:
#   - team-level counts (team_rz_targets, team_rz_carries,
#     team_inside_5_carries, team_two_minute_targets) -- these describe
#     the player's TEAM that week, not the player; a player's own
#     "prior-season average of their team's red-zone trips" isn't a
#     player attribute.
#   - game-context columns (days_rest, spread, game_total,
#     team_implied_total, temp, wind) -- these describe THIS season's
#     schedule/circumstances, not something carried over from last year.
# What's left is volume, share, snap, situational-share, and xFP columns
# -- the opportunity/role signals a manager would actually want going
# into week 1, before any of this season's games have set the in-season
# rolling windows.
_PREV_SEASON_EXCLUDED_COLUMNS = {
    "team_rz_targets", "team_rz_carries", "team_inside_5_carries", "team_two_minute_targets",
    "days_rest", "spread", "game_total", "team_implied_total", "temp", "wind",
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
    Add trailing summaries of every continuous feature from Families 1-5,
    plus games_played, snap_share_delta_3wk (deferred from step 1), and
    prior-season baselines for a core subset of those features.

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
        dropbacks itself; xfp_ewm3 is null for every QB row), or -- for
        prev_season_* only -- wherever the player has no row at all in
        the immediately prior season.
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
