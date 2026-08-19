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

Usage:
    from src.usage import (
        add_volume_features, add_snap_features,
        add_situational_features, add_context_features,
    )
    df = add_volume_features(weekly_scored, pbp)
    df = add_snap_features(df, snaps, crosswalk)
    df = add_situational_features(df, pbp)
    df = add_context_features(df, schedule)
"""

from __future__ import annotations

import logging

import pandas as pd

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
