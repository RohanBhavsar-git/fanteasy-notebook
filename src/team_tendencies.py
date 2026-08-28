"""
FanTeasy Stats — Team Tendencies: real offense-level identity from pbp

The honest version of an earlier "coaching scheme" idea: nflverse has no
coordinator table (it would need permanent hand-maintenance, and a
per-coordinator sample is tiny), so this measures what a TEAM actually
does, not who calls the plays. Four families, all derived directly from
real play-by-play, never from a pre-aggregated feature column:

- PROE (pass rate over expected) -- nflverse's pbp already ships play-level
  `pass_oe`/`xpass` (nflfastR's own down/distance/score/time expected-pass
  model), verified directly against real 2025 pbp (99.6% populated on real
  pass/run plays; null elsewhere is nflfastR's OWN garbage-time/kickoff
  exclusion, not a gap here). Team PROE is just the mean of `pass_oe` over
  a team's real neutral-situation plays each week -- no second, competing
  expected-pass model needed.
- Pace -- two numbers, not one, since they answer different questions.
  `seconds_per_play` is the real tempo measure (see _team_week_pace);
  `plays_per_game` is the simple, robust, widely-cited raw-volume number.
- Red-zone play-calling split -- pass rate among real plays inside the 20
  and, separately, inside the 10.
- Target distribution by position -- share of a team's real weekly targets
  going to RB/WR/TE.

Design note -- no position gate, unlike Family 5B (opponent strength):
    Opponent strength is inherently position-vs-position (a defense's
    rating against WRs differs from its rating against RBs), so it needed
    an OPP_STRENGTH_POSITIONS gate and a (team, position, season) join.
    Team tendencies describe the OFFENSE AS A WHOLE -- every player on a
    team sees the same PROE/pace/red-zone-split regardless of their own
    position, so the join here is a plain (team, season, week) lookup, no
    position dimension, no QB-null carve-out. Target distribution by
    position is the one metric that's inherently position-shaped, but it's
    still delivered as three sibling values (RB/WR/TE share) attached to
    the TEAM, not gated per player -- these describe an offense, not a
    player. A player who joins a team inherits this environment, not the
    incumbent's usage within it.

Design note -- point-in-time safety:
    Every rate is rolled with the SAME ewm(halflife)/expanding-then-
    shift(1) recipe usage.py's build_defense_strength_table already uses,
    generalized from (team, position, season) groups to a plain (team,
    season) grouping (see the "no position gate" note above). `_ewm3`
    (recent form) and `_s2d` (season-defining, more stable) both ride
    along per metric, same as every other family in this pipeline.

Design note -- sample size, reported not implied:
    Every rate has a matching rolled SAMPLE-SIZE column (a cumulative,
    shift(1)-safe SUM of real play counts through the prior week, not a
    mean) so a caller can say plainly how much real data a number rests
    on. TEAM_TENDENCY_SPARSE_PLAYS marks a thin sample with a `sparse`
    flag at the export layer -- the same "shown, not hidden, but flagged"
    convention Phase 5's heatmap `sparse`/`~` already uses, not a hard
    eligibility gate: unlike a single player, a team has SOME real plays
    from week 1 onward, so there's no "zero data" case to gate against,
    just thin-vs-solid.

Design note -- seconds_per_play's neutral-script filter:
    Mean time between the start of consecutive offensive snaps by the same
    team WITHIN THE SAME DRIVE (game_seconds_remaining diff to the next
    real play in the same (game_id, drive)), excluding the last play of
    each drive (no "next play" to time), and two-minute-drill situations
    (half_seconds_remaining <= TWO_MINUTE_SECONDS -- deliberately fast on
    purpose, not the offense's normal tempo). Restricting to
    play_type.isin(["pass", "run"]) already excludes qb_kneel/qb_spike
    rows on both sides of the diff (verified directly: zero real 2025
    pass/run rows have qb_kneel==1 or qb_spike==1 -- nflverse tags those
    with their own play_type values, not "run"). A gap outside (0, 90]
    seconds is dropped as a real stoppage (injury, replay review, TV
    timeout), not tempo -- a reasonable, documented approximation, not a
    claim of matching any specific third-party "pace" methodology exactly.

Usage:
    from src.team_tendencies import add_team_tendency_features
    df = add_team_tendency_features(df, pbp)  # after position is known
"""

from __future__ import annotations

import logging

import pandas as pd

from src.usage import EWM_HALFLIFE, INSIDE_10_YARDLINE, RZ_YARDLINE, TWO_MINUTE_SECONDS

logger = logging.getLogger(__name__)

# A thin-but-real sample gets flagged, not hidden -- see this module's own
# docstring. 15 real plays is roughly a quarter's worth of offensive
# snaps -- enough that a single fluke play can't dominate the average, not
# a claim of statistical significance.
TEAM_TENDENCY_SPARSE_PLAYS = 15

# plays_per_game's own sample is a GAMES count (how many weekly totals this
# average has seen), not a play count -- a much lower-variance quantity, so
# it gets its own, much smaller floor rather than being held to
# TEAM_TENDENCY_SPARSE_PLAYS's play-count bar.
TEAM_TENDENCY_SPARSE_GAMES = 3

# A real stoppage (injury, replay review, TV timeout) inflates the gap
# between two snaps well past anything a real 40-second play clock could
# explain -- dropped as "not tempo," not averaged in as if it were.
_PACE_MAX_GAP_SECONDS = 90

# (raw rate column in the merged long frame, public output name, raw
# sample-size column). plays_per_game reuses total_plays as its own rate
# input (rolling its MEAN across weeks IS "plays per game") with `games`
# (one per team-week) as ITS sample size -- "how many games has this
# average seen," not "how many plays," which is the honest unit for a
# per-game average's own reliability.
_METRIC_SPECS = [
    ("proe_raw", "proe", "neutral_plays"),
    ("total_plays", "plays_per_game", "games"),
    ("seconds_per_play_raw", "seconds_per_play", "pace_gaps"),
    ("rz20_pass_rate_raw", "rz20_pass_rate", "rz20_plays"),
    ("rz10_pass_rate_raw", "rz10_pass_rate", "rz10_plays"),
    ("target_share_rb_raw", "target_share_rb", "team_targets"),
    ("target_share_wr_raw", "target_share_wr", "team_targets"),
    ("target_share_te_raw", "target_share_te", "team_targets"),
]

# Candidate model features -- rates only. Raw sample-size columns are
# deliberately excluded (a play COUNT isn't a model feature, the same
# "raw counts aren't features, only derived shares/rates are" convention
# every other family in this pipeline follows).
TEAM_TENDENCY_OUTPUT_COLUMNS = [
    f"{name}_{stat}" for _, name, _ in _METRIC_SPECS for stat in ("ewm3", "s2d")
]

# Display-only: how much real data each TEAM_TENDENCY_OUTPUT_COLUMNS rate
# rests on, keyed by its OWN sample column name (several rates share one,
# e.g. all three target_share_* columns share team_targets).
TEAM_TENDENCY_SAMPLE_COLUMNS = sorted({f"{sample}_s2d_n" for _, _, sample in _METRIC_SPECS})

# name -> which sample column its reliability is judged against, for the
# export layer to look up without re-deriving _METRIC_SPECS' own mapping.
TEAM_TENDENCY_METRIC_SAMPLE = {name: f"{sample}_s2d_n" for _, name, sample in _METRIC_SPECS}


def _team_week_play_rates(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per (team, season, week): total offensive plays, PROE (mean pass_oe)
    over real neutral-situation plays, and red-zone (<=20, <=10) pass rate
    -- plus each metric's own real sample size. REG season only, excludes
    two_point_attempt == 1 (same team-week-denominator convention as
    usage.py's _team_week_totals/_situational_team_totals). Restricting to
    play_type.isin(["pass", "run"]) excludes kneels/spikes/special teams/
    no-plays in one filter -- see this module's docstring.
    """
    reg = pbp[(pbp["season_type"] == "REG") & (pbp["two_point_attempt"] == 0)]
    pr = reg[reg["play_type"].isin(["pass", "run"])].copy()
    pr["is_pass"] = (pr["play_type"] == "pass").astype(float)

    total = (
        pr.groupby(["posteam", "season", "week"]).size().reset_index(name="total_plays")
    )

    neutral = pr[pr["pass_oe"].notna()]
    proe = (
        neutral.groupby(["posteam", "season", "week"])
        .agg(proe_raw=("pass_oe", "mean"), neutral_plays=("pass_oe", "size"))
        .reset_index()
    )

    rz20 = pr[pr["yardline_100"] <= RZ_YARDLINE]
    rz20_rate = (
        rz20.groupby(["posteam", "season", "week"])
        .agg(rz20_pass_rate_raw=("is_pass", "mean"), rz20_plays=("is_pass", "size"))
        .reset_index()
    )

    rz10 = pr[pr["yardline_100"] <= INSIDE_10_YARDLINE]
    rz10_rate = (
        rz10.groupby(["posteam", "season", "week"])
        .agg(rz10_pass_rate_raw=("is_pass", "mean"), rz10_plays=("is_pass", "size"))
        .reset_index()
    )

    out = (
        total.merge(proe, on=["posteam", "season", "week"], how="outer")
        .merge(rz20_rate, on=["posteam", "season", "week"], how="outer")
        .merge(rz10_rate, on=["posteam", "season", "week"], how="outer")
    )
    out["games"] = 1
    return out.rename(columns={"posteam": "team"})


def _team_week_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per (team, season, week): mean neutral-script seconds between
    consecutive offensive snaps (seconds_per_play_raw), and the real
    number of snap-to-snap gaps it rests on (pace_gaps). See this module's
    docstring for the exact filter (drive-boundary, two-minute-drill,
    stoppage-gap exclusions).
    """
    reg = pbp[(pbp["season_type"] == "REG") & (pbp["two_point_attempt"] == 0)]
    pr = reg[
        reg["play_type"].isin(["pass", "run"]) & (reg["half_seconds_remaining"] > TWO_MINUTE_SECONDS)
    ].copy()
    pr = pr.sort_values(["game_id", "drive", "play_id"])

    next_gsr = pr.groupby(["game_id", "drive"])["game_seconds_remaining"].shift(-1)
    gap = pr["game_seconds_remaining"] - next_gsr
    valid = next_gsr.notna() & (gap > 0) & (gap <= _PACE_MAX_GAP_SECONDS)

    pr = pr.loc[valid].assign(seconds_to_next=gap[valid])
    out = (
        pr.groupby(["posteam", "season", "week"])
        .agg(seconds_per_play_raw=("seconds_to_next", "mean"), pace_gaps=("seconds_to_next", "size"))
        .reset_index()
    )
    return out.rename(columns={"posteam": "team"})


def _team_week_target_distribution(pbp: pd.DataFrame, position_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Per (team, season, week): real target counts to RB/WR/TE and the
    team's real target total (denominator -- ALL positions, matching
    _team_week_totals's team_targets, so a trick-play target to a lineman
    doesn't silently inflate the RB/WR/TE shares). position_lookup:
    player_id, position, season (one row per player-season).
    """
    reg = pbp[(pbp["season_type"] == "REG") & (pbp["two_point_attempt"] == 0)]
    targeted = reg[(reg["pass_attempt"] == 1) & reg["receiver_player_id"].notna()]

    team_targets = (
        targeted.groupby(["posteam", "season", "week"]).size().reset_index(name="team_targets")
    )

    tagged = targeted.merge(
        position_lookup, left_on=["receiver_player_id", "season"], right_on=["player_id", "season"], how="left"
    )
    by_pos = (
        tagged[tagged["position"].isin(["RB", "WR", "TE"])]
        .groupby(["posteam", "season", "week", "position"])
        .size()
        .unstack("position", fill_value=0)
        .reindex(columns=["RB", "WR", "TE"], fill_value=0)
        .reset_index()
        .rename(columns={"RB": "targets_rb", "WR": "targets_wr", "TE": "targets_te"})
    )

    out = team_targets.merge(by_pos, on=["posteam", "season", "week"], how="left")
    for col in ("targets_rb", "targets_wr", "targets_te"):
        out[col] = out[col].fillna(0)
    out["target_share_rb_raw"] = out["targets_rb"] / out["team_targets"]
    out["target_share_wr_raw"] = out["targets_wr"] / out["team_targets"]
    out["target_share_te_raw"] = out["targets_te"] / out["team_targets"]

    return out.rename(columns={"posteam": "team"})[
        ["team", "season", "week", "team_targets", "target_share_rb_raw", "target_share_wr_raw", "target_share_te_raw"]
    ]


def _rolling_team(long_df: pd.DataFrame, value_col: str, how: str = "mean") -> pd.DataFrame:
    """
    Adds `{value_col}_ewm3`/`{value_col}_s2d` (how="mean") to long_df, the
    same window-then-shift(1) recipe as usage.py's _rolling_team_position,
    generalized to (team, season) groups (no position axis -- see this
    module's docstring). how="sum" instead adds a single
    `{value_col}_s2d_n` -- a cumulative, shift(1)-safe SUM (a real sample
    COUNT through the prior week, not an average) for the sample-size
    columns. long_df must have team, season, week, and value_col, one row
    per (team, season, week).
    """
    sorted_df = long_df.sort_values(["team", "season", "week"])
    group_keys = [sorted_df["team"], sorted_df["season"]]
    grouped = sorted_df.groupby(["team", "season"], sort=False)

    def _shift_within_group(s: pd.Series) -> pd.Series:
        return s.groupby(group_keys, sort=False).shift(1)

    if how == "sum":
        s2d_raw = grouped[value_col].expanding().sum().droplevel([0, 1])
        new_cols = pd.DataFrame(
            {f"{value_col}_s2d_n": _shift_within_group(s2d_raw).reindex(long_df.index)}, index=long_df.index
        )
    else:
        ewm_raw = grouped[value_col].ewm(halflife=EWM_HALFLIFE, min_periods=1).mean().droplevel([0, 1])
        s2d_raw = grouped[value_col].expanding().mean().droplevel([0, 1])
        new_cols = pd.DataFrame(
            {
                f"{value_col}_ewm3": _shift_within_group(ewm_raw).reindex(long_df.index),
                f"{value_col}_s2d": _shift_within_group(s2d_raw).reindex(long_df.index),
            },
            index=long_df.index,
        )
    return pd.concat([long_df, new_cols], axis=1)


def build_team_tendency_table(df: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Per (team, season, week): every TEAM_TENDENCY_OUTPUT_COLUMNS rate,
    point-in-time-safe (this week's own plays never leak into this week's
    own value), plus every TEAM_TENDENCY_SAMPLE_COLUMNS real sample-size
    count (both `.shift(1)`-ed the same way -- see _rolling_team). Null
    wherever the team has no prior in-season games yet, same convention as
    build_defense_strength_table. Public (not underscore-prefixed):
    src/export.py calls this directly for the league-wide Team Tendencies
    view.

    Args:
        df: any frame with player_id, position, season (weekly_scored or
            later is fine) -- used ONLY to resolve a target's position,
            since pbp itself has no receiver-position column.
        pbp: from get_pbp().

    Returns:
        team, season, week, plus TEAM_TENDENCY_OUTPUT_COLUMNS and
        TEAM_TENDENCY_SAMPLE_COLUMNS.
    """
    required = ["player_id", "position", "season"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"build_team_tendency_table: df is missing columns {missing}")

    position_lookup = df[["player_id", "position", "season"]].drop_duplicates(subset=["player_id", "season"])

    rates = _team_week_play_rates(pbp)
    pace = _team_week_pace(pbp)
    targets = _team_week_target_distribution(pbp, position_lookup)

    long_df = (
        rates.merge(pace, on=["team", "season", "week"], how="outer")
        .merge(targets, on=["team", "season", "week"], how="outer")
    )

    out = long_df[["team", "season", "week"]].copy()
    for raw_col, out_name, _ in _METRIC_SPECS:
        rolled = _rolling_team(long_df[["team", "season", "week", raw_col]], raw_col)
        out[f"{out_name}_ewm3"] = rolled[f"{raw_col}_ewm3"]
        out[f"{out_name}_s2d"] = rolled[f"{raw_col}_s2d"]

    for sample_col in sorted({sample for _, _, sample in _METRIC_SPECS}):
        rolled = _rolling_team(long_df[["team", "season", "week", sample_col]], sample_col, how="sum")
        out[f"{sample_col}_s2d_n"] = rolled[f"{sample_col}_s2d_n"]

    return out


def add_team_tendency_features(df: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Add TEAM_TENDENCY_OUTPUT_COLUMNS and TEAM_TENDENCY_SAMPLE_COLUMNS to a
    player-week frame, joined on (team, season, week) -- no position gate
    (see this module's docstring: every player on a team sees the SAME
    team-level values, since these describe the offense, not the player).

    Idempotent: existing output columns are dropped before recomputing.

    Args:
        df: player-week frame with player_id, position, team, season, week.
        pbp: from get_pbp().
    """
    required = ["player_id", "position", "team", "season", "week"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_team_tendency_features: df is missing columns {missing}")

    drop_cols = list(TEAM_TENDENCY_OUTPUT_COLUMNS) + list(TEAM_TENDENCY_SAMPLE_COLUMNS)
    out = df.drop(columns=[c for c in drop_cols if c in df.columns])

    table = build_team_tendency_table(df, pbp)
    return out.merge(table, on=["team", "season", "week"], how="left")
