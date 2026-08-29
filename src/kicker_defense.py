"""
FanTeasy Stats — Kicker and team-defense descriptive stats.

Neither K nor DEF is part of the point-in-time player-week feature
pipeline in src/usage.py at all -- FANTASY_POSITIONS there is
deliberately ("QB", "RB", "WR", "TE") only, and K/DST are out of scope
for the projection model (see CLAUDE.md's scope boundaries: kicker output
is close to noise week to week, and a defense would need a team-defense
model layered on an offense model). This module is NOT a projection
model for either position -- it's real, honest, season-to-date DESCRIPTIVE
stats for the dashboard's Player Detail page, replacing "Opportunity
Shares" (snap %/target share/carry share/red-zone share), which are
meaningless for a kicker or a team unit the same way they're meaningless
for a QB who takes every offensive snap.

No point-in-time-safety concern here the way Family 1-7 have one: these
are season-to-date summaries for a human reader (the same "real total,
not a model feature" reasoning src/export.py's build_xfp_summary already
uses), not something a model trains on, so there's nothing to leak.

Usage:
    from src.kicker_defense import build_kicker_season_stats, build_defense_season_stats
    kicker_stats = build_kicker_season_stats(weekly_stats, season)
    defense_stats = build_defense_season_stats(pbp, schedule, season)
"""

from __future__ import annotations

import pandas as pd

KICKER_STATS_OUTPUT_COLUMNS = [
    "games_played", "fg_made", "fg_att", "pat_made", "pat_att",
    "pat_rate", "attempts_per_game",
    "fg_rate_under_30", "fg_rate_30_39", "fg_rate_40_49", "fg_rate_50_plus",
]

DEFENSE_STATS_OUTPUT_COLUMNS = [
    "games_played", "sacks", "interceptions",
    "pass_yards_allowed_per_game", "rush_yards_allowed_per_game", "points_allowed_per_game",
]


def build_kicker_season_stats(weekly_stats: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Season-cumulative kicker stats, summed over whatever REG weeks of
    `season` are present in `weekly_stats` (naturally "through now" for an
    incomplete season -- an unplayed week simply isn't a row yet).

    FG rate is reported by distance band (under 30, 30-39, 40-49, 50+),
    not one blended rate -- a kicker's make rate falls off sharply with
    distance (the same reasoning src/features.py's fgmiss rule already
    treats distance as load-bearing). nflverse's weekly stats already
    ship these bands pre-split (fg_made_0_19/fg_made_20_29/.../
    fg_missed_0_19/...), so this sums those directly rather than
    re-parsing fg_made_list/fg_missed_list's semicolon-joined distance
    strings.

    Per-band attempts are made+missed only -- blocked kicks aren't
    pre-binned by distance in nflverse's weekly stats (only
    fg_blocked_list/fg_blocked_distance exist, no fg_blocked_<band>
    columns), and blocks are rare enough league-wide not to meaningfully
    bias a per-band rate. `fg_att` (the official stat, used for
    attempts_per_game) DOES include blocks, so attempts_per_game and the
    sum of per-band attempts won't reconcile to the exact same total --
    a disclosed, deliberate simplification, not a bug.

    Args:
        weekly_stats: raw weekly player stats from get_weekly_stats() --
            UNFILTERED by position (K is never in weekly_scored/
            FANTASY_POSITIONS, so this needs the raw frame directly, not
            the usual QB/RB/WR/TE-filtered pipeline output).
        season: which season to summarize.

    Returns:
        One row per real K player_id (gsis_id) with player_display_name,
        team, and KICKER_STATS_OUTPUT_COLUMNS. Empty (correctly-shaped)
        DataFrame if no K rows exist for this season yet.
    """
    required = ["player_id", "player_display_name", "team", "position", "season", "week", "season_type"]
    missing = [c for c in required if c not in weekly_stats.columns]
    if missing:
        raise KeyError(f"build_kicker_season_stats: weekly_stats is missing columns {missing}")

    reg = weekly_stats[
        (weekly_stats["season_type"] == "REG")
        & (weekly_stats["season"] == season)
        & (weekly_stats["position"] == "K")
    ]
    if reg.empty:
        return pd.DataFrame(columns=["player_id", "player_display_name", "team"] + KICKER_STATS_OUTPUT_COLUMNS)

    def _col(name: str) -> pd.Series:
        return reg[name].fillna(0) if name in reg.columns else pd.Series(0, index=reg.index)

    made_under_30 = _col("fg_made_0_19") + _col("fg_made_20_29")
    missed_under_30 = _col("fg_missed_0_19") + _col("fg_missed_20_29")
    made_50_plus = _col("fg_made_50_59") + _col("fg_made_60_")
    missed_50_plus = _col("fg_missed_50_59") + _col("fg_missed_60_")

    tmp = pd.DataFrame({
        "player_id": reg["player_id"].to_numpy(),
        "player_display_name": reg["player_display_name"].to_numpy(),
        "team": reg["team"].to_numpy(),
        "week": reg["week"].to_numpy(),
        "fg_made": _col("fg_made").to_numpy(),
        "fg_att": _col("fg_att").to_numpy(),
        "pat_made": _col("pat_made").to_numpy(),
        "pat_att": _col("pat_att").to_numpy(),
        "made_under_30": made_under_30.to_numpy(), "att_under_30": (made_under_30 + missed_under_30).to_numpy(),
        "made_30_39": _col("fg_made_30_39").to_numpy(), "att_30_39": (_col("fg_made_30_39") + _col("fg_missed_30_39")).to_numpy(),
        "made_40_49": _col("fg_made_40_49").to_numpy(), "att_40_49": (_col("fg_made_40_49") + _col("fg_missed_40_49")).to_numpy(),
        "made_50_plus": made_50_plus.to_numpy(), "att_50_plus": (made_50_plus + missed_50_plus).to_numpy(),
    })

    agg = tmp.groupby(["player_id", "player_display_name"], as_index=False).agg(
        team=("team", "last"),
        games_played=("week", "nunique"),
        fg_made=("fg_made", "sum"), fg_att=("fg_att", "sum"),
        pat_made=("pat_made", "sum"), pat_att=("pat_att", "sum"),
        made_under_30=("made_under_30", "sum"), att_under_30=("att_under_30", "sum"),
        made_30_39=("made_30_39", "sum"), att_30_39=("att_30_39", "sum"),
        made_40_49=("made_40_49", "sum"), att_40_49=("att_40_49", "sum"),
        made_50_plus=("made_50_plus", "sum"), att_50_plus=("att_50_plus", "sum"),
    )

    agg["pat_rate"] = (agg["pat_made"] / agg["pat_att"]).where(agg["pat_att"] > 0)
    agg["attempts_per_game"] = agg["fg_att"] / agg["games_played"]
    agg["fg_rate_under_30"] = (agg["made_under_30"] / agg["att_under_30"]).where(agg["att_under_30"] > 0)
    agg["fg_rate_30_39"] = (agg["made_30_39"] / agg["att_30_39"]).where(agg["att_30_39"] > 0)
    agg["fg_rate_40_49"] = (agg["made_40_49"] / agg["att_40_49"]).where(agg["att_40_49"] > 0)
    agg["fg_rate_50_plus"] = (agg["made_50_plus"] / agg["att_50_plus"]).where(agg["att_50_plus"] > 0)

    return agg[["player_id", "player_display_name", "team"] + KICKER_STATS_OUTPUT_COLUMNS]


def build_defense_season_stats(pbp: pd.DataFrame, schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Season-cumulative team-defense box-score stats: sacks and
    interceptions (summed directly from real play-by-play, attributed to
    `defteam`), and passing/rushing yards allowed + points allowed
    (per-game averages). No team-defense row exists anywhere in
    nflverse's PLAYER weekly stats (a defense isn't a player), so this is
    genuinely new aggregation from pbp + the schedule's real final
    scores, not a re-export of something already computed elsewhere.

    games_played comes from the SCHEDULE's real completed games (not from
    counting pbp weeks), matching the same "a games count is a
    lower-variance quantity, don't derive it from a play-count source"
    reasoning src/team_tendencies.py's plays_per_game already uses.

    Args:
        pbp: play-by-play frame from get_pbp(), covering `season`.
        schedule: from get_schedule(), covering `season`.
        season: which season to summarize.

    Returns:
        One row per real NFL team (32, once the season has any completed
        games) with team, and DEFENSE_STATS_OUTPUT_COLUMNS.
    """
    required_sched = ["game_type", "season", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required_sched if c not in schedule.columns]
    if missing:
        raise KeyError(f"build_defense_season_stats: schedule is missing columns {missing}")

    if pbp.empty:
        # Season hasn't started yet (weekly_update.py's own get_pbp caller
        # already turns "no pbp published yet" into a genuinely-empty
        # DataFrame -- see its own comment -- so an empty pbp reaching here
        # IS the real "nothing to aggregate yet" signal, not a bug to fail
        # loudly on). Skip straight to schedule-only output -- there's
        # nothing to group by on a frame with no rows (and, this early,
        # typically no columns either).
        sacks_ints = pd.DataFrame(columns=["team", "sacks", "interceptions"])
        pass_yards = pd.DataFrame(columns=["team", "pass_yards_allowed_total"])
        rush_yards = pd.DataFrame(columns=["team", "rush_yards_allowed_total"])
    else:
        required_pbp = ["season_type", "season", "defteam", "play_type", "sack", "interception", "passing_yards", "rushing_yards"]
        missing = [c for c in required_pbp if c not in pbp.columns]
        if missing:
            raise KeyError(f"build_defense_season_stats: pbp is missing columns {missing}")

        reg_pbp = pbp[(pbp["season_type"] == "REG") & (pbp["season"] == season)]

        sacks_ints = (
            reg_pbp.groupby("defteam")[["sack", "interception"]].sum()
            .rename_axis("team").reset_index()
            .rename(columns={"sack": "sacks", "interception": "interceptions"})
        )
        pass_yards = (
            reg_pbp[reg_pbp["play_type"] == "pass"].groupby("defteam")["passing_yards"].sum()
            .rename_axis("team").reset_index().rename(columns={"passing_yards": "pass_yards_allowed_total"})
        )
        rush_yards = (
            reg_pbp[reg_pbp["play_type"] == "run"].groupby("defteam")["rushing_yards"].sum()
            .rename_axis("team").reset_index().rename(columns={"rushing_yards": "rush_yards_allowed_total"})
        )

    reg_sched = schedule[(schedule["game_type"] == "REG") & (schedule["season"] == season)]
    completed = reg_sched[reg_sched["home_score"].notna() & reg_sched["away_score"].notna()]
    home = completed[["home_team", "away_score"]].rename(columns={"home_team": "team", "away_score": "points_allowed"})
    away = completed[["away_team", "home_score"]].rename(columns={"away_team": "team", "home_score": "points_allowed"})
    points_allowed = pd.concat([home, away], ignore_index=True)
    games_and_points = (
        points_allowed.groupby("team")
        .agg(games_played=("points_allowed", "size"), points_allowed_total=("points_allowed", "sum"))
        .reset_index()
    )

    out = games_and_points.merge(sacks_ints, on="team", how="left")
    out = out.merge(pass_yards, on="team", how="left")
    out = out.merge(rush_yards, on="team", how="left")

    for col in ["sacks", "interceptions", "pass_yards_allowed_total", "rush_yards_allowed_total"]:
        out[col] = out[col].fillna(0)

    out["pass_yards_allowed_per_game"] = out["pass_yards_allowed_total"] / out["games_played"]
    out["rush_yards_allowed_per_game"] = out["rush_yards_allowed_total"] / out["games_played"]
    out["points_allowed_per_game"] = out["points_allowed_total"] / out["games_played"]

    return out[["team"] + DEFENSE_STATS_OUTPUT_COLUMNS]
