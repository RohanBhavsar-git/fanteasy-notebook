"""
FanTeasy Stats — Feature engineering (Phase 2)

Starts with the piece everything else depends on: computing fantasy points
the way THIS league actually scores them.

Why this exists:
    nflverse ships `fantasy_points` (standard) and `fantasy_points_ppr`
    (full PPR). This league is 0.5 PPR with a stacking fumble penalty and
    return-yardage scoring, so neither column is a valid target for the
    Phase 6 model or a valid baseline against Sleeper's projections.

Design note:
    Weights are read from the league's live `scoring_settings` dict rather
    than hardcoded. If the commissioner changes scoring next season, this
    keeps working. It also mirrors index.html's computeCustomScore(), which
    matters — the notebook and the dashboard must agree.

Validation status (2025 Wk 10, 146 rostered players):
    Pre-return/kicking mapping: ~75%+ matched Sleeper exactly; all residuals
    were kickers (unmapped) or return men (unmapped). Both are now mapped.
    Re-run validate_against_sleeper() to confirm the gap closed.

Usage:
    from src.features import compute_custom_score, scoring_coverage_report

    league = get_sleeper_league()
    weekly["custom_points"] = compute_custom_score(weekly, league["scoring_settings"])
"""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sleeper scoring key -> nflverse weekly-stats column(s).
# A list means "sum these columns" (e.g. fumbles come in three flavors).
# ---------------------------------------------------------------------------
SCORING_MAP: dict[str, list[str]] = {
    # Passing
    "pass_yd":   ["passing_yards"],
    "pass_td":   ["passing_tds"],
    "pass_int":  ["passing_interceptions"],
    "pass_2pt":  ["passing_2pt_conversions"],
    "pass_cmp":  ["completions"],
    "pass_att":  ["attempts"],
    "pass_fd":   ["passing_first_downs"],
    "pass_sack": ["sacks_suffered"],

    # Rushing
    "rush_yd":   ["rushing_yards"],
    "rush_td":   ["rushing_tds"],
    "rush_2pt":  ["rushing_2pt_conversions"],
    "rush_att":  ["carries"],
    "rush_fd":   ["rushing_first_downs"],

    # Receiving
    "rec":       ["receptions"],
    "rec_yd":    ["receiving_yards"],
    "rec_td":    ["receiving_tds"],
    "rec_2pt":   ["receiving_2pt_conversions"],
    "rec_fd":    ["receiving_first_downs"],

    # Fumbles. NOTE the stacking behavior in this league:
    #   fum      = -1  fires on EVERY fumble
    #   fum_lost = -1  fires ADDITIONALLY when the fumble is lost
    # So a lost fumble costs -2 and a self-recovered fumble costs -1.
    # Built from the three offensive components rather than `fumbles_total`,
    # which may also include special-teams fumbles.
    # Uses the *_total columns, not a sum of the three offensive components.
    # Verified 2025 Wk 10: Lamar Jackson and Rashid Shaheed each had
    # fumbles_total = 1 with rushing/receiving/sack fumbles ALL zero — there
    # are fumble categories (aborted snaps, muffed returns) outside those
    # three, and summing components missed the penalty by exactly 1.00.
    "fum":        ["fumbles_total"],
    "fum_lost":   ["fumbles_lost_total"],
    "fum_rec_td": ["fumble_recovery_tds"],

    # Special teams — a skill player returning kicks. Confirmed by the
    # league's scoring page: "Player Punt/Kick Return Yards +0.03 per yard",
    # "Special teams player td +6".
    #
    # Deliberately using punt_return_yards / kickoff_return_yards rather than
    # the pt_* columns, which appear to duplicate punt returns. If validation
    # shows returners scoring double, that's the first thing to check.
    # Pick-six thrown (-1). Not in weekly stats — populate this column
    # first with add_pick_six_column(weekly, pbp).
    "pass_int_td": ["pass_int_tds"],

    "pr_yd":  ["punt_return_yards"],
    "kr_yd":  ["kickoff_return_yards"],
    "st_td":  ["special_teams_tds"],

    # Kicking. Mapped so the scorer can validate cleanly across all nine
    # roster slots — NOT because kickers should be modeled in Phase 6.
    # Kicker output depends on how often the offense stalls in FG range,
    # which is close to noise week to week.
    "fgm":     ["fg_made"],
    # NOTE: "fgmiss" is NOT here — it's distance-dependent. See DERIVED_RULES.
    "xpm":     ["pat_made"],
    # Verified 2025 Wk 10: Eddy Pineiro had 2 made / 3 att / 0 missed /
    # 1 blocked, and Sleeper charged him the -1. Blocked PATs count as misses.
    "xpmiss":  ["pat_missed", "pat_blocked"],
}

# ---------------------------------------------------------------------------
# Rules needing arithmetic rather than a plain column sum.
# Each entry: key -> (required columns, function(df) -> Series of the stat).
# ---------------------------------------------------------------------------
def _parse_distances(value) -> list[float]:
    """
    Parse nflverse's FG distance lists.

    Observed format is a semicolon-joined string ("44;28;30;22"), NaN when the
    kicker attempted none. Also handles a real list, in case the schema changes.
    """
    if isinstance(value, (list, tuple)):
        items = value
    elif isinstance(value, str):
        items = value.split(";")
    else:
        return []
    out = []
    for item in items:
        try:
            out.append(float(str(item).strip()))
        except (TypeError, ValueError):
            continue
    return out


def _fg_yards_over_30(df: pd.DataFrame) -> pd.Series:
    """
    Yards beyond 30 on made field goals, computed PER KICK.

    This is the subtle one. "Points per FG yard over 30" applies to each
    kick individually — a 22-yarder contributes 0, not -8. Computing it from
    the aggregate (total_distance - 30 * num_made) lets short kicks eat into
    long ones' credit.

    Verified 2025 Wk 10: Tyler Loop made 44;28;30;22 (124 total). Per-kick
    gives 14 yards over 30 -> 1.4 pts, matching Sleeper. The aggregate
    formula gave 4 yards -> 0.4 pts, short by exactly the 1.0 observed.
    """
    return df["fg_made_list"].map(
        lambda v: sum(max(0.0, d - 30.0) for d in _parse_distances(v))
    )


def _short_fg_misses(df: pd.DataFrame) -> pd.Series:
    """
    Count missed FGs under 50 yards — the only ones this league penalizes.

    Empirically established, not assumed. Across 2025 weeks 5/8/10/12/15,
    every kicker whose only miss was under 50 reconciled exactly with
    Sleeper (4 of 4), and every kicker with a 50+ miss was over-penalized by
    exactly 1.00 per miss (0 of 12 matched). Sleeper exposes tiered
    fgmiss_50p rules, so a zeroed long-miss tier is the likely mechanism.

    CAVEAT: the observed long misses were 51-59 and the short ones well
    under, so the exact cutoff is inferred from Sleeper's standard 50-yard
    tier boundary rather than measured. A miss at exactly 49 or 50 would
    test it.
    """
    return df["fg_missed_list"].map(
        lambda v: float(sum(1 for d in _parse_distances(v) if d < 50.0))
    )


DERIVED_RULES = {
    # +0.1 per FG yard beyond 30, per kick. See _fg_yards_over_30.
    "fgm_yds_over_30": (["fg_made_list"], _fg_yards_over_30),
    # -1 per missed FG, but only under 50 yards. See _short_fg_misses.
    "fgmiss": (["fg_missed_list"], _short_fg_misses),
}

# Scoring keys we knowingly cannot compute from weekly stats.
# Listing them explicitly means the coverage report doesn't cry wolf.
KNOWN_UNCOMPUTABLE = {
    # QB throwing a pick-six. Confirmed active ("Pick 6 Thrown -1") but needs
    # play-by-play — a defensive TD attributed back to the passer. Rare.
    # (pass_int_td moved out — it IS computable from play-by-play.
    #  See add_pick_six_column().)
    # Player-level fumble recovery (+1). The league's scoring page lists
    # Fumble Recovery only under Team Defense and Special Teams, not under
    # Misc, so it appears NOT to apply to offensive players. Left out
    # deliberately — if validation shows rostered players short by exactly 1
    # on fumble-recovery weeks, move it back into SCORING_MAP.
    "fum_rec",
}

# OPEN QUESTION — fgmiss (-1 per missed FG):
#   Across 2025 weeks 5/8/10/12/15, kickers are consistently over-scored by
#   about 1.00 per missed FG, and in Wk 10 all four such misses were 50+
#   yards (59, 53, 55, 56). Removing the penalty reconciles those rows, but
#   so would "no penalty on 50+ misses" — one week can't distinguish them,
#   and Sleeper's tiered rules normally STACK on the generic one rather than
#   replacing it. Deliberately NOT changed without a mechanism.
#   Run kicker_miss_audit() over more weeks to settle it.
#   This is low priority: kickers are out of scope for the Phase 6 model.

# NOTE_ON_DST:
#   This league has unusually detailed team-defense scoring (def_3_and_out,
#   def_4_and_stop, a tiered yds_allow_* ladder). None of it is mapped — DST
#   scoring needs team-level pbp aggregation, not player weekly stats.
#   Recommendation: scope the Phase 6 model to QB/RB/WR/TE and keep showing
#   Sleeper's numbers for K and DST, clearly labeled as Sleeper's.


def compute_custom_score(
    df: pd.DataFrame,
    scoring_settings: dict,
    warn: bool = True,
) -> pd.Series:
    """
    Compute per-row fantasy points using the league's own scoring rules.

    Args:
        df: weekly stats frame (one row per player-week)
        scoring_settings: from get_sleeper_league()["scoring_settings"]
        warn: log a warning for any non-zero offensive rule that can't be
            computed from the columns present. Leave this on — silence here
            is how a scoring bug survives to Phase 6.

    Returns:
        Series of fantasy points, aligned to df's index.
    """
    points = pd.Series(0.0, index=df.index)
    unmapped: list[str] = []

    for key, weight in scoring_settings.items():
        if not weight:                      # 0.0 rules contribute nothing
            continue
        if key in KNOWN_UNCOMPUTABLE:
            continue

        if key in DERIVED_RULES:
            needed, fn = DERIVED_RULES[key]
            if all(c in df.columns for c in needed):
                points += fn(df) * float(weight)
            else:
                unmapped.append(key)
            continue

        columns = SCORING_MAP.get(key)
        if columns is None:
            unmapped.append(key)
            continue
        present = [c for c in columns if c in df.columns]
        if not present:
            unmapped.append(key)
            continue
        points += df[present].fillna(0).sum(axis=1) * float(weight)

    if warn and unmapped:
        team_prefixes = ("def_", "st_ff", "st_fum", "idp_", "pts_allow",
                         "yds_allow", "sack", "int", "ff", "tkl", "safe",
                         "blk", "kr_td", "pr_td")
        offense_ish = [k for k in unmapped if not k.startswith(team_prefixes)]
        if offense_ish:
            logger.warning(
                "Non-zero scoring rules with no column mapping: %s. "
                "These are silently excluded from the score.",
                sorted(offense_ish),
            )

    return points


def scoring_coverage_report(df: pd.DataFrame, scoring_settings: dict) -> pd.DataFrame:
    """
    Audit which of the league's active scoring rules we can actually compute.

    Run this once and read it. It's the difference between "my scorer works"
    and "my scorer works for the rules I remembered to implement."
    """
    rows = []
    for key, weight in sorted(scoring_settings.items()):
        if not weight:
            continue
        if key in KNOWN_UNCOMPUTABLE:
            status, cols = "known-uncomputable", ""
        elif key in DERIVED_RULES:
            needed, _ = DERIVED_RULES[key]
            missing = [c for c in needed if c not in df.columns]
            status = "columns-missing" if missing else "computed (derived)"
            cols = ", ".join(needed)
        elif key not in SCORING_MAP:
            status, cols = "unmapped", ""
        else:
            present = [c for c in SCORING_MAP[key] if c in df.columns]
            status = "computed" if present else "columns-missing"
            cols = ", ".join(present)
        rows.append({"rule": key, "weight": weight, "status": status,
                     "columns": cols})
    return pd.DataFrame(rows)


def validate_against_sleeper(
    weekly: pd.DataFrame,
    crosswalk: pd.DataFrame,
    scoring_settings: dict,
    league_id: str,
    season: int,
    week: int,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """
    Compare the custom scorer against what Sleeper ACTUALLY awarded.

    This is ground truth, not an approximation — Sleeper ran the league, so
    its per-player points for a completed week are definitive. Every row
    where `diff` is 0 confirms a rule; every non-zero row points at a
    specific rule that's wrong or missing.

    Only covers rostered players, which is the right scope: you never need
    to score a defensive tackle correctly.

    Returns:
        Merged frame of Sleeper's points vs ours, sorted by absolute diff
        descending. Check `.head()` for the worst offenders and
        `(result['diff'].abs() <= tolerance).mean()` for the match rate.
    """
    import requests

    url = f"https://api.sleeper.app/v1/league/{league_id}/matchups/{week}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    actual: dict[str, float] = {}
    for matchup in resp.json():
        actual.update(matchup.get("players_points") or {})
    if not actual:
        raise RuntimeError(
            f"No player points returned for league {league_id} week {week}. "
            "Check the league ID matches the season you're validating."
        )

    act = pd.DataFrame({"sleeper_id": list(actual),
                        "sleeper_points": list(actual.values())})
    act = act.merge(
        crosswalk[["sleeper_id", "gsis_id"]].dropna(subset=["sleeper_id",
                                                            "gsis_id"]),
        on="sleeper_id", how="inner",
    )

    wk = weekly[(weekly["season"] == season) & (weekly["week"] == week)].copy()
    wk["custom_points"] = compute_custom_score(wk, scoring_settings, warn=False)

    keep = [c for c in ["player_id", "player_display_name", "position",
                        "custom_points"] if c in wk.columns]
    merged = act.merge(wk[keep], left_on="gsis_id", right_on="player_id",
                       how="inner")
    merged["diff"] = (merged["custom_points"] - merged["sleeper_points"]).round(2)

    return merged.reindex(
        merged["diff"].abs().sort_values(ascending=False).index
    )


def kicker_miss_audit(
    weekly: pd.DataFrame,
    crosswalk: pd.DataFrame,
    scoring_settings: dict,
    league_id: str,
    season: int,
    weeks,
) -> pd.DataFrame:
    """
    Gather every kicker discrepancy across several weeks, with the distances
    of their missed FGs attached.

    This exists to settle one question with evidence instead of a guess: is
    the -1 miss penalty simply not applied, or is it not applied only to
    long misses? If every row here has a diff of +1.00 per miss regardless
    of distance, `fgmiss` should come out of SCORING_MAP. If short misses
    reconcile correctly and only 50+ misses are over-scored, the rule is
    distance-dependent and needs a tiered mapping instead.
    """
    frames = []
    for week in weeks:
        res = validate_against_sleeper(
            weekly, crosswalk, scoring_settings, league_id, season, week
        )
        bad = res[(res["position"] == "K") & (res["diff"].abs() > 0.01)].copy()
        if bad.empty:
            continue
        bad["week"] = week
        wk = weekly[(weekly["season"] == season) & (weekly["week"] == week)]
        cols = [c for c in ["player_id", "fg_made_list", "fg_missed_list",
                            "fg_made", "fg_missed", "fg_blocked",
                            "pat_made", "pat_att", "pat_missed", "pat_blocked"]
                if c in wk.columns]
        frames.append(bad.merge(wk[cols], on="player_id", how="left"))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["n_missed"] = out["fg_missed_list"].map(lambda v: len(_parse_distances(v)))
    out["diff_per_miss"] = (out["diff"] / out["n_missed"].replace(0, pd.NA)).round(2)
    keep = ["week", "player_display_name", "sleeper_points", "custom_points",
            "diff", "fg_made_list", "fg_missed_list", "n_missed",
            "diff_per_miss", "pat_att", "pat_missed", "pat_blocked"]
    return out[[c for c in keep if c in out.columns]]


def add_pick_six_column(
    weekly: pd.DataFrame,
    pbp: pd.DataFrame,
    column: str = "pass_int_tds",
) -> pd.DataFrame:
    """
    Add a per-player-week count of pick-sixes thrown, derived from pbp.

    This is the one active scoring rule with no weekly-stats column. It was
    the sole cause of every skill-position discrepancy in the 2025 week
    5/8/10/12/15 validation — five QBs, each off by exactly +1.00.

    A pick-six is an interception returned for a touchdown, charged to the
    PASSER. Play-by-play has all three facts; weekly stats don't.

    Args:
        weekly: weekly stats frame (needs player_id, season, week)
        pbp: play-by-play frame from get_pbp()
        column: name for the new column

    Safe to call repeatedly. Any existing `column` is dropped first — a
    plain merge would collide and silently produce `<column>_x`/`<column>_y`
    instead, which then KeyErrors on the fillna. Notebook cells get re-run.

    Returns:
        A copy of `weekly` with the count column added (0 where none).
    """
    needed = ["interception", "return_touchdown", "passer_player_id",
              "td_team", "defteam", "season", "week"]
    missing = [c for c in needed if c not in pbp.columns]
    if missing:
        raise KeyError(
            f"pbp is missing columns needed for pick-six detection: {missing}. "
            "If you fetched pbp with a `columns=` subset, include these."
        )

    # The td_team == defteam condition is essential, not defensive coding.
    # Verified 2025 Wk 5: Cam Ward was intercepted by ARI's Taylor-Demerson,
    # who then FUMBLED the return into the end zone, where TEN's Tyler
    # Lockett recovered it for a Tennessee touchdown. Both `interception`
    # and `return_touchdown` fire, but the scoring team is the OFFENSE — the
    # opposite of a pick-six. Requiring the defense to score excludes it.
    #
    # Chose this over matching interceptor to scorer: a real pick-six with a
    # lateral to a teammate has different player IDs but still td_team ==
    # defteam, so the ID match would produce false negatives.
    picks = pbp[
        (pbp["interception"] == 1)
        & (pbp["return_touchdown"] == 1)
        & (pbp["td_team"] == pbp["defteam"])
    ]
    counts = (
        picks.dropna(subset=["passer_player_id"])
        .groupby(["passer_player_id", "season", "week"])
        .size()
        .reset_index(name=column)
        .rename(columns={"passer_player_id": "player_id"})
    )

    base = weekly.drop(columns=[column], errors="ignore")

    if counts.empty:
        # No pick-sixes in range. Return the zero column rather than skipping
        # it, so compute_custom_score() finds the mapping instead of warning
        # about an unmapped rule.
        out = base.copy()
        out[column] = 0.0
        return out

    out = base.merge(counts, on=["player_id", "season", "week"], how="left")
    out[column] = out[column].fillna(0)
    return out
