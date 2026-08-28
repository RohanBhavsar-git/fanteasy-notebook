"""
FanTeasy Stats -- season archive generator.

Produces data/output/archive/{season}.json: a frozen, final snapshot of one
COMPLETED season, same schema as data/output/player_advanced_stats.json
(projection/usage/trend/xfp/radar/heatmap per player), built with the exact
same point-in-time-safe machinery scripts/weekly_update.py uses for a live
week -- just with target_week set to one week PAST that season's actual
final real week, so every rolling/radar/heatmap aggregate reflects the
WHOLE real season, matching what the dashboard's live-Sleeper-sourced KPI
cards already show for an archived season. Predicting that hypothetical
"week after the season ended" is never leakage (see determine_target_week's
own docstring in weekly_update.py for the identical reasoning on the live
path) -- there IS no real row at or after it to leak from, by construction.

`simulation` is always null for an archive -- win probability / playoff
odds for a week that never happened isn't a real thing to compute, so this
script doesn't try.

Uses the already-committed model artifact (same one weekly_update.py
uses) rather than training a fresh one -- an archive is a snapshot with
today's best model, not a retrain. Reads the full 2018-2025
weekly_features.parquet directly (03_usage_features.ipynb's output)
instead of weekly_update.py's constrained history_seed approach: this is
a manual, occasional script, not a CI job under a fetch-cost budget, and
the full cached history is already on disk.

Run manually, per season, as new seasons complete:
    .venv\\Scripts\\python.exe scripts\\archive_season.py 2025

See src/ingest.py's SEASON_LEAGUE_IDS for which seasons are currently
archivable (this league's real Sleeper history) and PROJECT_CONTEXT.md's
Phase 9 findings (season archives + selector) for why only 2023-2025 are
actually shipped as of this writing despite 2021-2022 also existing in
SEASON_LEAGUE_IDS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.artifacts import load_model_artifact  # noqa: E402
from src.export import (  # noqa: E402
    CAVEATS, assemble_player_advanced_stats, build_heatmap_snapshot, build_radar_snapshot,
    build_season_defense_rankings, build_season_team_tendencies, build_target_week_features,
    build_trend_snapshot, build_usage_snapshot, build_weekly_matchup, build_weekly_xfp, build_xfp_summary,
    get_archive_candidates, get_export_scope, get_season_team_map, predict_target_week_from_artifact,
    validate_export,
)
from src.ingest import (  # noqa: E402
    DATA_OUTPUT, SEASON_LEAGUE_IDS, get_id_crosswalk, get_pbp, get_schedule, get_sleeper_league,
    get_sleeper_rosters,
)

TOP_N_FREE_AGENTS = 300

ARCHIVE_EXTRA_CAVEATS = [
    "This is a final, frozen snapshot of a completed season -- not a live projection. "
    "The projection.point/floor/ceiling fields describe a hypothetical week after the "
    "season ended and are not meaningful for this season; usage/trend/radar/heatmap "
    "reflect the player's real, complete season.",
]


def determine_archive_target_week(schedule_season: pd.DataFrame, season: int) -> int:
    """One past this season's real final REG week -- raises if the season
    isn't actually over yet (a real bug: don't archive a season that's still
    in progress)."""
    reg = schedule_season[schedule_season["game_type"] == "REG"]
    if reg.empty:
        raise RuntimeError(f"No REG-season schedule rows for {season} -- can't archive a season with no schedule.")
    completed = reg[reg["home_score"].notna() & reg["away_score"].notna()]
    if completed.empty:
        raise RuntimeError(f"Season {season} has no completed REG games yet -- not archivable.")
    max_week = int(reg["week"].max())
    final_week = int(completed["week"].max())
    if final_week < max_week:
        raise RuntimeError(
            f"Season {season}'s REG season isn't fully played yet (last completed week {final_week} of {max_week}) "
            f"-- archive a season only after it's actually over."
        )
    return final_week + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", type=int, help="Completed season to archive, e.g. 2025")
    args = parser.parse_args()
    season = args.season

    if season not in SEASON_LEAGUE_IDS:
        raise RuntimeError(
            f"season {season} has no entry in src/ingest.py's SEASON_LEAGUE_IDS -- "
            f"known seasons: {sorted(SEASON_LEAGUE_IDS)}. Walk DEFAULT_LEAGUE_ID's "
            f"previous_league_id chain to find it before adding one."
        )
    league_id = SEASON_LEAGUE_IDS[season]

    print(f"[1/6] Loading model artifact + real {season} league/schedule/rosters (league {league_id})...")
    artifact = load_model_artifact()
    league = get_sleeper_league(league_id, refresh=True)
    assert int(league["season"]) == season, f"league {league_id} reports season {league['season']}, expected {season}"
    schedule_season = get_schedule([season], refresh=True)
    rosters_raw = get_sleeper_rosters(league_id, refresh=True)
    target_week = determine_archive_target_week(schedule_season, season)
    print(f"    target: season {season}, one past its real final week -> stub week {target_week}")

    print("[2/6] Loading full cached historical features (2018-2025, from 03_usage_features.ipynb)...")
    weekly_features = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "weekly_features.parquet")
    print(f"    weekly_features: {len(weekly_features):,} rows, seasons {weekly_features['season'].min()}-{weekly_features['season'].max()}")
    season_rows = weekly_features[weekly_features["season"] == season]
    if season_rows.empty:
        raise RuntimeError(
            f"weekly_features.parquet has zero rows for season {season} -- rebuild it "
            f"(03_usage_features.ipynb) before archiving a season it doesn't cover yet."
        )
    print(f"    {len(season_rows):,} real {season} rows, weeks {int(season_rows['week'].min())}-{int(season_rows['week'].max())}")

    # No get_sleeper_players() fetch here -- get_archive_candidates resolves
    # `team` from this player's own historical rows, not from today's live
    # Sleeper player DB (see its own docstring for why that matters).
    crosswalk = get_id_crosswalk()

    print("[3/6] Building archive candidates (team from history, not from today's Sleeper snapshot)...")
    candidates, candidate_report = get_archive_candidates(weekly_features, crosswalk)
    print(f"    candidate report: {candidate_report}")
    if candidates.empty:
        raise RuntimeError("Zero archive candidates -- aborting rather than writing an empty archive.")

    season_team = get_season_team_map(weekly_features, season)
    print(f"    season-team map: {len(season_team)} players resolved to their real {season} team")

    print("[4/6] Building target-week features and predicting...")
    combined_features = build_target_week_features(weekly_features, candidates, schedule_season, season, target_week)
    predictions = predict_target_week_from_artifact(combined_features, season, target_week, artifact)
    if predictions.empty:
        raise RuntimeError("predict_target_week_from_artifact returned zero rows -- aborting.")
    assert (predictions["floor"] <= predictions["point"]).all()
    assert (predictions["point"] <= predictions["ceiling"]).all()
    print(f"    {len(predictions)} predictions, floor <= point <= ceiling holds")

    usage = build_usage_snapshot(combined_features, season, target_week)
    trend = build_trend_snapshot(combined_features, season, target_week)
    xfp_summary = build_xfp_summary(weekly_features, season)
    weekly_xfp = build_weekly_xfp(weekly_features, season)
    weekly_matchup = build_weekly_matchup(weekly_features, season)
    print(f"    weekly matchup: {len(weekly_matchup)} (player, week) rows for season {season}")

    print("[5/6] Building radar + heatmap (real pbp for the season)...")
    radar = build_radar_snapshot(combined_features, season, target_week, league["roster_positions"], len(rosters_raw))
    n_radar_eligible = sum(1 for r in radar.values() if r["eligible"])
    print(f"    radar: {n_radar_eligible}/{len(radar)} candidates eligible")

    pbp_season = get_pbp([season])
    heatmap = build_heatmap_snapshot(combined_features, pbp_season, season, target_week)
    n_heatmap_eligible = sum(1 for h in heatmap.values() if h["eligible"])
    print(f"    heatmap: {n_heatmap_eligible}/{len(heatmap)} candidates eligible")

    # Family 5B: a full, real, COMPLETED season's defense rankings --
    # NOT build_defense_rankings at the stub week (see
    # build_season_defense_rankings's own docstring for why that would
    # come back empty: the stub week has no real schedule game to resolve
    # an opponent from). The single upcoming-week `matchup` block has no
    # coherent retrospective meaning for an archive either (same "a
    # hypothetical week that never happened isn't a real thing to compute"
    # reasoning already applied to `simulation` below) and is deliberately
    # left null -- `weekly_matchup` above is what gives an archive a REAL
    # per-player matchup, one per actually-played week instead.
    defense_rankings = build_season_defense_rankings(weekly_features, schedule_season, season)
    print(f"    defense rankings: { {pos: len(teams) for pos, teams in defense_rankings.items()} } teams ranked (of 32) per position")

    # Same "full real season, unshifted, nothing left to leak" reasoning as
    # build_season_defense_rankings just above -- reuses pbp_season fetched
    # for the heatmap above.
    team_tendencies = build_season_team_tendencies(weekly_features, pbp_season, season)
    print(f"    team tendencies: {len(team_tendencies)}/32 teams")

    print("[6/6] Scoping, assembling, validating, writing...")
    rostered_sleeper_ids = {pid for r in rosters_raw for pid in (r.get("players") or [])}
    cw_lookup = crosswalk.dropna(subset=["sleeper_id", "gsis_id"]).drop_duplicates(subset=["sleeper_id"])
    sleeper_to_gsis = dict(zip(cw_lookup["sleeper_id"], cw_lookup["gsis_id"]))
    rostered_gsis_ids = {sleeper_to_gsis[sid] for sid in rostered_sleeper_ids if sid in sleeper_to_gsis}

    scoped_predictions, scope_report = get_export_scope(rostered_gsis_ids, predictions, top_n=TOP_N_FREE_AGENTS)
    print(f"    scope report: {scope_report}")

    payload, crosswalk_report = assemble_player_advanced_stats(
        scoped_predictions, usage, trend, xfp_summary, weekly_xfp, radar, heatmap, crosswalk,
        season, target_week, season, artifact["seasons_trained"], artifact["model_version"],
        performance=artifact["performance"],
        caveats=CAVEATS + ARCHIVE_EXTRA_CAVEATS,
        season_team=season_team,
        defense_rankings=defense_rankings,
        weekly_matchup=weekly_matchup,
        team_tendencies=team_tendencies,
    )
    payload["simulation"] = None  # a hypothetical post-season week has no real matchups to simulate
    print(f"    crosswalk match rate: {crosswalk_report}")

    validation_report = validate_export(payload, crosswalk)
    print(f"    validation: {validation_report}")

    out_path = DATA_OUTPUT / "archive" / f"{season}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)

    with open(out_path) as f:
        reparsed = json.load(f)
    assert reparsed == payload, "JSON round-trip mismatch -- refusing to treat this write as clean."

    size_bytes = out_path.stat().st_size
    print(f"\nWrote {out_path} ({size_bytes:,} bytes, {len(payload['players'])} players)")


if __name__ == "__main__":
    main()
