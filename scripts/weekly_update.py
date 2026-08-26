"""
FanTeasy Stats -- Phase 8: weekly-update.yml's entry point.

Inference only -- never trains anything. Loads the model artifact
retrain.yml already committed (src/artifacts.py::load_model_artifact),
fetches ONLY the current season's raw data, predicts the upcoming week, and
regenerates data/output/player_advanced_stats.json.

Why "current season only" is enough to predict a real week, even though
Phase 6/7's original point-in-time feature pipeline (Family 6's
add_rolling_features) needs prev_season_* from the season BEFORE the one
being predicted: retrain.yml's artifact carries a `history_seed` -- a small,
trimmed slice (src/pipeline.py::HISTORY_SEED_COLUMNS) of the most recent
couple of COMPLETED seasons' raw feature-input columns, embedded in the
artifact at retrain time. This script concatenates that seed with THIS
season's freshly-fetched raw features (Families 1-5 + xFP, computed the
same way retrain.py computes them, just for one season) and a target-week
stub row, then reuses src/export.py::build_target_week_features UNMODIFIED
to run add_context_features/add_rolling_features/add_trend_features over
the combined frame -- the exact same point-in-time mechanism the original
Phase 7 notebook used for a full 8-season table, just with a much smaller
"historical" base.

The tradeoff this creates, stated plainly rather than left implicit: the
candidate universe (get_export_candidates) and prev_season_* are only as
current as the LAST retrain's history_seed. A player who hasn't appeared in
the seed's ~2 seasons or in the current season isn't a candidate until the
next retrain re-seeds with fresher history. This is disclosed in the
exported JSON's meta.caveats (see WEEKLY_EXTRA_CAVEATS below), not hidden.

Phase 8 round 2 adds a `simulation` block (win probability for the target
week's real matchups, playoff-qualification odds for the rest of the
season) via src/simulate.py -- computed since Phase 6.5 but never written
anywhere until now. Null-safe by construction: get_sleeper_matchups
returns empty for a league with no real schedule yet (2026 is pre-draft as
of this writing), which short-circuits the whole simulation section before
any of it runs, so `simulation` stays None and the JSON key is `null` --
the dashboard's job is to render that as an honest empty state, not to
receive a differently-shaped payload depending on season state.

Run it locally with:
    .venv\\Scripts\\python.exe scripts\\weekly_update.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.artifacts import load_model_artifact  # noqa: E402
from src.export import (  # noqa: E402
    CAVEATS, assemble_player_advanced_stats, assemble_simulation_block, build_defense_rankings,
    build_heatmap_snapshot, build_matchup_simulation, build_matchup_snapshot, build_playoff_odds,
    build_player_simulation_metrics, build_radar_snapshot, build_starter_quantile_rows,
    build_target_week_features, build_team_game_id_lookup, build_trend_snapshot, build_usage_snapshot,
    build_weekly_matchup, build_weekly_xfp, build_xfp_summary, get_export_candidates, get_export_scope,
    predict_target_week_from_artifact, validate_export, validate_simulation,
)
from src.ingest import (  # noqa: E402
    DATA_OUTPUT, DEFAULT_LEAGUE_ID, get_id_crosswalk, get_pbp, get_schedule, get_sleeper_league,
    get_sleeper_matchups, get_sleeper_players, get_sleeper_projections, get_sleeper_rosters,
)
from src.model import predict_quantiles_with_models, sleeper_projected_points  # noqa: E402
from src.pipeline import build_raw_features, build_weekly_scored  # noqa: E402

TOP_N_FREE_AGENTS = 300

WEEKLY_EXTRA_CAVEATS = [
    "This is a weekly-inference run: it fetches only the current season plus a "
    "trailing history seed (the last couple of completed seasons' role/opportunity "
    "features) baked into the model artifact at the last retrain. A player absent "
    "from both won't appear as a projection candidate until the next retrain "
    "re-seeds with fresher history.",
    "xfp/fp_over_expected are computed from this season's plays only during weekly "
    "updates (not the multi-season window retrain.py trains against), so they run "
    "noisier in the first few weeks of a season and settle down as it accumulates "
    "its own sample.",
]


def determine_target_week(schedule_current: pd.DataFrame) -> int | None:
    """
    The upcoming week to predict: one past the most recent REG week with a
    completed score, or week 1 if none are complete yet. Returns None if
    the regular season is already fully played out (nothing left to
    predict) -- schedules are published months ahead of kickoff, so an
    empty schedule isn't expected during the season this runs in.
    """
    reg = schedule_current[schedule_current["game_type"] == "REG"]
    if reg.empty:
        raise RuntimeError(
            "No REG-season schedule rows for the current season -- can't determine "
            "which week to predict. Check get_schedule()'s response before assuming "
            "this is a real off-season gap."
        )
    completed = reg[reg["home_score"].notna() & reg["away_score"].notna()]
    max_week = int(reg["week"].max())
    target_week = int(completed["week"].max()) + 1 if not completed.empty else 1
    return target_week if target_week <= max_week else None


def build_simulation_block(
    league: dict,
    league_id: str,
    current_season: int,
    target_week: int,
    historical_features: pd.DataFrame,
    candidates: pd.DataFrame,
    schedule_current: pd.DataFrame,
    sleeper_players: pd.DataFrame,
    crosswalk: pd.DataFrame,
    cw_lookup: pd.DataFrame,
    rosters_raw: list,
    artifact: dict,
) -> dict | None:
    """
    Orchestrates src/export.py's simulation helpers into the payload's
    `simulation` block: win probability for every real matchup in
    target_week, and playoff-qualification odds for the rest of the
    regular season. Returns None (null-safe) the moment there's nothing
    real to simulate -- no draft has happened yet, so Sleeper has no
    matchups -- before touching anything else in this function.

    Predicting BEYOND target_week (every remaining regular-season week,
    for playoff odds) reuses the exact same build_target_week_features
    mechanism as target_week itself: nothing new has actually happened
    between target_week and a later week either (both are unplayed), so
    Family 6's rolling/prev_season_* features come out IDENTICAL across
    every remaining week's stub-row prediction -- only Family 5's
    per-week schedule context (opponent, home/away, spread) legitimately
    differs, which build_target_week_features already recomputes fresh
    per call since it re-runs add_context_features against the real
    schedule for whichever week it's asked about.

    Two simplifying assumptions, stated here rather than left implicit:
      - Each remaining week's starters are whatever Sleeper reports
        get_sleeper_matchups returning for THAT week -- for a genuinely
        future week (the live, in-season case this runs for) Sleeper
        carries the current default lineup forward until a manager
        actively changes it, so this is effectively "today's lineup, held
        constant" for real forward-looking runs. There is no honest way
        to predict a manager's future lineup decisions instead.
      - K/DST and any skill player missing model coverage get Sleeper's
        own point projection as a fixed, zero-variance contribution (see
        src/export.py::build_starter_quantile_rows) -- the same
        convention src/simulate.py's module docstring documents and the
        dashboard's own K/DST display already uses.
    """
    matchups_target = get_sleeper_matchups(league_id, target_week, refresh=True)
    playoff_teams = (league.get("settings") or {}).get("playoff_teams")
    playoff_week_start = (league.get("settings") or {}).get("playoff_week_start")
    if matchups_target.empty or not playoff_teams or not playoff_week_start:
        return None

    team_game_id_lookup = build_team_game_id_lookup(schedule_current)
    game_id_by_team_week = dict(zip(
        zip(team_game_id_lookup["season"], team_game_id_lookup["week"], team_game_id_lookup["team"]),
        team_game_id_lookup["game_id"],
    ))
    team_by_sleeper = dict(zip(sleeper_players["sleeper_id"], sleeper_players["team"]))
    sleeper_to_gsis = dict(zip(cw_lookup["sleeper_id"], cw_lookup["gsis_id"]))
    scoring_settings = league["scoring_settings"]

    quantiles_cache: dict[int, pd.DataFrame] = {}
    sleeper_proj_cache: dict[int, dict] = {}

    def week_quantiles(week: int) -> pd.DataFrame:
        if week not in quantiles_cache:
            week_features = build_target_week_features(
                historical_features, candidates, schedule_current, current_season, week
            )
            # MUST filter to just this week's stub rows before predicting --
            # week_features is the FULL combined frame (all of
            # historical_features plus the one stub week), and passing that
            # whole thing to predict_quantiles_with_models would predict
            # every historical row too, then silently keep an arbitrary
            # PAST week's prediction per player instead of the intended
            # future one once build_starter_quantile_rows deduplicates by
            # player_id. Same masking predict_target_week_from_artifact
            # already applies for the single-target-week path.
            test_mask = (week_features["season"] == current_season) & (week_features["week"] == week)
            quantiles_cache[week] = predict_quantiles_with_models(
                week_features[test_mask], artifact["models"], artifact["cqr_widen_by_10_90"],
                artifact["cqr_widen_by_25_75"], feature_cols=artifact["feature_columns"],
            )
        return quantiles_cache[week]

    def week_sleeper_proj_points(week: int) -> dict:
        if week not in sleeper_proj_cache:
            proj = get_sleeper_projections(current_season, week, refresh=True)
            points = sleeper_projected_points(proj, scoring_settings)
            sleeper_proj_cache[week] = dict(zip(proj["sleeper_id"], points))
        return sleeper_proj_cache[week]

    starters_by_roster_week: dict[tuple, list] = {
        (roster_id, target_week): starters
        for roster_id, starters in zip(matchups_target["roster_id"], matchups_target["starters"])
    }

    def lineup_for(roster_id, week: int) -> pd.DataFrame:
        starters = starters_by_roster_week.get((roster_id, week))
        if starters is None:
            return pd.DataFrame(columns=["game_id"])
        return build_starter_quantile_rows(
            starters, current_season, week, week_quantiles(week), week_sleeper_proj_points(week),
            sleeper_to_gsis, team_by_sleeper, game_id_by_team_week,
        )

    matchup_results = build_matchup_simulation(matchups_target, lambda rid: lineup_for(rid, target_week))

    playoff_odds = {}
    if target_week < playoff_week_start:
        remaining_weeks = []
        for week in range(target_week, playoff_week_start):
            wk_matchups = matchups_target if week == target_week else get_sleeper_matchups(
                league_id, week, refresh=True
            )
            if wk_matchups.empty:
                continue
            for roster_id, starters in zip(wk_matchups["roster_id"], wk_matchups["starters"]):
                starters_by_roster_week[(roster_id, week)] = starters
            pairs = [
                tuple(group["roster_id"].tolist())
                for _, group in wk_matchups.groupby("matchup_id")
                if len(group) == 2
            ]
            remaining_weeks.append((week, pairs))

        starting_standings = pd.DataFrame([
            {
                "roster_id": r["roster_id"],
                "wins": (r.get("settings") or {}).get("wins", 0),
                "points_for": (
                    (r.get("settings") or {}).get("fpts", 0)
                    + (r.get("settings") or {}).get("fpts_decimal", 0) / 100
                ),
            }
            for r in rosters_raw
        ])

        playoff_odds = build_playoff_odds(
            remaining_weeks, starting_standings, lineup_for, int(playoff_teams)
        )

    return assemble_simulation_block(matchup_results, playoff_odds, target_week)


def main() -> None:
    print("[1/7] Loading model artifact...")
    artifact = load_model_artifact()
    print(
        f"    model_version={artifact['model_version']} trained_at={artifact['trained_at']} "
        f"seasons_trained={artifact['seasons_trained']} history_seed_seasons={artifact['history_seed_seasons']}"
    )

    league = get_sleeper_league(DEFAULT_LEAGUE_ID, refresh=True)
    current_season = int(league["season"])
    print(f"[2/7] League {DEFAULT_LEAGUE_ID}: season={current_season} status={league.get('status')}")

    schedule_current = get_schedule([current_season], refresh=True)
    target_week = determine_target_week(schedule_current)
    if target_week is None:
        print("Regular season already fully played out for this league -- nothing to predict. Exiting.")
        return
    print(f"    target: {current_season} week {target_week}")

    print("[3/7] Fetching current-season data and building this season's raw features...")
    weekly_scored_current = build_weekly_scored([current_season], DEFAULT_LEAGUE_ID)
    raw_current = build_raw_features(weekly_scored_current, [current_season], DEFAULT_LEAGUE_ID)
    print(f"    {len(raw_current):,} real rows played so far this season")

    history_seed = artifact["history_seed"]
    historical_features = (
        pd.concat([history_seed, raw_current], ignore_index=True, sort=False)
        if not raw_current.empty else history_seed.copy()
    )

    sleeper_players = get_sleeper_players()
    crosswalk = get_id_crosswalk()

    print("[4/7] Building target-week features and predicting...")
    candidates, candidate_report = get_export_candidates(historical_features, sleeper_players, crosswalk)
    print(f"    candidate report: {candidate_report}")
    if candidates.empty:
        raise RuntimeError("Zero export candidates -- aborting rather than writing an empty JSON.")

    combined_features = build_target_week_features(
        historical_features, candidates, schedule_current, current_season, target_week
    )
    predictions = predict_target_week_from_artifact(combined_features, current_season, target_week, artifact)
    if predictions.empty:
        raise RuntimeError("predict_target_week_from_artifact returned zero rows -- aborting.")
    assert (predictions["floor"] <= predictions["point"]).all()
    assert (predictions["point"] <= predictions["ceiling"]).all()
    print(f"    {len(predictions)} predictions, floor <= point <= ceiling holds")

    usage = build_usage_snapshot(combined_features, current_season, target_week)
    trend = build_trend_snapshot(combined_features, current_season, target_week)
    matchup = build_matchup_snapshot(combined_features, current_season, target_week)
    defense_rankings = build_defense_rankings(combined_features, schedule_current, current_season, target_week)
    n_ranked = {pos: len(teams) for pos, teams in defense_rankings.items()}
    print(f"    defense rankings: {n_ranked} teams ranked (of 32) per position")

    # Most recently COMPLETED season's xFP retrospective while nothing has
    # been played yet this season (raw_current empty); once real games
    # exist this season, show THIS season's running total instead -- see
    # this module's docstring for why that's a deliberate behavior change
    # from the original single pre-season export.
    xfp_season = current_season if not raw_current.empty else current_season - 1
    xfp_summary = build_xfp_summary(historical_features, xfp_season)

    # Per-week xfp for THIS season's played weeks -- always current_season,
    # unlike xfp_summary's xfp_season fallback above, since the Weekly
    # Production chart only ever plots current_season's own bars. Honestly
    # empty (not an error) if current_season has zero games played yet.
    weekly_xfp = build_weekly_xfp(historical_features, current_season)
    print(f"    weekly xfp: {len(weekly_xfp)} (player, week) rows for season {current_season}")

    # Real per-played-week matchup history for THIS season (empty in the
    # real pre-draft/week-1 case this run actually hits) -- lets a reader
    # browsing an already-played week of the LIVE season see that week's
    # real matchup, same shape/purpose as scripts/archive_season.py's own
    # use of this for a completed season. `matchup` above stays the single
    # UPCOMING-week block; this is the historical complement.
    weekly_matchup = build_weekly_matchup(historical_features, current_season)
    print(f"    weekly matchup: {len(weekly_matchup)} (player, week) rows for season {current_season}")

    print("[5/7] Scoping to real rosters + top free agents, assembling JSON...")
    rosters_raw = get_sleeper_rosters(DEFAULT_LEAGUE_ID, refresh=True)
    rostered_sleeper_ids = {pid for r in rosters_raw for pid in (r.get("players") or [])}
    cw_lookup = crosswalk.dropna(subset=["sleeper_id", "gsis_id"]).drop_duplicates(subset=["sleeper_id"])
    sleeper_to_gsis = dict(zip(cw_lookup["sleeper_id"], cw_lookup["gsis_id"]))
    rostered_gsis_ids = {sleeper_to_gsis[sid] for sid in rostered_sleeper_ids if sid in sleeper_to_gsis}

    scoped_predictions, scope_report = get_export_scope(rostered_gsis_ids, predictions, top_n=TOP_N_FREE_AGENTS)
    print(f"    scope report: {scope_report}")

    # Phase 4: radar percentiles, against this league's REAL roster_positions
    # and real team count -- not a guessed/default league shape.
    radar = build_radar_snapshot(
        combined_features, current_season, target_week, league["roster_positions"], len(rosters_raw)
    )
    n_eligible = sum(1 for r in radar.values() if r["eligible"])
    print(f"    radar: {n_eligible}/{len(radar)} candidates eligible (>= games played floor)")

    # Phase 5: field heatmap zones -- needs raw pbp for current_season.
    # build_raw_features (Step 3 above) does NOT already have this cached:
    # its `if weekly_scored.empty: return weekly_scored` early-out means it
    # never calls get_pbp at all when raw_current is empty (a not-yet-
    # started season, this run's real case) -- so this is a genuine first
    # fetch, not a cache read.
    #
    # nflreadpy.load_pbp() raises ValueError("Season must be between 1999
    # and <its own current-season guess>") for a season that hasn't
    # started yet -- a client-side check, before any network call, so it's
    # not the ConnectionError/404 shape _is_unpublished_season_error
    # already handles for get_weekly_stats. Same real condition
    # (build_weekly_scored's own current_season fetch above already logged
    # "0 real rows played so far this season"), so it gets the same
    # tolerance, scoped to just this call -- not pushed into get_pbp
    # itself, matching why build_weekly_scored's own tolerance lives at
    # its caller and not inside get_weekly_stats (see that function's
    # docstring): every OTHER get_pbp caller in this pipeline requests
    # seasons known to be published, where this error would be a real bug
    # that should still fail loudly.
    try:
        pbp_current = get_pbp([current_season])
    except ValueError as e:
        if "must be between" not in str(e):
            raise
        print(f"    pbp: season {current_season} has no published play-by-play yet -- heatmap zones will be empty.")
        pbp_current = pd.DataFrame()  # build_heatmap_snapshot treats an empty pbp as "nothing to zone" -- see its own comment
    heatmap = build_heatmap_snapshot(combined_features, pbp_current, current_season, target_week)
    n_heatmap_eligible = sum(1 for h in heatmap.values() if h["eligible"])
    print(f"    heatmap: {n_heatmap_eligible}/{len(heatmap)} candidates eligible (>= games played floor)")

    # Per-player Monte Carlo: boom/bust + threshold probabilities, plus the
    # game_id + quantiles the dashboard needs for an ad-hoc start-over-
    # replacement comparison between any two exported players (see
    # build_player_simulation_metrics's own docstring for why raw draws
    # aren't exported). Independent of build_simulation_block below -- this
    # covers the FULL candidate pool, not just this week's real starters.
    player_sim_metrics = build_player_simulation_metrics(
        combined_features, schedule_current, current_season, target_week, artifact
    )
    print(f"    player Monte Carlo metrics: {len(player_sim_metrics)} candidates")

    payload, crosswalk_report = assemble_player_advanced_stats(
        scoped_predictions, usage, trend, xfp_summary, weekly_xfp, radar, heatmap, crosswalk,
        current_season, target_week, xfp_season, artifact["seasons_trained"], artifact["model_version"],
        performance=artifact["performance"],
        caveats=CAVEATS + WEEKLY_EXTRA_CAVEATS,
        player_sim_metrics=player_sim_metrics,
        matchup=matchup,
        defense_rankings=defense_rankings,
        weekly_matchup=weekly_matchup,
    )
    print(f"    crosswalk match rate: {crosswalk_report}")

    validation_report = validate_export(payload, crosswalk)
    print(f"    validation: {validation_report}")

    print("[6/7] Simulating matchup win probability + playoff odds...")
    simulation = build_simulation_block(
        league, DEFAULT_LEAGUE_ID, current_season, target_week, historical_features, candidates,
        schedule_current, sleeper_players, crosswalk, cw_lookup, rosters_raw, artifact,
    )
    if simulation is not None:
        sim_validation = validate_simulation(simulation)
        print(f"    simulation: {sim_validation}")
    else:
        print("    no real matchups for this league yet (pre-draft/off-season) -- simulation is null")
    payload["simulation"] = simulation

    print("[7/7] Writing output...")
    out_path = DATA_OUTPUT / "player_advanced_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)

    with open(out_path) as f:
        reparsed = json.load(f)
    assert reparsed == payload, "JSON round-trip mismatch -- refusing to treat this write as clean."

    size_bytes = out_path.stat().st_size
    print(f"Wrote {out_path} ({size_bytes:,} bytes, {len(payload['players'])} players)")


if __name__ == "__main__":
    main()
