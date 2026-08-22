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
    CAVEATS, assemble_player_advanced_stats, build_target_week_features, build_trend_snapshot,
    build_usage_snapshot, build_xfp_summary, get_export_candidates, get_export_scope,
    predict_target_week_from_artifact, validate_export,
)
from src.ingest import (  # noqa: E402
    DATA_OUTPUT, DEFAULT_LEAGUE_ID, get_id_crosswalk, get_schedule, get_sleeper_league,
    get_sleeper_players, get_sleeper_rosters,
)
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


def main() -> None:
    print("[1/6] Loading model artifact...")
    artifact = load_model_artifact()
    print(
        f"    model_version={artifact['model_version']} trained_at={artifact['trained_at']} "
        f"seasons_trained={artifact['seasons_trained']} history_seed_seasons={artifact['history_seed_seasons']}"
    )

    league = get_sleeper_league(DEFAULT_LEAGUE_ID, refresh=True)
    current_season = int(league["season"])
    print(f"[2/6] League {DEFAULT_LEAGUE_ID}: season={current_season} status={league.get('status')}")

    schedule_current = get_schedule([current_season], refresh=True)
    target_week = determine_target_week(schedule_current)
    if target_week is None:
        print("Regular season already fully played out for this league -- nothing to predict. Exiting.")
        return
    print(f"    target: {current_season} week {target_week}")

    print("[3/6] Fetching current-season data and building this season's raw features...")
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

    print("[4/6] Building target-week features and predicting...")
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

    # Most recently COMPLETED season's xFP retrospective while nothing has
    # been played yet this season (raw_current empty); once real games
    # exist this season, show THIS season's running total instead -- see
    # this module's docstring for why that's a deliberate behavior change
    # from the original single pre-season export.
    xfp_season = current_season if not raw_current.empty else current_season - 1
    xfp_summary = build_xfp_summary(historical_features, xfp_season)

    print("[5/6] Scoping to real rosters + top free agents, assembling JSON...")
    rosters_raw = get_sleeper_rosters(DEFAULT_LEAGUE_ID, refresh=True)
    rostered_sleeper_ids = {pid for r in rosters_raw for pid in (r.get("players") or [])}
    cw_lookup = crosswalk.dropna(subset=["sleeper_id", "gsis_id"]).drop_duplicates(subset=["sleeper_id"])
    sleeper_to_gsis = dict(zip(cw_lookup["sleeper_id"], cw_lookup["gsis_id"]))
    rostered_gsis_ids = {sleeper_to_gsis[sid] for sid in rostered_sleeper_ids if sid in sleeper_to_gsis}

    scoped_predictions, scope_report = get_export_scope(rostered_gsis_ids, predictions, top_n=TOP_N_FREE_AGENTS)
    print(f"    scope report: {scope_report}")

    payload, crosswalk_report = assemble_player_advanced_stats(
        scoped_predictions, usage, trend, xfp_summary, crosswalk,
        current_season, target_week, artifact["seasons_trained"], artifact["model_version"],
        performance=artifact["performance"],
        caveats=CAVEATS + WEEKLY_EXTRA_CAVEATS,
    )
    print(f"    crosswalk match rate: {crosswalk_report}")

    validation_report = validate_export(payload, crosswalk)
    print(f"    validation: {validation_report}")

    print("[6/6] Writing output...")
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
