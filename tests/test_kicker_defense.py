"""
Tests for src/kicker_defense.py -- K/DEF descriptive season stats.

No leakage tests here the way Family 1-7 have them -- these are
season-to-date summaries for a human reader (same reasoning as
src/export.py's build_xfp_summary), not model features, so there's
nothing point-in-time-sensitive to prove.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.kicker_defense import (  # noqa: E402
    DEFENSE_STATS_OUTPUT_COLUMNS,
    KICKER_STATS_OUTPUT_COLUMNS,
    build_defense_season_stats,
    build_kicker_season_stats,
)


def _kicker_week(player_id="K1", name="Test Kicker", team="KC", week=1, **overrides):
    row = {
        "player_id": player_id, "player_display_name": name, "team": team,
        "position": "K", "season": 2024, "week": week, "season_type": "REG",
        "fg_made": 0, "fg_att": 0, "pat_made": 0, "pat_att": 0,
        "fg_made_0_19": 0, "fg_made_20_29": 0, "fg_made_30_39": 0,
        "fg_made_40_49": 0, "fg_made_50_59": 0, "fg_made_60_": 0,
        "fg_missed_0_19": 0, "fg_missed_20_29": 0, "fg_missed_30_39": 0,
        "fg_missed_40_49": 0, "fg_missed_50_59": 0, "fg_missed_60_": 0,
    }
    row.update(overrides)
    return row


def test_build_kicker_season_stats_computes_pat_rate_and_band_rates():
    weekly = pd.DataFrame([
        _kicker_week(week=1, fg_made=2, fg_att=2, pat_made=3, pat_att=3,
                     fg_made_20_29=1, fg_made_40_49=1),
        _kicker_week(week=2, fg_made=1, fg_att=2, pat_made=2, pat_att=2,
                     fg_made_50_59=1, fg_missed_40_49=1),
    ])
    out = build_kicker_season_stats(weekly, 2024)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["games_played"] == 2
    assert row["fg_made"] == 3
    assert row["fg_att"] == 4
    assert row["pat_rate"] == pytest.approx(1.0)  # 5/5
    assert row["attempts_per_game"] == pytest.approx(2.0)  # 4/2
    assert row["fg_rate_under_30"] == pytest.approx(1.0)  # 1 made / 1 att (20-29)
    assert row["fg_rate_40_49"] == pytest.approx(0.5)  # 1 made, 1 missed
    assert row["fg_rate_50_plus"] == pytest.approx(1.0)  # 1 made, 0 missed


def test_build_kicker_season_stats_null_rate_with_zero_attempts_in_band():
    weekly = pd.DataFrame([_kicker_week(week=1, fg_made=1, fg_att=1, fg_made_0_19=1)])
    out = build_kicker_season_stats(weekly, 2024)
    row = out.iloc[0]
    assert row["fg_rate_under_30"] == pytest.approx(1.0)
    assert pd.isna(row["fg_rate_30_39"])  # zero attempts in this band -- null, not 0
    assert pd.isna(row["fg_rate_40_49"])
    assert pd.isna(row["fg_rate_50_plus"])


def test_build_kicker_season_stats_empty_for_season_with_no_k_rows():
    weekly = pd.DataFrame([_kicker_week(week=1)])
    out = build_kicker_season_stats(weekly, 2099)
    assert out.empty
    assert list(out.columns) == ["player_id", "player_display_name", "team"] + KICKER_STATS_OUTPUT_COLUMNS


def _pbp_play(defteam, posteam, play_type, season=2024, week=1, **overrides):
    row = {
        "season_type": "REG", "season": season, "week": week,
        "defteam": defteam, "posteam": posteam, "play_type": play_type,
        "sack": 0, "interception": 0, "passing_yards": 0, "rushing_yards": 0,
    }
    row.update(overrides)
    return row


def test_build_defense_season_stats_sacks_interceptions_and_yards_allowed():
    pbp = pd.DataFrame([
        _pbp_play("KC", "DEN", "pass", sack=1, passing_yards=0),
        _pbp_play("KC", "DEN", "pass", interception=1, passing_yards=0),
        _pbp_play("KC", "DEN", "pass", passing_yards=20),
        _pbp_play("KC", "DEN", "run", rushing_yards=5),
        _pbp_play("DEN", "KC", "run", rushing_yards=3),
    ])
    schedule = pd.DataFrame([
        {"game_type": "REG", "season": 2024, "week": 1,
         "home_team": "KC", "away_team": "DEN", "home_score": 27, "away_score": 13},
    ])
    out = build_defense_season_stats(pbp, schedule, 2024)

    kc = out[out["team"] == "KC"].iloc[0]
    assert kc["sacks"] == 1
    assert kc["interceptions"] == 1
    assert kc["pass_yards_allowed_per_game"] == pytest.approx(20.0)
    assert kc["rush_yards_allowed_per_game"] == pytest.approx(5.0)
    assert kc["points_allowed_per_game"] == pytest.approx(13.0)  # DEN's (away) score
    assert kc["games_played"] == 1

    den = out[out["team"] == "DEN"].iloc[0]
    assert den["points_allowed_per_game"] == pytest.approx(27.0)  # KC's (home) score
    assert den["rush_yards_allowed_per_game"] == pytest.approx(3.0)


def test_build_defense_season_stats_zero_fills_teams_with_no_real_pbp_events():
    """A team with a real completed game but zero sacks/INTs that game
    must show 0, not null or a dropped row -- games_played/points_allowed
    come from the schedule regardless of how the pbp side merges."""
    pbp = pd.DataFrame([_pbp_play("KC", "DEN", "pass", passing_yards=10)])
    schedule = pd.DataFrame([
        {"game_type": "REG", "season": 2024, "week": 1,
         "home_team": "KC", "away_team": "DEN", "home_score": 27, "away_score": 13},
    ])
    out = build_defense_season_stats(pbp, schedule, 2024)
    den = out[out["team"] == "DEN"].iloc[0]
    assert den["sacks"] == 0
    assert den["interceptions"] == 0
    assert den["games_played"] == 1
    assert list(out.columns) == ["team"] + DEFENSE_STATS_OUTPUT_COLUMNS


def test_build_defense_season_stats_tolerates_completely_empty_pbp():
    """weekly_update.py's own get_pbp caller passes a bare pd.DataFrame()
    (zero rows, zero columns) when a season hasn't published pbp yet --
    the real pre-draft/week-1 condition, not a bug. This must produce a
    correctly-shaped result (schedule-only, or genuinely empty if no
    games are complete yet either), not a KeyError on missing pbp
    columns that were never going to be used anyway."""
    pbp = pd.DataFrame()
    schedule = pd.DataFrame([
        {"game_type": "REG", "season": 2024, "week": 1,
         "home_team": "KC", "away_team": "DEN", "home_score": 27, "away_score": 13},
    ])
    out = build_defense_season_stats(pbp, schedule, 2024)
    assert list(out.columns) == ["team"] + DEFENSE_STATS_OUTPUT_COLUMNS
    kc = out[out["team"] == "KC"].iloc[0]
    assert kc["sacks"] == 0
    assert kc["interceptions"] == 0
    assert kc["pass_yards_allowed_per_game"] == 0
    assert kc["rush_yards_allowed_per_game"] == 0
    assert kc["points_allowed_per_game"] == pytest.approx(13.0)


def test_build_defense_season_stats_empty_pbp_and_no_completed_games():
    """The actual pre-draft condition: neither pbp nor any completed game
    exists yet. Must return an empty, correctly-shaped frame -- same
    "unplayed week simply isn't a row yet" pattern as
    build_kicker_season_stats' own empty branch -- not a crash."""
    pbp = pd.DataFrame()
    schedule = pd.DataFrame([
        {"game_type": "REG", "season": 2024, "week": 1,
         "home_team": "KC", "away_team": "DEN", "home_score": None, "away_score": None},
    ])
    out = build_defense_season_stats(pbp, schedule, 2024)
    assert out.empty
    assert list(out.columns) == ["team"] + DEFENSE_STATS_OUTPUT_COLUMNS
