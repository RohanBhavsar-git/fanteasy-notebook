"""
Tests for src/artifacts.py -- Phase 8 model artifact persistence.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import load_model_artifact, save_model_artifact  # noqa: E402


def _sample_artifact_kwargs():
    return dict(
        models={"WR": {"point": "fake-model", "quantiles": {0.10: "fake-q10", 0.90: "fake-q90"}}},
        feature_columns=["feat_a", "feat_b"],
        cqr_widen_by_10_90={"WR": 1.2},
        performance={"WR": {"model_mae": 4.0, "sleeper_mae": 3.8, "s2d_mae": 4.2}},
        seasons_trained=[2024, 2025],
        history_seed=pd.DataFrame({"player_id": ["p1"], "season": [2025]}),
        history_seed_seasons=[2025],
        model_version="abc1234",
    )


def test_save_and_load_model_artifact_round_trips(tmp_path):
    path = tmp_path / "model.joblib"
    written_path = save_model_artifact(path=path, **_sample_artifact_kwargs())
    assert written_path == path
    assert path.exists()

    loaded = load_model_artifact(path=path)
    assert loaded["model_version"] == "abc1234"
    assert loaded["seasons_trained"] == [2024, 2025]
    assert loaded["feature_columns"] == ["feat_a", "feat_b"]
    assert loaded["cqr_widen_by_10_90"] == {"WR": 1.2}
    assert loaded["models"]["WR"]["point"] == "fake-model"
    pd.testing.assert_frame_equal(loaded["history_seed"], _sample_artifact_kwargs()["history_seed"])
    assert "trained_at" in loaded  # stamped by save_model_artifact, not passed in


def test_load_model_artifact_raises_clearly_when_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError, match="retrain"):
        load_model_artifact(path=missing_path)


def test_load_model_artifact_raises_on_malformed_artifact(tmp_path):
    import joblib

    path = tmp_path / "malformed.joblib"
    joblib.dump({"models": {}}, path)  # missing every other required key

    with pytest.raises(KeyError):
        load_model_artifact(path=path)
