import pandas as pd
import pytest

from engine.decision_frame import DecisionFrame
from engine.entry_features import ENTRY_FEATURE_NAMES, extract_entry_features
from engine.ml_brain import BrainScore
from engine.ml_brain_dataset import build_brain_dataset
from engine.ml_brain_entry import score_entry_models


def test_extract_entry_features_minimal():
    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={
            "bias": "UP",
            "distances": {"primary": 1.2},
            "primary_target": "primary",
            "sweep_state": "POST_SWEEP",
        },
        vol_regime="NORMAL",
        trend_regime="UP",
        entry_signals_present={"sweep": True, "fvg": True, "ob": False, "ifvg": False},
        market_profile_evidence={"displacement_score": 0.6},
        risk_per_trade=0.5,
        position_size=2.0,
    )

    features = extract_entry_features("ENTRY_FVG_RESPECT_CONTINUATION", frame)

    assert set(features.keys()) == set(ENTRY_FEATURE_NAMES)
    assert features["market_profile_state"] == "DISTRIBUTION"
    assert features["liquidity_bias_side"] == "UP"
    assert features["liquidity_nearest_target_distance"] == 1.2
    assert features["fvg_flag"] is True
    assert features["displacement_score"] == 0.6
    assert features["risk_per_trade"] == 0.5
    assert features["risk_expected_R"] == 1.5
    assert features["risk_mae_bucket_encoded"] == 1
    assert features["risk_mfe_bucket_encoded"] == 1
    assert features["risk_time_horizon_encoded"] == 1
    assert features["risk_aggressiveness_encoded"] == 1


def test_extract_entry_features_missing_values():
    frame = DecisionFrame()

    features = extract_entry_features("ENTRY_FVG_RESPECT_CONTINUATION", frame)

    for name in ENTRY_FEATURE_NAMES:
        assert name in features
    assert features["liquidity_nearest_target_distance"] is None
    assert features["sweep_flag"] is None
    assert features["position_size"] is None
    assert features["risk_expected_R"] == 1.5
    assert features["risk_mae_bucket_encoded"] == 1
    assert features["risk_mfe_bucket_encoded"] == 1
    assert features["risk_time_horizon_encoded"] == 1
    assert features["risk_aggressiveness_encoded"] == 1


def test_score_entry_models_calls_brain(monkeypatch):
    frame = DecisionFrame()
    artifacts = object()

    captured = {}

    def fake_score_combo(strategy_id, entry_model_id, condition_vector, artifacts):
        captured["strategy_id"] = strategy_id
        captured["entry_model_id"] = entry_model_id
        captured["condition_vector"] = condition_vector
        captured["artifacts"] = artifacts
        return BrainScore(
            prob_good=0.9, expected_reward=1.1, sample_size=3, flags={"ok": True}
        )

    monkeypatch.setattr("engine.ml_brain_entry.score_combo", fake_score_combo)

    results = score_entry_models(frame, ["ENTRY_FVG_RESPECT_CONTINUATION"], artifacts)

    assert "ENTRY_FVG_RESPECT_CONTINUATION" in results
    assert isinstance(results["ENTRY_FVG_RESPECT_CONTINUATION"], BrainScore)
    assert captured["strategy_id"] == "ENTRY_FVG_RESPECT_CONTINUATION"
    assert captured["condition_vector"] is not None


def test_dataset_builder_includes_entry_features():
    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={
            "bias": "UP",
            "distances": {"primary": 2.0},
            "primary_target": "primary",
            "sweep_state": "POST_SWEEP",
        },
        entry_signals_present={"fvg": True},
    ).to_dict()

    decisions = pd.DataFrame(
        [
            {
                "strategy_id": "STRAT1",
                "entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION",
                "outcome": 1.5,
                "condition_vector": {
                    "session": "LDN",
                    "macro": "NEUTRAL",
                    "vol": "NORMAL",
                    "trend": "UP",
                    "liquidity": "BALANCED",
                    "tod": "AM",
                },
                "decision_frame": frame,
            }
        ]
    )

    dataset = build_brain_dataset(decisions)

    feature_cols = [col for col in dataset.columns if col.startswith("entry_feature_")]
    assert not dataset.empty
    assert feature_cols
    assert "entry_feature_market_profile_state" in dataset.columns
    assert dataset.loc[0, "entry_model_id"] == "ENTRY_FVG_RESPECT_CONTINUATION"
