import pandas as pd
import pytest

from engine.decision_frame import DecisionFrame
from engine.entry_consistency import validate_entry_consistency
from engine.entry_eligibility import get_eligible_entry_models
from engine.entry_features import ENTRY_FEATURE_NAMES
from engine.entry_models import ENTRY_MODELS
from engine.ml_brain_dataset import validate_entry_dataset
from engine.tactical_replay import replay_tactical_brain


def _policy_row(
    entry_id: str,
    label: str,
    mp="DISTRIBUTION",
    sp="PROFILE_1C",
    bias="UP",
    prob=0.5,
    exp=0.1,
    sample=10,
):
    return {
        "entry_model_id": entry_id,
        "market_profile_state": mp,
        "session_profile": sp,
        "liquidity_bias_side": bias,
        "label": label,
        "prob_good": prob,
        "expected_reward": exp,
        "sample_size": sample,
    }


def test_policy_override_detection():
    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1A",
        liquidity_frame={
            "bias": "UP",
            "sweep_state": "POST_SWEEP",
            "distance_bucket": "NEAR",
        },
        entry_signals_present={"sweep": True, "displacement": True},
        market_profile_evidence={"displacement_score": 0.6},
        condition_vector={"vol": "NORMAL", "trend": "UP"},
    )
    frame.eligible_entry_models = get_eligible_entry_models(frame)

    policy_df = pd.DataFrame(
        [_policy_row("ENTRY_SWEEP_DISPLACEMENT_REVERSAL", label="DISABLED")]
    )

    report = validate_entry_consistency(frame, policy_df)
    assert "ENTRY_SWEEP_DISPLACEMENT_REVERSAL" in report["policy_override"]
    assert report["ok"] is False


def test_eligibility_mismatch_detection():
    frame = DecisionFrame(
        market_profile_state="ACCUMULATION",
        session_profile="PROFILE_1C",
        liquidity_frame={
            "bias": "UP",
            "sweep_state": "NO_SWEEP",
            "distance_bucket": "NEAR",
        },
        entry_signals_present={"sweep": False, "displacement": False},
    )
    frame.eligible_entry_models = []

    policy_df = pd.DataFrame(
        [_policy_row("ENTRY_FVG_RESPECT_CONTINUATION", label="PREFERRED")]
    )

    report = validate_entry_consistency(frame, policy_df)
    assert "ENTRY_FVG_RESPECT_CONTINUATION" in report["eligibility_mismatch"]
    assert report["ok"] is False


def test_signal_mismatch_detection():
    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={"bias": "UP", "sweep_state": "ANY", "distance_bucket": "NEAR"},
        entry_signals_present={"sweep": False, "fvg": False},
    )
    frame.eligible_entry_models = get_eligible_entry_models(frame)

    policy_df = pd.DataFrame(
        [_policy_row("ENTRY_FVG_RESPECT_CONTINUATION", label="PREFERRED")]
    )

    report = validate_entry_consistency(frame, policy_df)
    assert "ENTRY_FVG_RESPECT_CONTINUATION" in report["signal_mismatch"]
    assert report["ok"] is False


def test_tactical_replay_reconstructs_decision_frame():
    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={"bias": "UP", "distance_bucket": "NEAR", "sweep_state": "ANY"},
        entry_signals_present={"fvg": True},
        condition_vector={"vol": "NORMAL", "trend": "UP"},
    )
    frame.eligible_entry_models = ["ENTRY_FVG_RESPECT_CONTINUATION"]
    frame.entry_brain_labels = {"ENTRY_FVG_RESPECT_CONTINUATION": "PREFERRED"}
    frame.entry_brain_scores = {
        "ENTRY_FVG_RESPECT_CONTINUATION": {"prob_good": 0.8, "expected_reward": 1.0}
    }

    decision_logs = pd.DataFrame(
        [
            {
                "entry_model_id": "ENTRY_FVG_RESPECT_CONTINUATION",
                "eligible_entry_models": frame.eligible_entry_models,
                "entry_brain_labels": frame.entry_brain_labels,
                "entry_brain_scores": frame.entry_brain_scores,
                "decision_frame": frame.to_dict(),
            }
        ]
    )

    policy_df = pd.DataFrame(
        [
            _policy_row(
                "ENTRY_FVG_RESPECT_CONTINUATION", label="PREFERRED", prob=0.8, exp=1.0
            )
        ]
    )

    replay_df = replay_tactical_brain(decision_logs, policy_df)
    assert not replay_df.empty
    assert bool(replay_df.loc[0, "eligibility_match"]) is True
    assert bool(replay_df.loc[0, "policy_label_match"]) is True


def test_dataset_validation_detects_missing_features():
    df = pd.DataFrame(
        [
            {
                "entry_model_id": next(iter(ENTRY_MODELS.keys())),
                "market_profile_state": "DISTRIBUTION",
                "session_profile": "PROFILE_1C",
                "liquidity_bias_side": "UP",
                "entry_outcome": 1.0,
                "chosen_entry_model_id": next(iter(ENTRY_MODELS.keys())),
            }
        ]
    )
    result = validate_entry_dataset(df)
    assert result["missing_features"]
    assert result["ok"] is False


def test_dataset_validation_passes_on_clean_dataset():
    entry_id = next(iter(ENTRY_MODELS.keys()))
    row = {
        "entry_model_id": entry_id,
        "market_profile_state": "DISTRIBUTION",
        "session_profile": "PROFILE_1C",
        "liquidity_bias_side": "UP",
        "entry_outcome": 1.0,
        "chosen_entry_model_id": entry_id,
        "entry_brain_label": "PREFERRED",
        "entry_brain_prob_good": 0.7,
        "entry_brain_expected_R": 0.5,
    }
    for fname in ENTRY_FEATURE_NAMES:
        row[f"entry_feature_{fname}"] = 0

    df = pd.DataFrame([row])
    result = validate_entry_dataset(df)
    assert result["ok"] is True
    assert not result["missing_features"]
    assert not result["invalid_entry_ids"]
    assert result["rows_with_missing_outcomes"] == 0
