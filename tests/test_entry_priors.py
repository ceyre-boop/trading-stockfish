import json
from pathlib import Path

import pandas as pd

from engine.decision_frame import DecisionFrame
from engine.entry_priors import build_entry_priors, write_entry_priors
from engine.entry_selector_dataset import build_entry_selector_dataset
from engine.entry_selector_model import train_entry_selector_model

ENTRY_A = "ENTRY_SWEEP_DISPLACEMENT_REVERSAL"
ENTRY_B = "ENTRY_FVG_RESPECT_CONTINUATION"


TIPS = {
    ENTRY_A: {
        "entry_model_id": ENTRY_A,
        "avg_prob_select": 0.6,
        "avg_expected_R": 1.8,
        "confidence_score": 0.7,
        "best_market_profile_state": "MANIPULATION",
        "worst_market_profile_state": "ACCUMULATION",
        "best_session_profile": "PROFILE_1A",
        "worst_session_profile": "PROFILE_1C",
        "risk_expected_R": 2.5,
        "risk_aggressiveness": "AGGRESSIVE",
    },
    ENTRY_B: {
        "entry_model_id": ENTRY_B,
        "avg_prob_select": 0.4,
        "avg_expected_R": 1.2,
        "confidence_score": 0.4,
        "best_market_profile_state": "DISTRIBUTION",
        "worst_market_profile_state": "ACCUMULATION",
        "best_session_profile": "PROFILE_1B",
        "worst_session_profile": "PROFILE_1A",
        "risk_expected_R": 1.5,
        "risk_aggressiveness": "NEUTRAL",
    },
}


def _make_log(ts: str, entry_id: str, eligible: list[str], outcome: float | None):
    frame = DecisionFrame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1A",
        liquidity_frame={"bias": "UP", "distance_bucket": "NEAR", "sweep_state": "ANY"},
        vol_regime="NORMAL",
        trend_regime="UP",
        entry_signals_present={"fvg": True},
        timestamp_utc=ts,
        symbol="EURUSD",
    )
    return {
        "timestamp_utc": ts,
        "symbol": "EURUSD",
        "entry_model_id": entry_id,
        "eligible_entry_models": eligible,
        "entry_outcome": outcome,
        "decision_frame": frame.to_dict(),
    }


def test_priors_built_from_tips():
    priors = build_entry_priors(TIPS)
    assert ENTRY_A in priors and ENTRY_B in priors
    for prior in priors.values():
        for key in [
            "base_prob_prior",
            "base_expected_R_prior",
            "confidence",
            "preferred_market_profile_states",
            "risk_expected_R",
        ]:
            assert key in prior


def test_structural_preferences_correct():
    priors = build_entry_priors(TIPS)
    prior_a = priors[ENTRY_A]
    assert prior_a["preferred_market_profile_states"] == ["MANIPULATION"]
    assert prior_a["discouraged_market_profile_states"] == ["ACCUMULATION"]
    assert prior_a["preferred_session_profiles"] == ["PROFILE_1A"]
    assert prior_a["discouraged_session_profiles"] == ["PROFILE_1C"]


def test_prior_artifact_sorted(tmp_path):
    priors = build_entry_priors(TIPS)
    target = tmp_path / "entry_priors.json"
    write_entry_priors(priors, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    ids = [p["entry_model_id"] for p in payload]
    assert ids == sorted(ids)


def test_selector_training_with_priors():
    logs = pd.DataFrame(
        [
            _make_log(
                "2024-01-01T00:00:00Z",
                ENTRY_A,
                [ENTRY_A, ENTRY_B],
                1.0,
            ),
            _make_log(
                "2024-01-01T00:05:00Z",
                ENTRY_B,
                [ENTRY_A, ENTRY_B],
                -0.5,
            ),
        ]
    )
    ds = build_entry_selector_dataset(logs)
    priors = build_entry_priors(TIPS)
    weights = (
        ds["entry_model_id"]
        .map(
            {
                ENTRY_A: 1.0
                + priors[ENTRY_A]["confidence"]
                + priors[ENTRY_A]["base_expected_R_prior"],
                ENTRY_B: 1.0
                + priors[ENTRY_B]["confidence"]
                + priors[ENTRY_B]["base_expected_R_prior"],
            }
        )
        .fillna(1.0)
    )

    artifacts = train_entry_selector_model(
        ds,
        sample_weight=weights.to_numpy(),
        use_priors=True,
        random_state=13,
    )

    assert artifacts.metadata.get("use_priors") is True
    assert artifacts.metadata.get("sample_weight_sum") == float(weights.sum())


def test_determinism():
    priors1 = build_entry_priors(TIPS)
    priors2 = build_entry_priors(TIPS)
    assert priors1 == priors2
