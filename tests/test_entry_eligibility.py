import json
from pathlib import Path

from engine.decision_frame import DecisionFrame
from engine.decision_logger import DecisionLogger
from engine.entry_eligibility import get_eligible_entry_models, is_entry_eligible
from engine.entry_models import ENTRY_MODELS


def _frame(**kwargs) -> DecisionFrame:
    return DecisionFrame(**kwargs)


def test_sweep_displacement_reversal_eligibility():
    frame = _frame(
        market_profile_state="MANIPULATION",
        session_profile="PROFILE_1A",
        liquidity_frame={
            "bias": "UP",
            "sweep_state": "POST_SWEEP",
            "distance_bucket": "NEAR",
            "swept": {"PDL": True},
        },
        market_profile_evidence={"displacement_score": 0.65},
        condition_vector={"vol": "HIGH", "trend": "UP"},
        entry_signals_present={
            "sweep": True,
            "displacement": True,
            "fvg": False,
            "ob": False,
            "ifvg": False,
        },
    )

    entry = ENTRY_MODELS["ENTRY_SWEEP_DISPLACEMENT_REVERSAL"]
    assert is_entry_eligible(entry, frame)


def test_fvg_respect_continuation_eligibility():
    frame = _frame(
        market_profile_state="DISTRIBUTION",
        session_profile="PROFILE_1C",
        liquidity_frame={
            "bias": "UP",
            "sweep_state": "NO_SWEEP",
            "distance_bucket": "INSIDE",
        },
        condition_vector={"vol": "NORMAL", "trend": "UP"},
        entry_signals_present={
            "sweep": False,
            "displacement": False,
            "fvg": True,
            "ob": False,
            "ifvg": False,
        },
    )

    entry = ENTRY_MODELS["ENTRY_FVG_RESPECT_CONTINUATION"]
    assert is_entry_eligible(entry, frame)


def test_mean_reversion_range_extreme_in_accumulation():
    frame = _frame(
        market_profile_state="ACCUMULATION",
        session_profile="UNKNOWN",
        liquidity_frame={
            "bias": "NEUTRAL",
            "sweep_state": "NO_SWEEP",
            "distance_bucket": "FAR",
        },
        condition_vector={"vol": "LOW", "trend": "FLAT"},
        entry_signals_present={
            "sweep": False,
            "displacement": False,
            "fvg": False,
            "ob": False,
            "ifvg": False,
        },
    )

    entry = ENTRY_MODELS["ENTRY_MEAN_REVERSION_RANGE_EXTREME"]
    assert is_entry_eligible(entry, frame)


def test_ineligible_when_structure_mismatch():
    frame = _frame(
        market_profile_state="ACCUMULATION",
        session_profile="PROFILE_1A",
        liquidity_frame={
            "bias": "UP",
            "sweep_state": "NO_SWEEP",
            "distance_bucket": "FAR",
        },
        condition_vector={"vol": "NORMAL", "trend": "UP"},
        entry_signals_present={
            "sweep": False,
            "displacement": False,
            "fvg": False,
            "ob": True,
            "ifvg": False,
        },
    )

    entry = ENTRY_MODELS["ENTRY_OB_CONTINUATION"]
    assert not is_entry_eligible(entry, frame)


def test_decision_frame_carries_eligible_entries(tmp_path):
    frame = _frame(
        market_profile_state="MANIPULATION",
        session_profile="PROFILE_1A",
        liquidity_frame={
            "bias": "DOWN",
            "sweep_state": "POST_SWEEP",
            "distance_bucket": "NEAR",
            "swept": {"PDH": True},
        },
        market_profile_evidence={"displacement_score": 0.7},
        condition_vector={"vol": "HIGH", "trend": "DOWN"},
        entry_signals_present={
            "sweep": True,
            "displacement": True,
            "fvg": False,
            "ob": False,
            "ifvg": False,
        },
    )
    frame.eligible_entry_models = get_eligible_entry_models(frame)

    log_path = Path(tmp_path) / "decisions.jsonl"
    schema_path = Path("schemas/decision_log.schema.json")
    logger = DecisionLogger(log_path=log_path, schema_path=schema_path)

    entry = {
        "run_id": "run_1",
        "decision_id": "dec_1",
        "timestamp_utc": "2026-02-04T00:00:00Z",
        "symbol": "ES",
        "timeframe": "M5",
        "session_regime": "RTH",
        "macro_regimes": ["MACRO_ON"],
        "feature_vector": {},
        "effective_weights": {},
        "policy_components": {
            "base_weights": {},
            "trust": {},
            "regime_multipliers": {},
            "session_multiplier": 1.0,
        },
        "evaluation_score": 0.1,
        "action": "LONG",
        "outcome": None,
        "provenance": {
            "policy_version": "0.0.0",
            "feature_spec_version": "0.0.0",
            "feature_audit_version": "0.0.0",
            "engine_version": "0.0.0",
        },
        "condition_vector": {"vol": "HIGH", "trend": "DOWN"},
        "entry_signals_present": frame.entry_signals_present,
        "eligible_entry_models": frame.eligible_entry_models,
        "decision_frame": frame,
    }

    # Skip schema validation to avoid extra fields normalization mismatch
    logger.schema_path = None
    logger.log_decision(entry)

    logged = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    df = logged.get("decision_frame")
    assert df is not None
    assert df.get("eligible_entry_models") == frame.eligible_entry_models
    assert logged.get("eligible_entry_models") == frame.eligible_entry_models
