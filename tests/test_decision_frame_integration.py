import json
from pathlib import Path

from engine.decision_frame import DecisionFrame
from engine.decision_logger import DecisionLogger
from engine.structure_brain import (
    LiquidityFrame,
    MarketProfileFrame,
    SessionProfileFrame,
)


def test_decision_frame_includes_structure_brain_outputs():
    mp = MarketProfileFrame(
        state="MANIPULATION", confidence=0.7, evidence={"coarse": "MAN"}
    )
    sp = SessionProfileFrame(
        profile="PROFILE_1B", confidence=0.6, evidence={"early_volatility": 0.4}
    )
    liq = LiquidityFrame(
        primary_target="PDH",
        target_side="UP",
        distances={"PDH": 5.0},
        swept={"PDH": False},
        bias="UP",
    )

    frame = DecisionFrame.from_frames(
        timestamp_utc="2026-02-04T00:00:00Z",
        symbol="ES",
        session_context={"session": "RTH"},
        condition_vector={"session": "RTH", "vol": "HIGH"},
        market_profile=mp,
        session_profile=sp,
        liquidity=liq,
    )

    data = frame.to_dict()
    assert data["market_profile_state"] == "MANIPULATION"
    assert data["session_profile"] == "PROFILE_1B"
    assert data["liquidity_frame"]["primary_target"] == "PDH"
    assert data["liquidity_frame"]["bias"] == "UP"


def test_decision_frame_logged_in_decision_log(tmp_path):
    log_path = Path(tmp_path) / "decisions.jsonl"
    schema_path = Path("schemas/decision_log.schema.json")
    logger = DecisionLogger(log_path=log_path, schema_path=schema_path)

    mp = MarketProfileFrame(
        state="ACCUMULATION", confidence=0.5, evidence={"sweep": False}
    )
    liq = LiquidityFrame(
        primary_target="PDL",
        target_side="DOWN",
        distances={"PDL": 3.0},
        swept={"PDL": True},
        bias="DOWN",
    )
    decision_frame = DecisionFrame.from_frames(
        timestamp_utc="2026-02-04T00:00:00Z",
        symbol="ES",
        session_context={"session": "RTH"},
        condition_vector={"session": "RTH"},
        market_profile=mp,
        session_profile=None,
        liquidity=liq,
    )

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
        "condition_vector": {"session": "RTH"},
        "decision_frame": decision_frame,
    }

    logger.log_decision(entry)

    logged = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert "decision_frame" in logged
    assert logged["decision_frame"]["market_profile_state"] == "ACCUMULATION"
    assert logged["decision_frame"]["liquidity_frame"]["primary_target"] == "PDL"
