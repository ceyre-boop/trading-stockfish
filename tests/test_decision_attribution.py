import json
from pathlib import Path

from engine.condition_encoder import ConditionVector, encode_conditions
from engine.decision_logger import DecisionLogger
from engine.types import MarketState


def test_encode_conditions_deterministic():
    state = MarketState(
        session="RTH_OPEN",
        macro_regime="RISK_ON",
        volatility_regime="HIGH",
        trend_direction="UP",
        liquidity_regime="THIN",
        raw={"time_of_day_bucket": "OPEN"},
    )

    cv = encode_conditions(state)

    assert cv.session == "RTH_OPEN"
    assert cv.macro == "RISK_ON"
    assert cv.vol == "HIGH"
    assert cv.trend == "UP"
    assert cv.liquidity == "THIN"
    assert cv.tod == "OPEN"


def test_decision_logger_serializes_condition_vector(tmp_path: Path):
    log_path = tmp_path / "decision_log.jsonl"
    logger = DecisionLogger(log_path=log_path)

    cv = ConditionVector(
        session="RTH",
        macro="NEUTRAL",
        vol="NORMAL",
        trend="FLAT",
        liquidity="NORMAL",
        tod="MIDDAY",
    )

    entry = {
        "decision_id": "dec1",
        "condition_vector": cv,
        "strategy_id": "strat1",
        "entry_model_id": "entryA",
        "exit_model_id": None,
        "ml_policy_version": None,
    }

    logger.log_decision(entry)

    line = log_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)

    assert parsed["condition_vector"] == {
        "session": "RTH",
        "macro": "NEUTRAL",
        "vol": "NORMAL",
        "trend": "FLAT",
        "liquidity": "NORMAL",
        "tod": "MIDDAY",
    }
    assert parsed["strategy_id"] == "strat1"
    assert parsed["entry_model_id"] == "entryA"
    assert parsed["exit_model_id"] is None
    assert parsed["ml_policy_version"] is None
