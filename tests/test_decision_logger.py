import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.decision_logger import DecisionLogger


def _fixed_datetime(year: int, month: int, day: int, hour: int, minute: int = 0):
    # Helper to construct naive UTC timestamps for deterministic logging in tests
    return datetime(year, month, day, hour, minute)


def _minimal_entry():
    return {
        "run_id": "run_test",
        "decision_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "symbol": "ES",
        "timeframe": "M5",
        "session_regime": "RTH_OPEN",
        "macro_regimes": ["RISK_ON"],
        "feature_vector": {},
        "effective_weights": {},
        "policy_components": {
            "base_weights": {},
            "trust": {},
            "regime_multipliers": {},
        },
        "evaluation_score": 0.25,
        "action": "LONG",
        "provenance": {
            "policy_version": "0.0.0",
            "feature_spec_version": "1.0.0",
            "feature_audit_version": "1.0.0",
            "engine_version": "test",
        },
    }


def test_decision_logger_writes_valid_jsonl(tmp_path: Path):
    log_path = tmp_path / "decision_log.jsonl"
    logger = DecisionLogger(log_path=log_path)

    entry = _minimal_entry()
    logger.log_decision(entry)

    assert log_path.exists(), "Log file should be created"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "Expected exactly one log line"

    parsed = json.loads(lines[0])
    assert isinstance(parsed, dict), "Logged line must parse as JSON object"


def test_decision_logger_validates_against_schema_if_available(tmp_path: Path):
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        pytest.skip("jsonschema not installed; skipping schema validation test")

    schema_path = Path("schemas/decision_log.schema.json")
    if not schema_path.exists():
        pytest.skip("decision_log.schema.json not present")

    log_path = tmp_path / "decision_log.jsonl"
    logger = DecisionLogger(log_path=log_path, schema_path=schema_path)

    logger.log_decision(_minimal_entry())

    with pytest.raises(ValueError):
        bad_entry = _minimal_entry()
        bad_entry.pop("action")
        logger.log_decision(bad_entry)


def test_runner_emits_decision_log_entries(tmp_path: Path, monkeypatch):
    from engine import evaluator as evaluator_mod

    log_path = tmp_path / "decision_log.jsonl"
    monkeypatch.setattr(
        evaluator_mod,
        "_DECISION_LOGGER",
        DecisionLogger(log_path=log_path),
    )

    class DummyFactor:
        def __init__(self, name: str, score: float, weight: float):
            self.factor_name = name
            self.score = score
            self.weight = weight
            self.explanation = ""

    class DummyEvalResult:
        def __init__(self):
            self.eval_score = 0.4
            self.confidence = 0.8
            self.scoring_factors = [DummyFactor("trend", 0.4, 1.0)]
            self.timestamp = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )

    class DummyCausalEvaluator:
        def evaluate(self, market_state):
            return DummyEvalResult()

    class DummyExec:
        def __init__(self, position_size: float):
            self.position_size = position_size

    class DummyMarketState:
        def __init__(self):
            self.session = "RTH_OPEN"
            self.macro_regime = "RISK_ON"
            self.execution = DummyExec(position_size=1.25)

    state = {
        "run_id": "run_pipeline",
        "symbol": "ES",
        "timeframe": "M5",
        "features": {},
    }

    causal_eval = DummyCausalEvaluator()
    market_state = DummyMarketState()

    for _ in range(3):
        evaluator_mod.evaluate_with_causal(
            state=state,
            causal_evaluator=causal_eval,
            market_state=market_state,
            policy=None,
        )

    assert log_path.exists(), "Decision log should be created by evaluator"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0, "Log should have entries"

    for line in lines:
        parsed = json.loads(line)
        for field in [
            "run_id",
            "decision_id",
            "timestamp_utc",
            "action",
            "evaluation_score",
        ]:
            assert field in parsed, f"Missing required field: {field}"


def test_decision_log_includes_session_regime(tmp_path: Path, monkeypatch):
    from engine import evaluator as evaluator_mod
    from engine.session_regimes import classify_session

    log_path = tmp_path / "decision_log.jsonl"

    fixed_dt = _fixed_datetime(2024, 1, 2, 12, 30)
    fixed_regime = classify_session(fixed_dt.replace(tzinfo=timezone.utc))

    class FixedDateTime(datetime):
        @classmethod
        def utcnow(cls):
            return fixed_dt

    monkeypatch.setattr(evaluator_mod, "datetime", FixedDateTime)
    monkeypatch.setattr(
        evaluator_mod,
        "_DECISION_LOGGER",
        DecisionLogger(log_path=log_path),
    )

    class DummyFactor:
        def __init__(self, name: str, score: float, weight: float):
            self.factor_name = name
            self.score = score
            self.weight = weight
            self.explanation = ""

    class DummyEvalResult:
        def __init__(self):
            self.eval_score = 0.4
            self.confidence = 0.8
            self.scoring_factors = [DummyFactor("trend", 0.4, 1.0)]
            self.timestamp = fixed_dt.isoformat() + "Z"

    class DummyCausalEvaluator:
        def evaluate(self, market_state):
            return DummyEvalResult()

    class DummyExec:
        def __init__(self, position_size: float):
            self.position_size = position_size

    class DummyMarketState:
        def __init__(self):
            self.session = fixed_regime
            self.macro_regime = "RISK_ON"
            self.execution = DummyExec(position_size=1.0)

    state = {
        "run_id": "run_pipeline",
        "symbol": "ES",
        "timeframe": "M5",
        "features": {},
    }

    causal_eval = DummyCausalEvaluator()
    market_state = DummyMarketState()

    evaluator_mod.evaluate_with_causal(
        state=state,
        causal_evaluator=causal_eval,
        market_state=market_state,
        policy=None,
    )

    assert log_path.exists(), "Decision log should be created"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert "session_regime" in parsed
    assert parsed["session_regime"] == fixed_regime
