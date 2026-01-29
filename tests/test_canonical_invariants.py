import copy
import os

import pytest

from engine.canonical_stack_validator import validate_canonical_stack
from engine.canonical_validator import (
    canonical_enforced,
    enforce_official_env,
    validate_mode_is_canonical,
)
from engine.causal_evaluator import CausalEvaluator
from engine.evaluator import create_evaluator_factory, evaluate
from engine.governance_engine import GovernanceEngine


def _build_legacy_state():
    return {
        "timestamp": 1705441218.528,
        "symbol": "EURUSD",
        "tick": {"bid": 1.0850, "ask": 1.0852, "spread": 2.0},
        "indicators": {
            "rsi_14": 35.0,
            "sma_50": 1.0835,
            "sma_200": 1.0800,
            "atr_14": 0.0012,
            "volatility": 0.5,
        },
        "trend": {"regime": "uptrend", "strength": 0.75},
        "sentiment": {"score": 0.2, "confidence": 0.3, "source": "test"},
        "candles": {
            "M1": {"indicators": {"rsi_14": 25.0}},
            "M5": {"indicators": {"rsi_14": 32.0}},
            "M15": {"indicators": {"rsi_14": 35.0}},
            "H1": {"indicators": {"rsi_14": 38.0}},
        },
        "health": {"is_stale": False, "errors": []},
    }


def test_legacy_evaluator_blocked_in_official_mode(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MODE", "1")
    with pytest.raises(ValueError):
        evaluate(_build_legacy_state())
    monkeypatch.delenv("OFFICIAL_MODE", raising=False)


def test_canonical_factory_requires_causal_in_official(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MODE", "1")
    with pytest.raises(ValueError):
        create_evaluator_factory(use_causal=False)
    monkeypatch.delenv("OFFICIAL_MODE", raising=False)


def test_causal_evaluator_marks_official_env(monkeypatch):
    monkeypatch.delenv("OFFICIAL_MODE", raising=False)
    CausalEvaluator(official_mode=True)
    assert canonical_enforced()
    monkeypatch.delenv("OFFICIAL_MODE", raising=False)
    monkeypatch.delenv("CANONICAL_STACK_ONLY", raising=False)


def test_ml_hints_are_advisory_only(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MODE", "0")
    monkeypatch.setenv("CANONICAL_STACK_ONLY", "0")
    state = _build_legacy_state()
    result = evaluate(state)
    details = result["details"]
    assert details.get("ml_conf_adj", 0.0) == 0.0


def test_governance_blocks_extreme_volatility():
    governance = GovernanceEngine()
    market_state = {
        "volatility_state": {"vol_regime": "EXTREME"},
        "liquidity_state": {"liquidity_shock": False},
        "regime_state": {"macro_regime": "NORMAL", "regime_transition": False},
        "timestamp": 0,
    }
    eval_output = {"eval_score": 0.3}
    policy_decision = {"action": "BUY", "size": 1.0}
    execution = {"unrealized_pnl": 0.0}
    decision = governance.decide(market_state, eval_output, policy_decision, execution)
    assert decision.approved is False
    assert decision.adjusted_action == "FLAT"


def test_deterministic_evaluator_reproducible(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MODE", "0")
    monkeypatch.setenv("CANONICAL_STACK_ONLY", "0")
    state = _build_legacy_state()
    res1 = evaluate(copy.deepcopy(state))
    res2 = evaluate(copy.deepcopy(state))
    assert res1 == res2


def test_validate_mode_is_canonical_blocks_non_causal(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MODE", "1")
    with pytest.raises(ValueError):
        validate_mode_is_canonical(use_causal=False, governance_checked=True)
    monkeypatch.delenv("OFFICIAL_MODE", raising=False)


def test_canonical_stack_validator_blocks_legacy_and_ml(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MODE", "1")
    # Non-causal should fail
    with pytest.raises(ValueError):
        validate_canonical_stack(
            context="test_non_causal",
            use_causal=False,
            legacy_path_detected=False,
            governance_checked=True,
        )
    # Legacy path should fail
    with pytest.raises(ValueError):
        validate_canonical_stack(
            context="test_legacy",
            use_causal=True,
            legacy_path_detected=True,
            governance_checked=True,
        )
    # ML influence should fail
    with pytest.raises(ValueError):
        validate_canonical_stack(
            context="test_ml_influence",
            use_causal=True,
            legacy_path_detected=False,
            governance_checked=True,
            ml_influence=True,
        )
    monkeypatch.delenv("OFFICIAL_MODE", raising=False)
