import pytest

import engine.policy_engine as policy_engine
from engine.canonical_stack_validator import validate_canonical_stack
from engine.evaluator import evaluate_state as core_evaluate_state
from engine.governance_engine import GovernanceEngine
from engine.regime_engine import RegimeEngine
from engine.types import MarketState as EvalMarketState
from state.regime_engine import RegimeSignal
from state.schema import AMDState, MarketState


def test_schema_exposes_amd_defaults():
    state = MarketState()
    assert state.amd.amd_tag == "NEUTRAL"
    assert state.amd.amd_confidence == 0.0


def test_regime_engine_emits_amd_regime():
    regime = RegimeEngine(window=3)
    volatility_state = {"vol_regime": "NORMAL", "realized_vol": 0.01}
    liquidity_state = {"liquidity_resilience": 0.1, "depth_imbalance": 0.0}
    macro_state = {"hawkishness": 0.0, "risk_sentiment": 0.0}
    amd_state = {"amd_tag": "ACCUMULATION", "amd_confidence": 0.6}

    res = regime.compute(
        volatility_state,
        liquidity_state,
        macro_state,
        ml_state=None,
        amd_state=amd_state,
    )

    assert res.get("amd_regime") == "ACCUMULATION"
    assert pytest.approx(res.get("amd_confidence", 0.0), rel=1e-6) == 0.6


def test_evaluator_amd_adjustments_balanced():
    base_state = EvalMarketState(
        current_price=101.0,
        ma_short=100.0,
        ma_long=99.0,
        momentum=0.1,
        recent_returns=[0.001, -0.001, 0.0005],
        volatility=0.002,
    )
    neutral = core_evaluate_state(base_state)

    bullish_state = EvalMarketState(
        **{**base_state.__dict__, "amd_regime": "ACCUMULATION", "amd_confidence": 0.8}
    )
    bearish_state = EvalMarketState(
        **{**base_state.__dict__, "amd_regime": "DISTRIBUTION", "amd_confidence": 0.8}
    )
    manip_state = EvalMarketState(
        **{**base_state.__dict__, "amd_regime": "MANIPULATION", "amd_confidence": 0.9}
    )

    bull = core_evaluate_state(bullish_state)
    bear = core_evaluate_state(bearish_state)
    manip = core_evaluate_state(manip_state)

    assert bull.score > neutral.score
    assert bear.score < neutral.score
    assert manip.score < neutral.score
    assert manip.confidence <= neutral.confidence


def test_policy_respects_manipulation(monkeypatch):
    state = MarketState(
        amd=AMDState(amd_tag="MANIPULATION", amd_confidence=0.9),
        raw={
            "regime": RegimeSignal(
                vol="NORMAL", liq="NORMAL", macro="RISK_ON", confidence=0.5
            )
        },
    )

    monkeypatch.setattr(
        policy_engine,
        "evaluate_state",
        lambda _state: {"score": 0.5, "confidence": 0.9},
    )

    decision = policy_engine.select_action_regime(state)
    assert decision["action"] == "FLAT"
    assert decision["target_size"] == 0.0
    assert decision["amd_regime"] == "MANIPULATION"


def test_governance_applies_manipulation_veto():
    gov = GovernanceEngine()
    market_state = {"amd_state": {"amd_tag": "MANIPULATION"}}
    eval_out = {"eval_score": 0.3}
    policy_decision = {"action": "ENTER_FULL", "size": 1.0}
    execution = {"unrealized_pnl": 0.0}

    decision = gov.decide(market_state, eval_out, policy_decision, execution)
    assert decision.approved is False
    assert decision.reason == "AMD_MANIPULATION_VETO"
    assert decision.adjusted_action == "FLAT"


def test_canonical_validator_enforces_amd_when_enabled(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MODE", "1")
    # Happy path
    validate_canonical_stack(
        context="amd_ok",
        use_causal=True,
        governance_checked=True,
        amd_checks=True,
        amd_tag="ACCUMULATION",
        amd_regime_present=True,
        manipulation_veto_active=True,
    )

    # Failure when veto missing on manipulation
    with pytest.raises(ValueError):
        validate_canonical_stack(
            context="amd_missing_veto",
            use_causal=True,
            governance_checked=True,
            amd_checks=True,
            amd_tag="MANIPULATION",
            amd_regime_present=True,
            manipulation_veto_active=False,
        )
    monkeypatch.delenv("OFFICIAL_MODE", raising=False)
