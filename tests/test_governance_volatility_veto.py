from engine.governance_engine import GovernanceEngine


def test_governance_blocks_extreme_volatility_shock():
    engine = GovernanceEngine()
    market_state = {
        "volatility_state": {
            "volatility_shock": True,
            "volatility_shock_strength": 0.9,
            "vol_regime": "EXTREME",
        },
        "liquidity_state": {"liquidity_shock": False},
        "timestamp": 0,
    }
    eval_output = {"eval_score": 0.5}
    policy_decision = {"action": "ENTER_FULL", "size": 1.0}
    execution = {"unrealized_pnl": 0.0}

    decision = engine.decide(market_state, eval_output, policy_decision, execution)

    assert decision.approved is False
    assert decision.reason in {"VOLATILITY_SHOCK", "VOLATILITY_SHOCK_SIZE_REDUCTION"}
    assert decision.adjusted_action in {"FLAT", "REDUCE"}
