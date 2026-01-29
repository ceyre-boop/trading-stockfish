import math

from engine.governance_engine import GovernanceEngine
from risk_constraints import (
    RiskConfig,
    RiskDecision,
    RiskState,
    check_trade_risk_allowed,
    update_risk_state_after_fill,
    update_risk_state_end_of_day,
)


def _base_state(equity: float = 100000.0) -> RiskState:
    return RiskState(
        current_equity=equity,
        open_risk=0.0,
        realized_pnl_today=0.0,
        peak_equity=equity,
        risk_used_today=0.0,
        current_positions=0,
    )


def test_max_risk_per_trade_enforced():
    cfg = RiskConfig(max_risk_per_trade=0.01)
    state = _base_state()
    decision = check_trade_risk_allowed(cfg, state, proposed_risk=2000.0)
    assert not decision.allowed
    assert decision.reason == "MAX_RISK_PER_TRADE"


def test_max_risk_per_day_enforced():
    cfg = RiskConfig(max_risk_per_trade=0.02, max_risk_per_day=0.03)
    state = _base_state()
    decision1 = check_trade_risk_allowed(cfg, state, proposed_risk=2000.0)
    state = update_risk_state_after_fill(state, fill_risk=0.02)
    decision2 = check_trade_risk_allowed(cfg, state, proposed_risk=2000.0)
    assert decision1.allowed
    assert not decision2.allowed
    assert decision2.reason == "MAX_RISK_PER_DAY"


def test_max_drawdown_enforced():
    cfg = RiskConfig(max_drawdown=0.05)
    state = RiskState(
        current_equity=90000.0,
        open_risk=0.0,
        realized_pnl_today=-10000.0,
        peak_equity=100000.0,
        risk_used_today=0.0,
        current_positions=0,
    )
    decision = check_trade_risk_allowed(cfg, state, proposed_risk=500.0)
    assert not decision.allowed
    assert decision.reason == "MAX_DRAWDOWN"


def test_risk_state_updates_after_fill():
    state = _base_state()
    updated = update_risk_state_after_fill(
        state,
        fill_risk=0.01,
        fill_realized_pnl=500.0,
        fill_unrealized_risk_delta=0.02,
        position_delta=1,
    )
    assert math.isclose(updated.current_equity, state.current_equity + 500.0)
    assert updated.risk_used_today > state.risk_used_today
    assert updated.open_risk > state.open_risk
    assert updated.current_positions == 1


def test_end_of_day_resets_daily_counters():
    state = _base_state()
    state = update_risk_state_after_fill(
        state, fill_risk=0.02, fill_realized_pnl=1000.0
    )
    eod = update_risk_state_end_of_day(state)
    assert eod.risk_used_today == 0.0
    assert eod.realized_pnl_today == 0.0
    assert eod.current_equity == state.current_equity


def test_determinism_same_sequence_same_state():
    cfg = RiskConfig(max_risk_per_trade=0.02, max_risk_per_day=0.04)
    seq = [0.01, 0.01, 0.015]

    def run_sequence():
        st = _base_state()
        for r in seq:
            decision = check_trade_risk_allowed(
                cfg, st, proposed_risk=r * st.current_equity
            )
            if decision.allowed:
                st = update_risk_state_after_fill(st, fill_risk=decision.approved_risk)
        return st

    s1 = run_sequence()
    s2 = run_sequence()
    assert s1 == s2


def test_governance_veto_reason_for_risk_violations():
    gov = GovernanceEngine()
    decision = gov.decide(
        market_state={},
        eval_output={},
        policy_decision={
            "action": "ENTER",
            "risk_allowed": False,
            "risk_veto_reason": "MAX_RISK_PER_TRADE",
        },
        execution={},
    )
    assert not decision.approved
    assert decision.reason == "MAX_RISK_PER_TRADE"
