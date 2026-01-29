from cockpit_telemetry import build_cockpit_snapshot, to_dict, to_json
from engine.types import MarketState


def _sample_state() -> MarketState:
    return MarketState(
        current_price=101.0,
        ma_short=100.0,
        ma_long=99.0,
        momentum=0.1,
        recent_returns=[0.01] * 10,
        volatility=0.2,
        liquidity=0.5,
        rsi=55.0,
        session="LONDON",
        momentum_5=0.02,
        momentum_10=0.03,
        momentum_20=0.04,
        roc_5=0.02,
        roc_10=0.03,
        roc_20=0.04,
        bid_depth=10.0,
        ask_depth=8.0,
        depth_imbalance=0.111,
        spread=0.2,
        swing_high=102.0,
        swing_low=99.5,
    )


def test_cockpit_snapshot_fields_populated():
    snapshot = build_cockpit_snapshot(_sample_state())
    assert snapshot.evaluator.trend_regime
    assert snapshot.scenario.selected_scenario_type
    assert snapshot.policy.action
    assert snapshot.governance.veto_reason == "APPROVED"
    assert snapshot.execution.simulated_fill_price > 0
    assert snapshot.market_structure.spread >= 0
    assert snapshot.ict_smc.premium_discount_state
    assert snapshot.orderflow.bar_delta == 0.0
    assert snapshot.quant.p_sweep_reversal == 0.5


def test_cockpit_snapshot_determinism():
    s1 = build_cockpit_snapshot(_sample_state())
    s2 = build_cockpit_snapshot(_sample_state())
    assert to_dict(s1) == to_dict(s2)


def test_cockpit_snapshot_serialization_roundtrip():
    snap = build_cockpit_snapshot(_sample_state())
    d = to_dict(snap)
    j = to_json(snap)
    import json

    assert json.loads(j) == d
