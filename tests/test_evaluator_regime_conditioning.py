import json

from engine.causal_evaluator import (
    clamp,
    combine_confidence,
    evaluate_state,
    normalize_factor,
)
from state.regime_engine import RegimeSignal
from state.schema import (
    ExecutionContext,
    LiquidityState,
    MacroNewsState,
    MarketState,
    OrderFlowState,
    PriceState,
    VolatilityState,
)


def build_state(regime_signal: RegimeSignal) -> MarketState:
    price = PriceState(
        timestamp=1700000000.0, mid=1.2, bid=1.1999, ask=1.2001, spread=0.2
    )
    order_flow = OrderFlowState(
        aggressive_buy_volume=1000, aggressive_sell_volume=900, net_imbalance=100
    )
    liquidity = LiquidityState(
        top_depth_bid=100000,
        top_depth_ask=100000,
        cumulative_depth_bid=200000,
        cumulative_depth_ask=200000,
        depth_imbalance=0.0,
        liquidity_resilience=0.0,
        liquidity_pressure=0.2,
        liquidity_shock=False,
        regime="THIN",
    )
    volatility = VolatilityState(
        realized_vol=0.1,
        intraday_band_width=0.2,
        vol_of_vol=0.05,
        vol_regime="HIGH",
    )
    macro = MacroNewsState(
        hawkishness=0.05,
        risk_sentiment=0.1,
        surprise_score=0.2,
        macro_regime="RISK_OFF",
    )
    execution = ExecutionContext(
        position_size=0.0, avg_entry_price=None, unrealized_pnl=0.0, realized_pnl=0.0
    )
    raw = {"regime": regime_signal.to_dict()}
    return MarketState(
        price=price,
        order_flow=order_flow,
        liquidity=liquidity,
        volatility=volatility,
        macro=macro,
        execution=execution,
        raw=raw,
    )


def test_evaluate_state_deterministic_output():
    regime = RegimeSignal(vol="HIGH", liq="THIN", macro="RISK_OFF", confidence=0.8)
    state = build_state(regime)

    result1 = evaluate_state(state)
    result2 = evaluate_state(state)

    assert result1 == result2

    # Expected deterministic values from the formula
    expected_vol_raw = normalize_factor(0.5 - 0.1)  # 0.4
    expected_vol_weighted = clamp(expected_vol_raw * 0.4, -1.0, 1.0)

    depth_total = 200000 + 200000
    expected_liq_raw = normalize_factor(depth_total / 1_000_000.0 - 0.2 * 0.1)
    expected_liq_weighted = clamp(expected_liq_raw * 0.2, -1.0, 1.0)

    expected_macro_raw = normalize_factor(0.1 + 0.2 * 0.2 - 0.05 * 0.1)
    expected_macro_weighted = clamp(expected_macro_raw * 0.25, -1.0, 1.0)

    expected_score = clamp(
        expected_vol_weighted + expected_liq_weighted + expected_macro_weighted,
        -1.0,
        1.0,
    )
    expected_confidence = combine_confidence(
        [expected_vol_raw, expected_liq_raw, expected_macro_raw]
    )

    assert abs(result1["score"] - expected_score) < 1e-9
    assert abs(result1["confidence"] - expected_confidence) < 1e-9
    assert result1["factors"]["volatility"]["weighted"] == expected_vol_weighted
    assert result1["factors"]["liquidity"]["weighted"] == expected_liq_weighted
    assert result1["factors"]["macro"]["weighted"] == expected_macro_weighted
    assert "trend" in result1
    assert result1["trend"]["swing_structure"] in {
        "NEUTRAL",
        "HH",
        "HL",
        "LH",
        "LL",
    }
    assert "trend_direction" in result1["trend"]


def test_regime_weights_change_score():
    regime_high_thin = RegimeSignal(
        vol="HIGH", liq="THIN", macro="RISK_OFF", confidence=0.8
    )
    regime_low_deep = RegimeSignal(
        vol="LOW", liq="DEEP", macro="RISK_ON", confidence=0.8
    )

    state_high = build_state(regime_high_thin)
    state_low = build_state(regime_low_deep)

    res_high = evaluate_state(state_high)
    res_low = evaluate_state(state_low)

    # Different regimes should yield different weighted scores
    assert res_high["score"] != res_low["score"]


def test_json_serializable():
    regime = RegimeSignal(vol="HIGH", liq="THIN", macro="RISK_OFF", confidence=0.8)
    state = build_state(regime)
    result = evaluate_state(state)
    dumped = json.dumps(result)
    assert isinstance(dumped, str)
