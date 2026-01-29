from engine.policy_engine import select_action_regime
from state.regime_engine import RegimeSignal
from state.schema import (
    AMDState,
    ExecutionContext,
    LiquidityState,
    MacroNewsState,
    MarketState,
    OrderFlowState,
    PriceState,
    VolatilityState,
)


def test_policy_flattens_on_extreme_volatility_shock():
    state = MarketState(
        price=PriceState(),
        order_flow=OrderFlowState(),
        liquidity=LiquidityState(
            top_depth_bid=10000,
            top_depth_ask=10000,
            cumulative_depth_bid=20000,
            cumulative_depth_ask=20000,
            depth_imbalance=0.0,
            liquidity_resilience=0.1,
            liquidity_pressure=0.0,
            liquidity_shock=False,
            regime="NORMAL",
        ),
        volatility=VolatilityState(
            realized_vol=0.8,
            intraday_band_width=0.03,
            vol_of_vol=0.4,
            vol_regime="EXTREME",
            volatility_shock=True,
            volatility_shock_strength=0.9,
        ),
        macro=MacroNewsState(),
        execution=ExecutionContext(),
        amd=AMDState(),
        raw={
            "regime": RegimeSignal(
                vol="EXTREME",
                liq="THIN",
                macro="RISK_OFF",
                confidence=0.9,
                volatility_shock=True,
                volatility_shock_strength=0.9,
            )
        },
    )

    result = select_action_regime(state)

    assert result["action"] == "FLAT"
    assert result["target_size"] == 0.0
