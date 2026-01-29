"""Demonstration for evaluator v1.3 + policy v1.1 with regime subsystem."""

from engine.evaluator import evaluate_state
from engine.policy_engine import select_action
from engine.types import Action, MarketState


def main() -> None:
    # Scenario 1: Strong uptrend, high liquidity, risk_on
    bullish_state = MarketState(
        current_price=1.2500,
        recent_returns=[0.002, 0.0015, 0.0018],
        volatility=0.003,
        liquidity=0.9,
        volume=600_000.0,
        trend_regime="chop",  # recomputed
        volatility_regime="normal",  # recomputed
        liquidity_regime="normal",  # recomputed
        macro_regime="neutral",  # recomputed
        ma_short=1.2600,
        ma_long=1.2000,
        rsi=65.0,
        momentum=0.08,
        atr=0.0006,
        spread=0.0001,
        position_side="flat",
        position_size=0.0,
    )

    eval_bull = evaluate_state(bullish_state)
    action_bull: Action = select_action(bullish_state, eval_bull)

    print("=== Bullish (risk_on) Scenario ===")
    print(
        "Regimes:",
        eval_bull.trend_regime,
        eval_bull.volatility_regime,
        eval_bull.liquidity_regime,
        eval_bull.macro_regime,
    )
    print("Score/Conf:", round(eval_bull.score, 4), round(eval_bull.confidence, 4))
    print("Risk Flags:", eval_bull.risk_flags)
    print("Action:", action_bull)

    # Scenario 2: Choppy trend, high volatility, risk_off triggers soft veto/size cut
    bearish_state = MarketState(
        current_price=0.9950,
        recent_returns=[-0.003, 0.0025, -0.0022],
        volatility=0.006,
        liquidity=0.35,
        volume=300_000.0,
        trend_regime="chop",  # recomputed
        volatility_regime="normal",  # recomputed
        liquidity_regime="normal",  # recomputed
        macro_regime="neutral",  # recomputed
        ma_short=0.9980,
        ma_long=1.0000,
        rsi=35.0,
        momentum=-0.06,
        atr=0.0008,
        spread=0.0001,
        position_side="flat",
        position_size=0.0,
    )

    eval_bear = evaluate_state(bearish_state)
    action_bear: Action = select_action(bearish_state, eval_bear)

    print("\n=== Choppy / Risk_off Scenario ===")
    print(
        "Regimes:",
        eval_bear.trend_regime,
        eval_bear.volatility_regime,
        eval_bear.liquidity_regime,
        eval_bear.macro_regime,
    )
    print("Score/Conf:", round(eval_bear.score, 4), round(eval_bear.confidence, 4))
    print("Risk Flags:", eval_bear.risk_flags)
    print("Action:", action_bear)


if __name__ == "__main__":
    main()
