from engine.evaluator import evaluate_state as eval_simple
from engine.types import MarketState


def _base_state(volatility_shock: bool = False, strength: float = 0.0) -> MarketState:
    state = MarketState()
    state.current_price = 105.0
    state.ma_short = 100.0
    state.ma_long = 95.0
    state.momentum = 0.5
    state.recent_returns = [0.01] * 5
    state.volatility = 0.5
    state.volatility_shock = volatility_shock
    state.volatility_shock_strength = strength
    return state


def test_evaluator_dampens_under_volatility_shock():
    base = _base_state(False, 0.0)
    shocked = _base_state(True, 0.9)

    base_out = eval_simple(base)
    shock_out = eval_simple(shocked)

    assert abs(shock_out.score) <= abs(base_out.score)
    assert shock_out.confidence <= base_out.confidence
    assert "volatility_shock" in shock_out.risk_flags
