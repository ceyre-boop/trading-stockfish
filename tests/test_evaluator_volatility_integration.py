from engine.volatility_features import VolatilityFeatures
from engine.volatility_utils import compute_atr


def test_volatility_outputs_bounded_with_flat_prices():
    vf = VolatilityFeatures(window=10)
    prices = [100.0] * 10
    atr_value = compute_atr(prices)

    out = None
    for price in prices:
        out = vf.compute(price, candle_data={"atr": atr_value})

    assert out is not None
    assert out["volatility_shock"] is False
    assert 0.0 <= out["volatility_shock_strength"] <= 1.0
    assert out["realized_vol"] >= 0.0
