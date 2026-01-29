import pytest

from engine.volatility_features import VolatilityFeatures


def test_volatility_shock_strength_bounds():
    vf = VolatilityFeatures(window=5)
    neutral = vf.compute(100.0, candle_data={"atr": 0.1})
    assert neutral["volatility_shock"] is False
    assert 0.0 <= neutral["volatility_shock_strength"] <= 1.0

    prices = [100.2, 102.5, 105.0, 108.0]
    res = neutral
    for p in prices:
        res = vf.compute(p, candle_data={"atr": 2.0})

    assert res["volatility_shock"] is True
    assert 0.3 <= res["volatility_shock_strength"] <= 1.0


def test_volatility_shock_neutral_with_insufficient_history():
    vf = VolatilityFeatures(window=5)
    res = vf.compute(50.0)
    assert res["volatility_shock"] is False
    assert res["volatility_shock_strength"] == 0.0
