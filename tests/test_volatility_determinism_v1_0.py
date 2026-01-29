import pytest

from engine.volatility_features import VolatilityFeatures
from engine.volatility_utils import compute_atr


def test_early_history_neutral():
    vf = VolatilityFeatures(window=5)
    prices = [100.0, 100.1, 100.05]
    out = None
    for p in prices:
        out = vf.compute(p, candle_data={"atr": 0.5})

    assert out is not None
    assert out["volatility_shock"] is False
    assert out["volatility_shock_strength"] == 0.0


def test_atr_missing_graceful():
    vf = VolatilityFeatures(window=5)
    prices = [100.0, 100.02, 100.04, 100.03, 100.05]
    out = None
    for p in prices:
        out = vf.compute(p)

    assert out is not None
    assert out["volatility_shock"] is False
    assert 0.0 <= out["volatility_shock_strength"] <= 1.0


def test_band_break_threshold_regression():
    vf = VolatilityFeatures(window=5)
    prices = [100.0, 100.5, 101.0, 103.0, 106.0]
    out = None
    for p in prices:
        out = vf.compute(p, candle_data={"atr": 1.5})

    assert out is not None
    assert out["volatility_shock_strength"] >= 0.3


def test_repeated_tick_invariance():
    vf = VolatilityFeatures(window=5)
    first = vf.compute(100.0, candle_data={"atr": 0.2})
    repeat = vf.compute(100.0, candle_data={"atr": 0.2})

    assert first == repeat
    assert repeat["volatility_shock"] is False
    assert repeat["volatility_shock_strength"] == 0.0


def test_compute_atr_floor_is_used():
    atr = compute_atr(None)
    assert atr > 0
    atr_from_scalar = compute_atr(0.0)
    assert atr_from_scalar > 0
