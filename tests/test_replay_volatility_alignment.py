from engine.volatility_features import VolatilityFeatures
from engine.volatility_utils import compute_atr


def test_live_and_replay_volatility_match():
    prices = [100.0, 100.3, 100.6, 100.4, 101.0, 101.2]
    atr_series = [compute_atr(prices[: i + 1]) for i in range(len(prices))]

    live = VolatilityFeatures(window=5)
    replay = VolatilityFeatures(window=5)

    live_out = None
    for price, atr in zip(prices, atr_series):
        live_out = live.compute(price, candle_data={"atr": atr})

    replay_out = None
    for price, atr in zip(prices, atr_series):
        replay_out = replay.compute(price, candle_data={"atr": atr})

    assert live_out == replay_out
    assert (
        live_out["volatility_shock_strength"] == replay_out["volatility_shock_strength"]
    )
