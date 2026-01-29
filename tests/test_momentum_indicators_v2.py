from momentum_indicators_v2 import (
    _detect_divergence,
    compute_macd,
    compute_momentum_indicators,
    compute_rsi,
    compute_stochastic,
)


def test_rsi_against_known_series():
    prices_up = [float(i) for i in range(1, 30)]
    rsi_up = compute_rsi(prices_up, period=14)
    assert 50.0 < rsi_up[-1] <= 100.0

    prices_down = [float(100 - i) for i in range(30)]
    rsi_down = compute_rsi(prices_down, period=14)
    assert 0.0 <= rsi_down[-1] < 50.0


def test_macd_line_signal_histogram_signs():
    prices = [100 + i * 0.5 for i in range(60)]
    macd = compute_macd(prices)
    assert macd.macd_line[-1] > macd.signal_line[-1]
    assert macd.histogram[-1] > 0


def test_stochastic_bounds_and_state_classification():
    highs = [10 + i * 0.2 for i in range(20)]
    lows = [9 + i * 0.2 for i in range(20)]
    closes = [9.95 + i * 0.2 for i in range(20)]
    stoch = compute_stochastic(highs, lows, closes)
    assert 0.0 <= stoch.k[-1] <= 100.0
    assert 0.0 <= stoch.d[-1] <= 100.0
    # Close near high should be overbought
    feats = compute_momentum_indicators(closes, highs=highs, lows=lows)
    assert feats["stoch_state"] in ("OVERBOUGHT", "NEUTRAL")


def test_momentum_regime_classification():
    trend_prices = [100 + i * 0.5 for i in range(80)]
    feats_trend = compute_momentum_indicators(trend_prices)
    assert feats_trend["momentum_regime"] == "STRONG_TREND"

    chop_prices = [100 + ((-1) ** i) * 0.05 for i in range(80)]
    feats_chop = compute_momentum_indicators(chop_prices)
    assert feats_chop["momentum_regime"] == "CHOP"


def test_divergence_detection_on_synthetic_pairs():
    prices_bull = [10.0, 9.5, 9.0, 8.5, 8.0]
    indicator_bull = [30.0, 29.0, 28.0, 29.0, 30.0]
    bull, bear = _detect_divergence(prices_bull, indicator_bull)
    assert bull is True and bear is False

    prices_bear = [10.0, 10.5, 11.0, 11.5, 12.0]
    indicator_bear = [70.0, 71.0, 70.5, 69.5, 68.0]
    bull2, bear2 = _detect_divergence(prices_bear, indicator_bear)
    assert bear2 is True and bull2 is False


def test_replay_live_parity_for_momentum_indicators_v2():
    prices = [100.0, 100.5, 101.0, 101.5]
    live_prices = prices + [101.50005]
    replay = compute_momentum_indicators(prices)
    live = compute_momentum_indicators(live_prices)
    assert replay == live
