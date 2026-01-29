from candle_pattern_features import compute_candle_pattern_features


def _series(vals):
    prices = []
    ts = []
    t = 0.0
    for v in vals:
        prices.append(v)
        ts.append(t)
        t += 60.0
    return prices, ts


def test_engulfing_detection_on_synthetic_series():
    # Bearish then bullish engulfing
    prices, _ = _series([100, 99, 101])
    feats = compute_candle_pattern_features(prices)
    assert feats["bullish_engulfing"] is True
    assert feats["bearish_engulfing"] is False


def test_inside_outside_bar_detection():
    prices, _ = _series([100, 101, 100.5, 100.6])  # last inside previous
    feats = compute_candle_pattern_features(prices)
    assert feats["inside_bar"] is True
    assert feats["outside_bar"] is False
    prices2, _ = _series([100, 101, 99, 102])
    feats2 = compute_candle_pattern_features(prices2)
    assert feats2["outside_bar"] is True


def test_pin_bar_detection_via_wick_body_ratios():
    prices, _ = _series(
        [100, 103, 101]
    )  # long upper wick (open=103 close=101 high=103 low=100)
    feats = compute_candle_pattern_features(prices)
    assert feats["pin_bar_upper"] is True


def test_momentum_vs_exhaustion_classification():
    prices, _ = _series([100, 100.5, 102.5, 104.5])
    feats = compute_candle_pattern_features(prices)
    assert feats["momentum_bar"] is True
    prices_exh, _ = _series([104.5, 105.0, 103.0])  # long lower wick after drop
    feats_exh = compute_candle_pattern_features(prices_exh)
    assert feats_exh["exhaustion_bar"] is True


def test_volume_aware_flags():
    prices, _ = _series([100, 100.2, 100.4, 100.6])
    vols = [100, 110, 90, 200]
    feats = compute_candle_pattern_features(prices, vols)
    assert feats["high_volume_candle"] is True
    vols2 = [100, 110, 90, 40]
    feats2 = compute_candle_pattern_features(prices, vols2)
    assert feats2["low_volume_candle"] is True


def test_pattern_context_flags_near_liquidity_and_structure():
    prices, _ = _series([100, 101, 100.5])
    context = {
        "bsl_zone_price": 100.5,
        "ssl_zone_price": 0.0,
        "nearest_bsl_pool_above": 100.6,
        "nearest_ssl_pool_below": 99.4,
        "swing_high": 100.5,
        "swing_low": 99.0,
    }
    feats = compute_candle_pattern_features(prices, context=context)
    assert feats["pattern_at_liquidity"] is True
    assert feats["pattern_at_structure"] is True
    assert feats["pattern_context_importance"] == "HIGH"


def test_replay_live_parity_for_candle_patterns():
    prices, _ = _series([100, 101, 100.5])
    live_prices = prices + [100.55]
    feats_replay = compute_candle_pattern_features(prices)
    feats_live = compute_candle_pattern_features(live_prices)
    assert feats_replay == feats_live
