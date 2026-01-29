from liquidity_primitives import compute_liquidity_primitives


def test_equal_highs_lows_detection():
    prices = [100, 101, 100.99, 101.01, 100.5, 99.5, 99.51, 99.49]
    feats = compute_liquidity_primitives(prices, tolerance_pct=0.002)
    assert feats["has_equal_highs"]
    assert feats["has_equal_lows"]
    assert feats["bsl_zone_price"] > 0
    assert feats["ssl_zone_price"] > 0


def test_bsl_ssl_zone_tagging():
    prices = [10, 10.1, 10.09, 10.11, 9.9, 9.89, 9.91]
    feats = compute_liquidity_primitives(prices, tolerance_pct=0.002)
    assert feats["bsl_zone_price"] >= 10.09
    assert feats["ssl_zone_price"] > 0


def test_liquidity_pool_selection():
    prices = [10, 10.5, 10.4, 10.6, 10.45, 10.55]
    feats = compute_liquidity_primitives(prices)
    assert feats["nearest_bsl_pool_above"] >= prices[-1]
    assert (
        feats["nearest_ssl_pool_below"] == 0
        or feats["nearest_ssl_pool_below"] <= prices[-1]
    )


def test_liquidity_void_detection():
    prices = [100, 100.5, 102.0]  # >1% jump
    feats = compute_liquidity_primitives(prices)
    assert feats["has_liquidity_void"]
    assert feats["void_upper"] > feats["void_lower"]


def test_stop_cluster_placement():
    prices = [10, 10.1, 10.09, 10.11, 9.9, 9.89, 9.91]
    feats = compute_liquidity_primitives(prices, tolerance_pct=0.002)
    assert feats["stop_cluster_above"] >= feats["bsl_zone_price"]
    assert feats["stop_cluster_below"] <= feats["ssl_zone_price"]


def test_basic_sweep_detection():
    # Sweep above equal highs then close back below
    prices = [10, 10.1, 10.09, 10.11, 10.12, 10.05]
    feats = compute_liquidity_primitives(prices, tolerance_pct=0.002)
    assert feats["swept_bsl"] or feats["swept_ssl"]
    assert feats["last_sweep_direction"] in {"UP", "DOWN", "NONE"}


def test_replay_live_parity_for_liquidity_primitives():
    prices = [1, 1.01, 1.0, 1.02, 1.0]
    first = compute_liquidity_primitives(prices)
    second = compute_liquidity_primitives(list(prices))
    assert first == second
