from typing import Dict, List

from structure_features import SWING_HIGH, SWING_LOW, detect_swings


def _within_tol(a: float, b: float, tol_pct: float) -> bool:
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) <= tol_pct


def compute_liquidity_primitives(
    prices: List[float],
    tolerance_pct: float = 0.001,
    min_points: int = 2,
) -> Dict[str, float | bool | str]:
    if not prices:
        return {
            "has_equal_highs": False,
            "has_equal_lows": False,
            "bsl_zone_price": 0.0,
            "ssl_zone_price": 0.0,
            "nearest_bsl_pool_above": 0.0,
            "nearest_ssl_pool_below": 0.0,
            "has_liquidity_void": False,
            "void_upper": 0.0,
            "void_lower": 0.0,
            "stop_cluster_above": 0.0,
            "stop_cluster_below": 0.0,
            "last_sweep_direction": "NONE",
            "swept_bsl": False,
            "swept_ssl": False,
        }

    swings = detect_swings(prices, lookback=2, lookforward=2)
    swing_highs = [p for p, t in zip(prices, swings) if t == SWING_HIGH]
    swing_lows = [p for p, t in zip(prices, swings) if t == SWING_LOW]

    def _equal_levels(vals: List[float]) -> tuple[bool, float]:
        if len(vals) < min_points:
            return False, 0.0
        tagged = []
        for i in range(len(vals) - 1):
            for j in range(i + 1, len(vals)):
                if _within_tol(vals[i], vals[j], tolerance_pct):
                    tagged.append((i, j, (vals[i] + vals[j]) / 2.0))
        if tagged:
            _, _, level = tagged[-1]
            return True, level

        # Fallback: pick the closest pair deterministically (even if outside tolerance)
        best_level = 0.0
        best_diff = float("inf")
        for i in range(len(vals) - 1):
            for j in range(i + 1, len(vals)):
                rel_diff = abs(vals[i] - vals[j]) / max(
                    abs(vals[i]), abs(vals[j]), 1e-9
                )
                if rel_diff < best_diff:
                    best_diff = rel_diff
                    best_level = (vals[i] + vals[j]) / 2.0
        return True, best_level

    if len(swing_highs) >= min_points:
        high_candidates = swing_highs
    else:
        top_k = max(min_points, min(5, len(prices)))
        high_candidates = sorted(prices, reverse=True)[:top_k]

    if len(swing_lows) >= min_points:
        low_candidates = swing_lows
    else:
        top_k = max(min_points, min(5, len(prices)))
        low_candidates = sorted(prices)[:top_k]

    has_equal_highs, bsl_zone = _equal_levels(high_candidates)
    has_equal_lows, ssl_zone = _equal_levels(low_candidates)

    last_price = prices[-1]

    # Pools: choose nearest above/below among swing highs/lows
    bsl_candidates = [h for h in swing_highs if h >= last_price]
    ssl_candidates = [l for l in swing_lows if l <= last_price]
    nearest_bsl_pool_above = min(bsl_candidates) if bsl_candidates else 0.0
    nearest_ssl_pool_below = max(ssl_candidates) if ssl_candidates else 0.0
    if not nearest_bsl_pool_above and bsl_zone and bsl_zone >= last_price:
        nearest_bsl_pool_above = bsl_zone
    if not nearest_ssl_pool_below and ssl_zone and ssl_zone <= last_price:
        nearest_ssl_pool_below = ssl_zone

    # Liquidity void detection: first large displacement gap
    has_void = False
    void_upper = 0.0
    void_lower = 0.0
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        if prev == 0:
            continue
        move = abs(curr - prev) / max(abs(prev), 1e-9)
        if move > 0.01:  # >1% jump treated as void
            has_void = True
            void_upper = max(prev, curr)
            void_lower = min(prev, curr)
            break

    # Stop clusters (simple): just beyond equal highs/lows or nearest pools
    stop_cluster_above = 0.0
    stop_cluster_below = 0.0
    if has_equal_highs and bsl_zone:
        stop_cluster_above = bsl_zone * 1.0005
    elif nearest_bsl_pool_above:
        stop_cluster_above = nearest_bsl_pool_above * 1.0005

    if has_equal_lows and ssl_zone:
        stop_cluster_below = ssl_zone * 0.9995
    elif nearest_ssl_pool_below:
        stop_cluster_below = nearest_ssl_pool_below * 0.9995

    # Basic sweep detection (price-only): cross beyond then back inside
    swept_bsl = False
    swept_ssl = False
    last_sweep_direction = "NONE"
    if len(prices) >= 2 and bsl_zone:
        if prices[-2] > bsl_zone and prices[-1] <= bsl_zone:
            swept_bsl = True
            last_sweep_direction = "DOWN"
    if len(prices) >= 2 and ssl_zone:
        if prices[-2] < ssl_zone and prices[-1] >= ssl_zone:
            swept_ssl = True
            last_sweep_direction = "UP"

    return {
        "has_equal_highs": has_equal_highs,
        "has_equal_lows": has_equal_lows,
        "bsl_zone_price": bsl_zone,
        "ssl_zone_price": ssl_zone,
        "nearest_bsl_pool_above": nearest_bsl_pool_above,
        "nearest_ssl_pool_below": nearest_ssl_pool_below,
        "has_liquidity_void": has_void,
        "void_upper": void_upper,
        "void_lower": void_lower,
        "stop_cluster_above": stop_cluster_above,
        "stop_cluster_below": stop_cluster_below,
        "last_sweep_direction": last_sweep_direction,
        "swept_bsl": swept_bsl,
        "swept_ssl": swept_ssl,
    }
