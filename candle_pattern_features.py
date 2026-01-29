"""
Deterministic candle pattern engine v1.
Computes candle anatomy metrics, core pattern flags, simple volume-aware signals,
and context-aware flags using existing structure/liquidity references.
"""

from typing import Any, Dict, Sequence

from structure_features import LEG_CORRECTION, LEG_IMPULSE


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _avg(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _body_range(prices: Sequence[float]) -> Dict[str, float]:
    if len(prices) < 2:
        return {
            "open": prices[-1] if prices else 0.0,
            "close": prices[-1] if prices else 0.0,
            "high": prices[-1] if prices else 0.0,
            "low": prices[-1] if prices else 0.0,
        }
    open_ = prices[-2]
    close = prices[-1]
    window = prices[-3:] if len(prices) >= 3 else prices[-2:]
    high = max(window)
    low = min(window)
    return {"open": open_, "close": close, "high": high, "low": low}


def compute_candle_pattern_features(
    prices: Sequence[float],
    volumes: Sequence[float] | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context = context or {}
    if len(prices) >= 2:
        latest_delta = abs(prices[-1] - prices[-2])
        threshold = max(abs(prices[-2]) * 0.001, 0.01)
        if latest_delta <= threshold:
            prices = prices[:-1]
    if len(prices) < 2:
        return {
            "body_size": 0.0,
            "upper_wick_size": 0.0,
            "lower_wick_size": 0.0,
            "total_range": 0.0,
            "wick_to_body_upper": 0.0,
            "wick_to_body_lower": 0.0,
            "wick_to_body_total": 0.0,
            "bullish_engulfing": False,
            "bearish_engulfing": False,
            "inside_bar": False,
            "outside_bar": False,
            "pin_bar_upper": False,
            "pin_bar_lower": False,
            "momentum_bar": False,
            "exhaustion_bar": False,
            "high_volume_candle": False,
            "low_volume_candle": False,
            "pattern_at_liquidity": False,
            "pattern_at_structure": False,
            "pattern_context_importance": "LOW",
        }

    candle = _body_range(prices)
    open_ = candle["open"]
    close = candle["close"]
    high = candle["high"]
    low = candle["low"]
    body = abs(close - open_)
    total_range = max(high - low, 0.0)
    upper_wick = max(high - max(open_, close), 0.0)
    lower_wick = max(min(open_, close) - low, 0.0)
    spare_range = max(total_range - body - upper_wick - lower_wick, 0.0)
    if spare_range > 0:
        upper_wick += spare_range * 0.5
        lower_wick += spare_range * 0.5

    wick_to_body_upper = _safe_div(upper_wick, body or abs(close) * 1e-6)
    wick_to_body_lower = _safe_div(lower_wick, body or abs(close) * 1e-6)
    wick_to_body_total = _safe_div(upper_wick + lower_wick, body or abs(close) * 1e-6)

    # Previous candle anatomy
    prev = _body_range(prices[:-1])
    prev_body = abs(prev["close"] - prev["open"])
    prev_high = max(prev["high"], prev["open"], prev["close"])
    prev_low = min(prev["low"], prev["open"], prev["close"])

    bullish_engulfing = (
        close > open_
        and prev["close"] < prev["open"]
        and high >= prev_high
        and low <= prev_low
        and body >= prev_body * 1.05
    )
    bearish_engulfing = (
        close < open_
        and prev["close"] > prev["open"]
        and high >= prev_high
        and low <= prev_low
        and body >= prev_body * 1.05
    )

    inside_bar = high <= prev_high and low >= prev_low
    outside_bar = high >= prev_high and low <= prev_low and not inside_bar

    pin_bar_upper = (high - min(open_, close)) >= body * 1.0 and (
        max(open_, close) - low
    ) <= body * 1.8
    pin_bar_lower = (max(open_, close) - low) >= body * 1.0 and (
        high - min(open_, close)
    ) <= body * 1.8

    body_history = []
    ranges = []
    for i in range(1, min(len(prices), 10)):
        o = prices[-(i + 1)]
        c = prices[-i]
        hi = max(prices[-(i + 2) :] if i + 2 <= len(prices) else prices[-(i + 1) :])
        lo = min(prices[-(i + 2) :] if i + 2 <= len(prices) else prices[-(i + 1) :])
        body_history.append(abs(c - o))
        ranges.append(max(hi - lo, 0.0))
    avg_body = _avg(body_history) or body
    avg_range = _avg(ranges) or total_range or abs(close) * 0.001

    momentum_bar = (
        body >= avg_body * 1.3 and _safe_div(body, total_range or 1e-6) >= 0.5
    )

    prior_direction = "UP" if close > open_ else "DOWN" if close < open_ else "RANGE"
    exhaustion_bar = False
    if prior_direction == "UP" and upper_wick >= body * 1.5 and lower_wick <= body:
        exhaustion_bar = True
    if prior_direction == "DOWN" and lower_wick >= body * 1.5 and upper_wick <= body:
        exhaustion_bar = True
    if prior_direction == "DOWN" and total_range >= body * 1.0:
        exhaustion_bar = True

    # Volume-aware flags
    high_volume = False
    low_volume = False
    if volumes and len(volumes) >= 2:
        curr_vol = volumes[-1]
        avg_vol = _avg(volumes[-min(20, len(volumes)) :])
        if avg_vol > 0:
            high_volume = curr_vol >= avg_vol * 1.3
            low_volume = curr_vol <= avg_vol * 0.7

    # Context flags
    close_price = close
    refs = [
        context.get(k)
        for k in [
            "bsl_zone_price",
            "ssl_zone_price",
            "nearest_bsl_pool_above",
            "nearest_ssl_pool_below",
        ]
        if context.get(k) is not None
    ]
    pattern_at_liquidity = any(
        abs(close_price - ref) <= avg_range * 0.5 for ref in refs
    )
    struct_refs = [
        context.get(k)
        for k in [
            "swing_high",
            "swing_low",
            "current_bullish_ob_high",
            "current_bullish_ob_low",
            "current_bearish_ob_high",
            "current_bearish_ob_low",
        ]
        if context.get(k) is not None
    ]
    pattern_at_structure = any(
        abs(close_price - ref) <= avg_range * 0.5 for ref in struct_refs
    )

    if pattern_at_liquidity and pattern_at_structure:
        context_importance = "HIGH"
    elif pattern_at_liquidity or pattern_at_structure:
        context_importance = "MEDIUM"
    else:
        context_importance = "LOW"

    return {
        "body_size": body,
        "upper_wick_size": upper_wick,
        "lower_wick_size": lower_wick,
        "total_range": total_range,
        "wick_to_body_upper": wick_to_body_upper,
        "wick_to_body_lower": wick_to_body_lower,
        "wick_to_body_total": wick_to_body_total,
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
        "inside_bar": inside_bar,
        "outside_bar": outside_bar,
        "pin_bar_upper": pin_bar_upper,
        "pin_bar_lower": pin_bar_lower,
        "momentum_bar": momentum_bar,
        "exhaustion_bar": exhaustion_bar,
        "high_volume_candle": high_volume,
        "low_volume_candle": low_volume,
        "pattern_at_liquidity": pattern_at_liquidity,
        "pattern_at_structure": pattern_at_structure,
        "pattern_context_importance": context_importance,
    }
