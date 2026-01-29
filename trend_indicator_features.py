"""
Deterministic trend indicator features.
Computes moving averages, stacking state, distance metrics, and a simple
ADX-style trend strength.
"""

from typing import Dict, List


def _safe_last(values: List[float]) -> float:
    return float(values[-1]) if values else 0.0


def _sma(series: List[float], window: int) -> float:
    if not series:
        return 0.0
    window = max(1, window)
    start_idx = max(0, len(series) - window)
    window_slice = series[start_idx:]
    return sum(window_slice) / float(len(window_slice))


def _ema(series: List[float], window: int) -> float:
    if not series:
        return 0.0
    window = max(1, window)
    alpha = 2.0 / (window + 1)
    ema_value = series[0]
    for price in series[1:]:
        ema_value = (price - ema_value) * alpha + ema_value
    return ema_value


def _distance_from(price: float, reference: float) -> float:
    if price == 0.0:
        return 0.0
    raw = (price - reference) / abs(price)
    # Bound the distance to keep telemetry stable.
    if raw > 5.0:
        return 5.0
    if raw < -5.0:
        return -5.0
    return raw


def _compute_dmi_strength(prices: List[float], window: int = 14) -> float:
    if len(prices) < 2:
        return 0.0
    window = max(1, window)
    start_idx = max(1, len(prices) - window)
    up_moves: List[float] = []
    down_moves: List[float] = []
    true_ranges: List[float] = []
    for idx in range(start_idx, len(prices)):
        curr = prices[idx]
        prev = prices[idx - 1]
        up_move = max(curr - prev, 0.0)
        down_move = max(prev - curr, 0.0)
        true_range = abs(curr - prev)
        up_moves.append(up_move)
        down_moves.append(down_move)
        true_ranges.append(true_range)
    tr_sum = sum(true_ranges) or 1e-9
    dmi_plus = 100.0 * (sum(up_moves) / tr_sum)
    dmi_minus = 100.0 * (sum(down_moves) / tr_sum)
    dx = 100.0 * abs(dmi_plus - dmi_minus) / (dmi_plus + dmi_minus + 1e-9)
    if dx < 0.0:
        return 0.0
    if dx > 100.0:
        return 100.0
    return dx


def compute_trend_indicator_features(prices: List[float]) -> Dict[str, float]:
    ema_9 = _ema(prices, 9)
    ema_20 = _ema(prices, 20)
    ema_50 = _ema(prices, 50)
    ema_200 = _ema(prices, 200)
    sma_20 = _sma(prices, 20)
    sma_50 = _sma(prices, 50)
    sma_200 = _sma(prices, 200)

    last_price = _safe_last(prices)
    gap_tolerance = abs(last_price) * 0.001 + 1e-6

    stack_state = "NEUTRAL"
    if len(prices) >= 20:
        if (ema_9 - ema_20) > gap_tolerance and (ema_20 - ema_50) > gap_tolerance:
            stack_state = "BULLISH"
        elif (ema_20 - ema_9) > gap_tolerance and (ema_50 - ema_20) > gap_tolerance:
            stack_state = "BEARISH"

    distance_from_ema_20 = _distance_from(last_price, ema_20)
    distance_from_ema_50 = _distance_from(last_price, ema_50)

    strength = _compute_dmi_strength(prices, window=14)
    if strength >= 50.0:
        strength_state = "STRONG"
    elif strength >= 25.0:
        strength_state = "MEDIUM"
    else:
        strength_state = "WEAK"

    return {
        "ema_9": ema_9,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "ma_stack_state": stack_state,
        "distance_from_ema_20": distance_from_ema_20,
        "distance_from_ema_50": distance_from_ema_50,
        "trend_strength": strength,
        "trend_strength_state": strength_state,
    }
