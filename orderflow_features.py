from typing import Dict, List, Optional


def _pad_series(values: Optional[List[float]], length: int) -> List[float]:
    base = list(values or [])
    if len(base) < length:
        base.extend([0.0] * (length - len(base)))
    return [float(v) for v in base[:length]]


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


def _sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def compute_orderflow_features(
    prices: List[float],
    aggressive_buy: Optional[List[float]] = None,
    aggressive_sell: Optional[List[float]] = None,
) -> Dict[str, float | bool | str]:
    if not prices:
        return {
            "bar_delta": 0.0,
            "cumulative_delta": 0.0,
            "footprint_imbalance": 0.0,
            "has_absorption": False,
            "absorption_side": "NONE",
            "has_exhaustion": False,
            "exhaustion_side": "NONE",
        }

    n = len(prices)
    buy_series = _pad_series(aggressive_buy, n)
    sell_series = _pad_series(aggressive_sell, n)
    delta_series = [b - s for b, s in zip(buy_series, sell_series)]
    total_series = [max(0.0, b) + max(0.0, s) for b, s in zip(buy_series, sell_series)]

    bar_delta = delta_series[-1]
    cumulative_delta = sum(delta_series)

    bar_total = total_series[-1]
    eps = 1e-6
    if bar_total > eps:
        footprint_imbalance = _clamp(bar_delta / (bar_total + eps), -1.0, 1.0)
    else:
        price_change = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
        footprint_imbalance = _sign(price_change)

    window = min(10, n)
    recent_totals = total_series[-window:]
    avg_total = sum(recent_totals) / max(1, len(recent_totals))

    price_move = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
    move_threshold = 0.0005 * abs(prices[-1]) if prices[-1] else 0.0
    has_absorption = (
        bar_total > 0.0
        and bar_total >= avg_total * 1.5
        and abs(price_move) <= max(move_threshold, 1e-6)
    )
    absorption_side = "NONE"
    if has_absorption:
        if bar_delta > 0:
            absorption_side = "BUY"
        elif bar_delta < 0:
            absorption_side = "SELL"

    has_exhaustion = False
    exhaustion_side = "NONE"
    if n >= 3:
        prev_move = prices[-2] - prices[-3]
        curr_move = prices[-1] - prices[-2]
        prev_total = total_series[-2]
        curr_total = bar_total
        move_prev_threshold = 0.001 * abs(prices[-2]) if prices[-2] else 0.0
        large_prev_move = abs(prev_move) >= max(move_prev_threshold, 1e-6)
        volume_heavy = prev_total >= avg_total * 1.5 and prev_total > 0.0
        stalled = abs(curr_move) <= abs(prev_move) * 0.25 or _sign(curr_move) != _sign(
            prev_move
        )
        volume_drop = curr_total < prev_total * 0.6 and curr_total >= 0.0
        if large_prev_move and volume_heavy and stalled and volume_drop:
            has_exhaustion = True
            exhaustion_side = "BUY" if prev_move > 0 else "SELL"

    return {
        "bar_delta": float(bar_delta),
        "cumulative_delta": float(cumulative_delta),
        "footprint_imbalance": float(footprint_imbalance),
        "has_absorption": has_absorption,
        "absorption_side": absorption_side,
        "has_exhaustion": has_exhaustion,
        "exhaustion_side": exhaustion_side,
    }
