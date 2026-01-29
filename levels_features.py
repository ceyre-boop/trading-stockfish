from typing import List, Optional


def _safe_len_match(arr: List, n: int):
    if len(arr) < n:
        arr.extend([arr[-1] if arr else 0] * (n - len(arr)))
    return arr


def compute_level_features(
    prices: List[float],
    timestamps: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
    session_labels: Optional[List[str]] = None,
    default_session: str = "UNKNOWN",
):
    if not prices:
        return {
            "session_high": 0.0,
            "session_low": 0.0,
            "day_high": 0.0,
            "day_low": 0.0,
            "previous_day_high": 0.0,
            "previous_day_low": 0.0,
            "previous_day_close": 0.0,
            "vwap_price": 0.0,
            "distance_from_vwap": 0.0,
        }

    n = len(prices)
    timestamps = timestamps or list(range(n))
    volumes = volumes or [1.0] * n
    session_labels = session_labels or [default_session] * n
    _safe_len_match(timestamps, n)
    _safe_len_match(volumes, n)
    _safe_len_match(session_labels, n)

    # Day grouping
    day_keys = [int(ts // 86400) for ts in timestamps]
    last_day = day_keys[-1]
    days = {}
    for idx, day in enumerate(day_keys):
        days.setdefault(day, []).append(idx)

    current_day_indices = days.get(last_day, list(range(n)))
    day_prices = [prices[i] for i in current_day_indices]
    day_high = max(day_prices)
    day_low = min(day_prices)

    sorted_days = sorted(days.keys())
    previous_day_high = 0.0
    previous_day_low = 0.0
    previous_day_close = 0.0
    if len(sorted_days) > 1:
        prev_day_key = sorted_days[-2]
        prev_indices = days[prev_day_key]
        prev_prices = [prices[i] for i in prev_indices]
        previous_day_high = max(prev_prices)
        previous_day_low = min(prev_prices)
        previous_day_close = prices[prev_indices[-1]]

    # Session highs/lows for the last session label
    last_session = session_labels[-1] if session_labels else default_session
    session_indices = [i for i, s in enumerate(session_labels) if s == last_session]
    if session_indices:
        session_prices = [prices[i] for i in session_indices]
        session_high = max(session_prices)
        session_low = min(session_prices)
    else:
        session_high = day_high
        session_low = day_low

    # VWAP over current day
    day_volumes = [volumes[i] for i in current_day_indices]
    vwap_num = sum(p * v for p, v in zip(day_prices, day_volumes))
    vwap_den = sum(day_volumes)
    vwap_price = vwap_num / vwap_den if vwap_den else 0.0
    last_price = prices[-1]
    distance_from_vwap = (last_price - vwap_price) / vwap_price if vwap_price else 0.0

    return {
        "session_high": session_high,
        "session_low": session_low,
        "day_high": day_high,
        "day_low": day_low,
        "previous_day_high": previous_day_high,
        "previous_day_low": previous_day_low,
        "previous_day_close": previous_day_close,
        "vwap_price": vwap_price,
        "distance_from_vwap": distance_from_vwap,
    }
