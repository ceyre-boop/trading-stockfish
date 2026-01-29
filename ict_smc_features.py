from typing import Dict, List, Optional

from structure_features import (
    SWING_HIGH,
    SWING_LOW,
    compute_structure_features,
    detect_swings,
)


def _candle_bounds(prices: List[float], idx: int) -> tuple[float, float]:
    if idx <= 0:
        return prices[idx], prices[idx]
    return min(prices[idx], prices[idx - 1]), max(prices[idx], prices[idx - 1])


def _killzone_flags(timestamp: Optional[float]) -> Dict[str, bool]:
    if timestamp is None:
        return {"in_london_killzone": False, "in_ny_killzone": False}
    import datetime as _dt

    hour = _dt.datetime.fromtimestamp(timestamp, tz=_dt.timezone.utc).hour
    return {
        "in_london_killzone": 7 <= hour < 10,
        "in_ny_killzone": 13 <= hour < 16,
    }


def compute_ict_smc_features(
    prices: List[float],
    timestamps: Optional[List[float]] = None,
) -> Dict[str, float | bool | str]:
    if not prices:
        base = _killzone_flags(None)
        base.update(
            {
                "current_bullish_ob_low": 0.0,
                "current_bullish_ob_high": 0.0,
                "current_bearish_ob_low": 0.0,
                "current_bearish_ob_high": 0.0,
                "last_touched_ob_type": "NONE",
                "has_mitigation": False,
                "has_flip_zone": False,
                "mitigation_low": 0.0,
                "mitigation_high": 0.0,
                "flip_low": 0.0,
                "flip_high": 0.0,
                "has_fvg": False,
                "fvg_upper": 0.0,
                "fvg_lower": 0.0,
                "has_ifvg": False,
                "ifvg_upper": 0.0,
                "ifvg_lower": 0.0,
                "premium_discount_state": "EQ",
                "equilibrium_level": 0.0,
            }
        )
        return base

    n = len(prices)
    timestamps = timestamps or list(range(n))
    swings = detect_swings(prices, lookback=2, lookforward=2)
    struct = compute_structure_features(prices)

    # Orderblocks
    prev_max = prices[0]
    prev_min = prices[0]
    bull_ob = (0.0, 0.0)
    bear_ob = (0.0, 0.0)
    bos_up_index = None
    bos_down_index = None
    for i in range(1, n):
        p = prices[i]
        if p > prev_max:
            bos_up_index = i
            # find last down candle before i
            for k in range(i - 1, 0, -1):
                if prices[k] < prices[k - 1]:
                    low, high = _candle_bounds(prices, k)
                    bull_ob = (low, high)
                    break
            prev_max = p
        if p < prev_min:
            bos_down_index = i
            for k in range(i - 1, 0, -1):
                if prices[k] > prices[k - 1]:
                    low, high = _candle_bounds(prices, k)
                    bear_ob = (low, high)
                    break
            prev_min = p
        prev_max = max(prev_max, p)
        prev_min = min(prev_min, p)

    # Mitigation / flip zones
    has_mitigation = False
    has_flip = False
    mitigation = (0.0, 0.0)
    flip = (0.0, 0.0)
    last_ob_touch = "NONE"
    last_price = prices[-1]
    if bull_ob != (0.0, 0.0):
        low, high = bull_ob
        if low <= last_price <= high:
            has_mitigation = True
            mitigation = bull_ob
            last_ob_touch = "BULLISH"
        if last_price < low:
            has_flip = True
            flip = bull_ob
    if bear_ob != (0.0, 0.0):
        low, high = bear_ob
        if low <= last_price <= high:
            has_mitigation = True
            mitigation = bear_ob
            last_ob_touch = "BEARISH"
        if last_price > high:
            has_flip = True
            flip = bear_ob

    # FVG detection
    has_fvg = False
    fvg_band = (0.0, 0.0)
    has_ifvg = False
    ifvg_band = (0.0, 0.0)
    for i in range(n - 2):
        lo0, hi0 = _candle_bounds(prices, i)
        lo2, hi2 = _candle_bounds(prices, i + 2)
        # bullish gap
        if lo0 > hi2:
            has_fvg = True
            fvg_band = (hi2, lo0)
            break
        # bearish gap
        if hi0 < lo2:
            has_fvg = True
            fvg_band = (hi0, lo2)
            break
    if has_fvg and n >= 6:
        for i in range(n - 4):
            lo0, hi0 = _candle_bounds(prices, i)
            lo2, hi2 = _candle_bounds(prices, i + 2)
            if hi0 < lo2:
                has_ifvg = True
                ifvg_band = (hi0, lo2)
                break
            if lo0 > hi2:
                has_ifvg = True
                ifvg_band = (hi2, lo0)
                break

    # Premium / discount using recent swing range
    recent_high = max(prices[-10:]) if len(prices) >= 10 else max(prices)
    recent_low = min(prices[-10:]) if len(prices) >= 10 else min(prices)
    equilibrium = (recent_high + recent_low) / 2.0
    tol = 0.001 * equilibrium if equilibrium else 0.0
    if abs(last_price - equilibrium) <= tol:
        pd_state = "EQ"
    elif last_price > equilibrium:
        pd_state = "PREMIUM"
    else:
        pd_state = "DISCOUNT"

    killzones = _killzone_flags(timestamps[-1] if timestamps else None)

    return {
        "current_bullish_ob_low": bull_ob[0],
        "current_bullish_ob_high": bull_ob[1],
        "current_bearish_ob_low": bear_ob[0],
        "current_bearish_ob_high": bear_ob[1],
        "last_touched_ob_type": last_ob_touch,
        "has_mitigation": has_mitigation,
        "has_flip_zone": has_flip,
        "mitigation_low": mitigation[0],
        "mitigation_high": mitigation[1],
        "flip_low": flip[0],
        "flip_high": flip[1],
        "has_fvg": has_fvg,
        "fvg_upper": max(fvg_band),
        "fvg_lower": min(fvg_band),
        "has_ifvg": has_ifvg,
        "ifvg_upper": max(ifvg_band),
        "ifvg_lower": min(ifvg_band),
        "premium_discount_state": pd_state,
        "equilibrium_level": equilibrium,
        "in_london_killzone": killzones["in_london_killzone"],
        "in_ny_killzone": killzones["in_ny_killzone"],
    }
