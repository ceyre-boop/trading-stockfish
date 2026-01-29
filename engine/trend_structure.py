"""
Deterministic swing structure detector (HH/HL/LH/LL) with volatility-aware thresholds.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_trend_structure(
    prices: Iterable[float],
    highs: Optional[Iterable[float]] = None,
    lows: Optional[Iterable[float]] = None,
    window: int = 20,
    volatility_state: Optional[Dict] = None,
) -> Dict[str, float | str]:
    """Compute swing structure deterministically using rolling windows.

    Args:
        prices: iterable of mid/close prices (most recent last)
        highs: optional iterable of highs aligned with prices
        lows: optional iterable of lows aligned with prices
        window: lookback window for swing detection
        volatility_state: optional dict with realized_vol / intraday_band_width

    Returns:
        Dict with swing_high, swing_low, swing_structure, trend_direction, trend_strength
    """
    prices_arr = np.array(list(prices), dtype=float)
    if prices_arr.size < 5:
        return _neutral_structure()

    highs_arr = (
        np.array(list(highs), dtype=float) if highs is not None else prices_arr.copy()
    )
    lows_arr = (
        np.array(list(lows), dtype=float) if lows is not None else prices_arr.copy()
    )

    # Use rolling window; if insufficient history, shrink window deterministically
    win = min(window, prices_arr.size)
    recent_slice = slice(-win, None)
    prev_slice = (
        slice(-(2 * win), -win) if prices_arr.size >= 2 * win else slice(0, -win)
    )

    recent_high = float(np.max(highs_arr[recent_slice]))
    recent_low = float(np.min(lows_arr[recent_slice]))

    if prices_arr.size - win <= 0:
        prev_high = recent_high
        prev_low = recent_low
    else:
        prev_high = float(np.max(highs_arr[prev_slice]))
        prev_low = float(np.min(lows_arr[prev_slice]))

    realized_vol = 0.0
    band_width = 0.0
    if volatility_state:
        realized_vol = float(volatility_state.get("realized_vol", 0.0) or 0.0)
        band_width = float(volatility_state.get("intraday_band_width", 0.0) or 0.0)
    # Volatility-aware threshold (deterministic floor to avoid zero)
    thresh = max(0.001, realized_vol * 0.6 + band_width * 0.5)

    swing_structure = "NEUTRAL"
    if recent_high > prev_high * (1.0 + thresh):
        swing_structure = "HH"
    elif recent_low > prev_low * (1.0 + thresh):
        swing_structure = "HL"
    elif recent_high < prev_high * (1.0 - thresh):
        swing_structure = "LH"
    elif recent_low < prev_low * (1.0 - thresh):
        swing_structure = "LL"

    if swing_structure in {"HH", "HL"}:
        trend_direction = "UP"
    elif swing_structure in {"LH", "LL"}:
        trend_direction = "DOWN"
    else:
        trend_direction = "RANGE"

    span = max(recent_high - recent_low, 0.0)
    denom = max(prev_high + 1e-8, recent_low + 1e-8)
    raw_strength = span / denom
    vol_scale = max(0.2, min(1.0, 1.0 - realized_vol))
    trend_strength = _clamp(raw_strength * vol_scale * 2.0, 0.0, 1.0)

    return {
        "swing_high": recent_high,
        "swing_low": recent_low,
        "swing_structure": swing_structure,
        "trend_direction": trend_direction,
        "trend_strength": trend_strength,
    }


def _neutral_structure() -> Dict[str, float | str]:
    return {
        "swing_high": 0.0,
        "swing_low": 0.0,
        "swing_structure": "NEUTRAL",
        "trend_direction": "RANGE",
        "trend_strength": 0.0,
    }
