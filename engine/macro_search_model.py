"""
Macro-aware search modulation: adjusts search depth and aggressiveness based on macro pressure.
"""

from __future__ import annotations

from typing import Dict


def _clamp(value: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def compute_search_modulation(
    macro_pressure_score: float,
    next_event_time_delta: float,
) -> Dict[str, float]:
    pressure = _clamp(macro_pressure_score, 0.0, 1.0)
    minutes = float(next_event_time_delta or 0.0)

    search_depth_multiplier = 1.0
    aggressiveness_bias = 0.0

    if pressure >= 0.8 and minutes <= 60:
        search_depth_multiplier = 0.7
        aggressiveness_bias = -0.5
    elif pressure >= 0.5 and minutes <= 120:
        search_depth_multiplier = 0.85
        aggressiveness_bias = -0.25
    elif pressure <= 0.3:
        search_depth_multiplier = 1.1
        aggressiveness_bias = 0.2

    return {
        "search_depth_multiplier": _clamp(search_depth_multiplier, 0.5, 1.5),
        "aggressiveness_bias": _clamp(aggressiveness_bias, -1.0, 1.0),
    }
