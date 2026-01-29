"""
Macro risk model: maps macro pressure and event proximity into entry gating and sizing.
"""

from __future__ import annotations

from typing import Dict


def _clamp(value: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def compute_macro_risk_adjustment(
    macro_pressure_score: float,
    next_event_time_delta: float,
    next_event_impact: str,
) -> Dict[str, float | bool]:
    """Return risk controls driven by macro pressure and event timing."""
    pressure = _clamp(macro_pressure_score, 0.0, 1.0)
    minutes = float(next_event_time_delta or 0.0)
    impact = (next_event_impact or "MEDIUM").upper()

    entry_allowed = True
    position_size_multiplier = 1.0
    max_leverage_multiplier = 1.0

    # High pressure and imminent event: gate entries and slash size.
    if pressure >= 0.8 and minutes <= 60:
        entry_allowed = False
        position_size_multiplier = 0.25
        max_leverage_multiplier = 0.5
    # Medium pressure, close event: reduce risk but allow controlled entries.
    elif pressure >= 0.5 and minutes <= 30:
        entry_allowed = True
        position_size_multiplier = 0.5
        max_leverage_multiplier = 0.75
    # Low pressure: full size.
    else:
        entry_allowed = True
        position_size_multiplier = 1.0
        max_leverage_multiplier = 1.0

    # High impact events tighten risk even when slightly further out.
    if impact == "HIGH" and pressure >= 0.6 and minutes <= 120:
        position_size_multiplier = min(position_size_multiplier, 0.5)
        max_leverage_multiplier = min(max_leverage_multiplier, 0.75)

    return {
        "position_size_multiplier": _clamp(position_size_multiplier, 0.0, 1.0),
        "entry_allowed": bool(entry_allowed),
        "max_leverage_multiplier": _clamp(max_leverage_multiplier, 0.0, 1.0),
    }
