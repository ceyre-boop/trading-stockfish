"""Deterministic News & Macro Engine v1.

Provides scheduled event handling, expected volatility, liquidity withdrawal
flags, and macro regime classification. Designed to be replay/live parity safe
with no external I/O.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Deterministic configuration tables
EVENT_PRE_WINDOWS_MIN = {
    "FOMC": 120,
    "CPI": 90,
    "NFP": 60,
    "PMI": 45,
    "Earnings": 30,
    "FedSpeaker": 20,
}

EVENT_POST_WINDOWS_MIN = {
    "FOMC": 60,
    "CPI": 45,
    "NFP": 45,
    "PMI": 30,
    "Earnings": 30,
    "FedSpeaker": 20,
}

EVENT_VOL_STATE = {
    "FOMC": ("EXTREME", 1.0),
    "CPI": ("EXTREME", 0.95),
    "NFP": ("HIGH", 0.8),
    "PMI": ("MEDIUM", 0.6),
    "Earnings": ("MEDIUM", 0.55),
    "FedSpeaker": ("LOW", 0.35),
}

IMPACT_MULTIPLIER = {
    "LOW": 0.8,
    "MEDIUM": 1.0,
    "HIGH": 1.15,
}


@dataclass(frozen=True)
class UpcomingEvents:
    next_event_type: str = "NONE"
    next_event_time_delta: float = 0.0
    event_risk_window: str = "NONE"  # PRE_EVENT / POST_EVENT / NONE
    expected_volatility_state: str = "LOW"
    expected_volatility_score: float = 0.0
    liquidity_withdrawal_flag: bool = False
    macro_regime: str = "NEUTRAL"
    macro_regime_score: float = 0.0


def load_event_calendar(
    calendar_data: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Normalize and sort calendar entries.

    Each entry must contain event_type, timestamp, expected_impact.
    Unknown fields are ignored; invalid entries are dropped.
    """
    if not calendar_data:
        return []
    normalized: List[Dict[str, Any]] = []
    for entry in calendar_data:
        event_type = str(entry.get("event_type", "")).strip() or ""
        timestamp = entry.get("timestamp")
        impact = str(entry.get("expected_impact", "")).upper() or "MEDIUM"
        if not event_type or timestamp is None:
            continue
        normalized.append(
            {
                "event_type": event_type,
                "timestamp": float(timestamp),
                "expected_impact": impact if impact in IMPACT_MULTIPLIER else "MEDIUM",
            }
        )
    return sorted(normalized, key=lambda e: e["timestamp"])


def _event_risk_window(event_type: str, current_time: float, event_time: float) -> str:
    pre_window = EVENT_PRE_WINDOWS_MIN.get(event_type, 15) * 60.0
    post_window = EVENT_POST_WINDOWS_MIN.get(event_type, 15) * 60.0
    if event_time - pre_window <= current_time < event_time:
        return "PRE_EVENT"
    if event_time <= current_time <= event_time + post_window:
        return "POST_EVENT"
    return "NONE"


def _volatility_from_event(event_type: str, expected_impact: str) -> Tuple[str, float]:
    state, base_score = EVENT_VOL_STATE.get(event_type, ("LOW", 0.25))
    multiplier = IMPACT_MULTIPLIER.get(expected_impact, 1.0)
    score = max(0.0, min(1.0, base_score * multiplier))
    # Clamp state escalation for extreme events
    if state == "EXTREME" and score < 0.9:
        score = max(score, 0.9)
    return state, score


def get_upcoming_events(
    calendar: List[Dict[str, Any]], current_time: float, horizon_minutes: float
) -> Dict[str, Any]:
    """Return deterministic upcoming event summary within horizon_minutes."""
    horizon_seconds = max(horizon_minutes, 0) * 60.0
    default = {
        "next_event_type": "NONE",
        "next_event_time_delta": 0.0,
        "event_risk_window": "NONE",
        "expected_volatility_state": "LOW",
        "expected_volatility_score": 0.0,
        "liquidity_withdrawal_flag": False,
    }

    if not calendar:
        return default

    for evt in calendar:
        event_time = evt["timestamp"]
        risk_window = _event_risk_window(evt["event_type"], current_time, event_time)
        in_future_horizon = current_time <= event_time <= current_time + horizon_seconds
        in_post_window = (
            event_time
            < current_time
            <= event_time + (EVENT_POST_WINDOWS_MIN.get(evt["event_type"], 15) * 60.0)
        )
        if in_future_horizon or in_post_window:
            time_delta_min = max(0.0, (event_time - current_time) / 60.0)
            vol_state, vol_score = _volatility_from_event(
                evt["event_type"], evt["expected_impact"]
            )
            withdrawal = risk_window == "PRE_EVENT"
            return {
                "next_event_type": evt["event_type"],
                "next_event_time_delta": time_delta_min,
                "event_risk_window": risk_window,
                "expected_volatility_state": vol_state,
                "expected_volatility_score": vol_score,
                "liquidity_withdrawal_flag": withdrawal,
            }

    # Fallback to first future event even if outside horizon
    next_event = next((e for e in calendar if e["timestamp"] >= current_time), None)
    if next_event:
        time_delta_min = (next_event["timestamp"] - current_time) / 60.0
        return {
            "next_event_type": next_event["event_type"],
            "next_event_time_delta": time_delta_min,
            "event_risk_window": "NONE",
            "expected_volatility_state": "LOW",
            "expected_volatility_score": 0.0,
            "liquidity_withdrawal_flag": False,
        }

    return default


def classify_macro_regime(macro_inputs: Optional[Dict[str, Any]]) -> Tuple[str, float]:
    """Classify deterministic macro regime using VIX/DXY/yields/SPX trend."""
    if not macro_inputs:
        return "NEUTRAL", 0.0
    vix = float(macro_inputs.get("vix", 0.0) or 0.0)
    dxy = float(macro_inputs.get("dxy", 0.0) or 0.0)
    us10y = float(macro_inputs.get("us10y", 0.0) or 0.0)
    spx_trend = float(macro_inputs.get("spx_trend", 0.0) or 0.0)

    risk_on_conditions = vix < 15.0 and dxy <= 103.0 and us10y <= 3.75
    risk_off_conditions = vix > 25.0 and dxy >= 105.0 and us10y >= 4.0

    if risk_on_conditions and spx_trend >= 0:
        # Favor higher score when SPX trend aligns
        score = min(1.0, 0.6 + 0.2 * max(0.0, min(1.0, spx_trend)))
        return "RISK_ON", score
    if risk_off_conditions and spx_trend <= 0:
        score = min(1.0, 0.6 + 0.2 * max(0.0, min(1.0, abs(spx_trend))))
        return "RISK_OFF", score
    return "NEUTRAL", 0.3 if 15.0 <= vix <= 25.0 else 0.0


def compute_news_macro_features(
    current_time: float,
    calendar_data: Optional[List[Dict[str, Any]]] = None,
    macro_inputs: Optional[Dict[str, Any]] = None,
    horizon_minutes: float = 240.0,
) -> Dict[str, Any]:
    """Main entrypoint for deterministic news & macro features."""
    calendar = load_event_calendar(calendar_data)
    event_snapshot = get_upcoming_events(calendar, current_time, horizon_minutes)
    macro_regime, macro_score = classify_macro_regime(macro_inputs)
    event_snapshot["macro_regime"] = macro_regime
    event_snapshot["macro_regime_score"] = macro_score
    return event_snapshot
