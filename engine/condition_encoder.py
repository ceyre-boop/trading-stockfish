import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from engine.types import MarketState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionVector:
    session: str
    macro: str
    vol: str
    trend: str
    liquidity: str
    tod: str


def _upper_safe(value: Any, default: str = "UNKNOWN") -> str:
    try:
        return str(value).upper() if value is not None else default
    except Exception:
        return default


def encode_session(state: MarketState) -> str:
    session = (
        getattr(state, "session", None) or state.raw.get("session_regime")
        if isinstance(state.raw, dict)
        else None
    )
    encoded = _upper_safe(session, default="UNKNOWN")
    logger.debug("condition_encoder.session=%s", encoded)
    return encoded


def encode_macro(state: MarketState) -> str:
    macro = (
        getattr(state, "macro_regime", None) or state.raw.get("macro_regime")
        if isinstance(state.raw, dict)
        else None
    )
    encoded = _upper_safe(macro, default="NEUTRAL")
    logger.debug("condition_encoder.macro=%s", encoded)
    return encoded


def encode_volatility(state: MarketState) -> str:
    vol = (
        getattr(state, "volatility_regime", None) or state.raw.get("volatility_regime")
        if isinstance(state.raw, dict)
        else None
    )
    encoded = _upper_safe(vol, default="NORMAL")
    logger.debug("condition_encoder.vol=%s", encoded)
    return encoded


def encode_trend(state: MarketState) -> str:
    trend = getattr(state, "trend_direction", None) or getattr(
        state, "trend_regime", None
    )
    encoded = _upper_safe(trend, default="FLAT")
    logger.debug("condition_encoder.trend=%s", encoded)
    return encoded


def encode_liquidity(state: MarketState) -> str:
    liquidity = (
        getattr(state, "liquidity_regime", None) or state.raw.get("liquidity_regime")
        if isinstance(state.raw, dict)
        else None
    )
    encoded = _upper_safe(liquidity, default="NORMAL")
    logger.debug("condition_encoder.liquidity=%s", encoded)
    return encoded


def encode_time_of_day(state: MarketState) -> str:
    # Prefer explicit bucket on state.raw, else derive from session marker as coarse proxy.
    tod = None
    if isinstance(state.raw, dict):
        tod = state.raw.get("time_of_day_bucket") or state.raw.get("session_bucket")
    if tod is None:
        session = getattr(state, "session", "")
        session_upper = _upper_safe(session, default="UNKNOWN")
        # Simple mapping; deterministic and static.
        if session_upper in {"RTH_OPEN", "OPEN"}:
            tod = "OPEN"
        elif session_upper in {"MIDDAY", "LUNCH"}:
            tod = "MIDDAY"
        elif session_upper in {"POWER_HOUR", "AFTERNOON"}:
            tod = "AFTERNOON"
        elif session_upper in {"CLOSE"}:
            tod = "CLOSE"
        else:
            tod = "UNKNOWN"
    encoded = _upper_safe(tod, default="UNKNOWN")
    logger.debug("condition_encoder.tod=%s", encoded)
    return encoded


def encode_conditions(state: MarketState) -> ConditionVector:
    """Deterministic mapping from MarketState to ConditionVector.

    No side effects, no runtime behavior changes. Logging is debug-level only.
    """

    session = encode_session(state)
    macro = encode_macro(state)
    vol = encode_volatility(state)
    trend = encode_trend(state)
    liquidity = encode_liquidity(state)
    tod = encode_time_of_day(state)

    cv = ConditionVector(
        session=session,
        macro=macro,
        vol=vol,
        trend=trend,
        liquidity=liquidity,
        tod=tod,
    )
    logger.debug("condition_encoder.vector=%s", cv)
    return cv
