"""Deterministic regime classification helpers for Stockfish-Trade.

These functions are pure, typed, and contain no side effects. They provide
lightweight regime labels that higher-level components can use for
regime-aware scoring and decisions.
"""

from math import sqrt
from typing import List


def classify_trend(ma_short: float, ma_long: float) -> str:
    """Classify trend as 'up', 'down', or 'chop'.

    Rules:
    - If ma_short > ma_long * 1.01 → 'up'
    - If ma_short < ma_long * 0.99 → 'down'
    - Else → 'chop'
    """

    if ma_short > ma_long * 1.01:
        return "up"
    if ma_short < ma_long * 0.99:
        return "down"
    return "chop"


def classify_volatility(volatility: float, recent_returns: List[float]) -> str:
    """Classify volatility as 'high' or 'low'.

    Rules:
    - Compute simple stddev of recent_returns (pure Python).
    - If volatility > 1.5 * stddev → 'high'
    - Else → 'low'
    """

    if not recent_returns:
        return "low"

    mean = sum(recent_returns) / len(recent_returns)
    variance = sum((r - mean) ** 2 for r in recent_returns) / len(recent_returns)
    stddev = sqrt(variance)

    if stddev == 0:
        return "low"

    return "high" if volatility > 1.5 * stddev else "low"


def classify_liquidity(liquidity: float) -> str:
    """Classify liquidity as 'high' or 'low'."""

    return "high" if liquidity > 0.6 else "low"


def classify_macro(momentum: float, rsi: float) -> str:
    """Classify macro regime as 'risk_on' or 'risk_off'."""

    if momentum > 0 and rsi > 50:
        return "risk_on"
    return "risk_off"


def trend_strength(ma_short: float, ma_long: float) -> float:
    """Return trend strength in [-1.0, 1.0] based on MA ratio."""

    if ma_long == 0:
        return 0.0
    ratio = ma_short / ma_long
    if ratio > 1.05:
        return min(1.0, ratio - 1.0)
    if ratio < 0.95:
        return max(-1.0, ratio - 1.0)
    return 0.0


def volatility_intensity(volatility: float, stddev: float) -> float:
    """Return volatility intensity in [0.0, 1.0]."""

    if stddev == 0:
        return 0.0
    return min(1.0, volatility / (stddev * 2))


def liquidity_penalty(liquidity: float) -> float:
    """Return liquidity penalty in [0.0, 0.3]."""

    if liquidity < 0.3:
        return 0.3
    if liquidity < 0.5:
        return 0.15
    return 0.0


def macro_bias(momentum: float, rsi: float) -> float:
    """Return macro bias in [-0.3, 0.3]."""

    if momentum > 0 and rsi > 60:
        return 0.3
    if momentum < 0 and rsi < 40:
        return -0.3
    return 0.0
