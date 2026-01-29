"""Deterministic regime modifiers used for evaluator and policy adjustments."""


def volatility_modifier(vol_regime: str) -> float:
    if vol_regime == "LOW":
        return 1.2
    if vol_regime == "NORMAL":
        return 1.0
    if vol_regime == "HIGH":
        return 0.7
    if vol_regime == "EXTREME":
        return 0.4
    return 1.0


def liquidity_modifier(liq_regime: str) -> float:
    if liq_regime in ("ROBUST", "DEEP"):
        return 1.1
    if liq_regime == "NORMAL":
        return 1.0
    if liq_regime in ("THIN", "FRAGILE"):
        return 0.6
    return 1.0


def trend_modifier(trend_regime: str) -> float:
    if trend_regime == "UP" or trend_regime == "DOWN":
        return 1.05
    if trend_regime == "SIDEWAYS":
        return 0.9
    return 1.0
