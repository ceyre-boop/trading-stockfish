"""
Macro Bayesian adjustment: tilts base priors using macro pressure and event impact.
"""

from __future__ import annotations

from typing import Dict


def _normalize(priors: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in priors.values()) or 1.0
    return {k: max(v, 0.0) / total for k, v in priors.items()}


def adjust_bayesian_priors(
    base_priors: Dict[str, float],
    macro_pressure_score: float,
    next_event_impact: str,
) -> Dict[str, float]:
    """Adjust priors to reflect regime instability during macro pressure."""
    priors = dict(base_priors)
    pressure = max(0.0, min(1.0, float(macro_pressure_score or 0.0)))
    impact = (next_event_impact or "MEDIUM").upper()

    # Higher pressure shifts probability toward breakout/volatility regimes and away from continuation.
    if pressure >= 0.7:
        priors["trend_continuation"] = max(
            priors.get("trend_continuation", 0.0) - 0.1, 0.0
        )
        priors["breakout"] = priors.get("breakout", 0.0) + 0.1
    elif pressure <= 0.3:
        # Low pressure keeps priors close to base.
        priors = priors
    else:
        priors["trend_continuation"] = max(
            priors.get("trend_continuation", 0.0) - 0.05, 0.0
        )
        priors["breakout"] = priors.get("breakout", 0.0) + 0.05

    # High-impact events slightly increase mean reversion risk post-release.
    if impact == "HIGH":
        priors["mean_reversion"] = priors.get("mean_reversion", 0.0) + 0.05

    return _normalize(priors)
