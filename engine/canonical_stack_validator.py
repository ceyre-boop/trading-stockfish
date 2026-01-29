"""
Canonical stack validator for Stockfish-Trade.

Runs at startup (invoked by official-mode components) and in CI tests to ensure
no architectural drift from the Stockfish mapping.
"""

from typing import Any

from .canonical_validator import canonical_enforced


def validate_canonical_stack(
    *,
    context: str,
    use_causal: bool,
    legacy_path_detected: bool = False,
    governance_checked: bool = True,
    ml_influence: bool = False,
    cockpit_influence: bool = False,
    random_used: bool = False,
    time_causal: bool = True,
    regime_source: str = "regime_engine",
    allowed_regime_source: str = "regime_engine",
    amd_checks: bool = True,
    amd_tag: str = "NEUTRAL",
    amd_regime_present: bool = True,
    manipulation_veto_active: bool = True,
    volatility_shock_present: bool = True,
    volatility_shock_deterministic: bool = True,
    governance_vol_shock_veto_active: bool = True,
    swing_structure_present: bool = True,
    trend_structure_deterministic: bool = True,
) -> None:
    """Validate core invariants when canonical enforcement is active.

    Raises ValueError with descriptive messages on any violation.
    """
    if not canonical_enforced():
        return

    # Always enforce AMD presence when canonical enforcement is active
    amd_checks = True

    if not use_causal:
        raise ValueError(
            f"[{context}] Canonical evaluator required: only causal evaluator is permitted in official/tournament/live modes."
        )
    if legacy_path_detected:
        raise ValueError(
            f"[{context}] Legacy/alternate evaluator path detected; canonical stack forbids legacy evaluators in official/tournament/live modes."
        )
    if not governance_checked:
        raise ValueError(
            f"[{context}] Governance/safety gate is mandatory; action flow cannot bypass governance_engine/safety_layer."
        )
    if ml_influence:
        raise ValueError(
            f"[{context}] ML hints must be advisory-only; detected influence on core decisions/confidence."
        )
    if cockpit_influence:
        raise ValueError(
            f"[{context}] Cockpit/monitoring must be read-only; decision influence is forbidden."
        )
    if random_used:
        raise ValueError(
            f"[{context}] Randomness detected; canonical modes require deterministic operation."
        )
    if not time_causal:
        raise ValueError(
            f"[{context}] Time-causality violated; official modes require strictly causal data flow."
        )
    if regime_source != allowed_regime_source:
        raise ValueError(
            f"[{context}] Regime source mismatch: expected '{allowed_regime_source}', got '{regime_source}'."
        )
    if amd_checks:
        if not amd_regime_present:
            raise ValueError(
                f"[{context}] AMD regime missing from canonical MarketState/Regime bundle."
            )
        if amd_tag is None:
            raise ValueError(f"[{context}] AMD tag missing while amd_checks enabled.")
        if amd_tag == "MANIPULATION" and not manipulation_veto_active:
            raise ValueError(
                f"[{context}] AMD manipulation detected but veto not active."
            )
    if not volatility_shock_present:
        raise ValueError(
            f"[{context}] Volatility shock flag missing from canonical MarketState."
        )
    if not volatility_shock_deterministic:
        raise ValueError(
            f"[{context}] Volatility shock detection must be deterministic in canonical modes."
        )
    if not swing_structure_present:
        raise ValueError(
            f"[{context}] Swing/trend structure missing from canonical MarketState."
        )
    if not trend_structure_deterministic:
        raise ValueError(
            f"[{context}] Swing/trend structure must be deterministic in canonical modes."
        )
    if not governance_vol_shock_veto_active:
        raise ValueError(
            f"[{context}] Volatility shock governance veto inactive in official/tournament/live mode."
        )


def validate_official_mode_startup(context: str, *, use_causal: bool) -> None:
    """Convenience wrapper for components entering official mode."""
    validate_canonical_stack(
        context=context,
        use_causal=use_causal,
        legacy_path_detected=False,
        governance_checked=True,
        ml_influence=False,
        cockpit_influence=False,
        random_used=False,
        time_causal=True,
        regime_source="regime_engine",
    )
