"""
Canonical stack validator for Stockfish-Trade.

Encodes enforceable invariants for official/tournament/live modes to prevent
architectural drift. This module is intentionally deterministic and uses
process-level guards, not heuristics.
"""

import os
from typing import Any

OFFICIAL_ENV = "OFFICIAL_MODE"
CANONICAL_ENV = "CANONICAL_STACK_ONLY"


def _is_truthy(val: str) -> bool:
    return str(val).lower() in {"1", "true", "yes", "on"}


def enforce_official_env(flag: bool) -> None:
    """Mark the process as official/canonical when flag is True."""
    if flag:
        os.environ[OFFICIAL_ENV] = "1"
        os.environ[CANONICAL_ENV] = "1"


def canonical_enforced() -> bool:
    return _is_truthy(os.environ.get(OFFICIAL_ENV, "")) or _is_truthy(
        os.environ.get(CANONICAL_ENV, "")
    )


def assert_causal_required(use_causal: bool) -> None:
    """Ensure the canonical causal evaluator is used when enforcement is active."""
    if canonical_enforced() and not use_causal:
        raise ValueError(
            "Canonical stack required: causal evaluator + policy only in official/tournament/live modes."
        )


def assert_ml_advisory_only(advisory_adjustment_applied: bool) -> None:
    """Ensure ML hints remain advisory and never drive decisions."""
    if advisory_adjustment_applied and canonical_enforced():
        raise ValueError(
            "ML hints may not alter eval/policy when canonical enforcement is active."
        )


def assert_governance_gate(governance_checked: bool) -> None:
    """Governance/safety must be the sole legality gate for actions."""
    if canonical_enforced() and not governance_checked:
        raise ValueError("Governance/safety gate must run in canonical modes.")


def assert_deterministic_mode(random_used: bool = False) -> None:
    if canonical_enforced() and random_used:
        raise ValueError("Randomness is forbidden in canonical modes.")


def validate_mode_is_canonical(
    use_causal: bool, governance_checked: bool = True
) -> None:
    """Aggregate check for canonical stack invariants."""
    assert_causal_required(use_causal)
    assert_governance_gate(governance_checked)


def forbid_module(module_name: str) -> None:
    if canonical_enforced():
        raise ValueError(f"Module '{module_name}' is not permitted in canonical modes.")
