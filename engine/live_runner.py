"""Live runner policy loader with safety awareness.

Deterministically selects which policy variant to use at runtime based on
SafetyState. If SAFE_MODE is active, the returned policy includes any
safe-mode adjustments already embedded by the feedback loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from engine.policy_loader import PolicyConfig, load_policy
from engine.safety_mode import SafetyState


def load_policy_for_state(
    policy_path: Path, safety_state: Optional[SafetyState] = None
) -> Optional[PolicyConfig]:
    """Load the correct policy variant based on safety state.

    If SAFE_MODE, we still load the persisted policy (which the feedback loop
    already dampened or reverted) so behavior stays deterministic. The
    SafetyState is used only to decide which path to read when multiple are
    provided; here we use the active path for both states to keep a single
    source of truth.
    """
    state = safety_state.current_state if safety_state else "NORMAL"
    # Single policy source of truth; SAFE_MODE adjustments are persisted by feedback loop
    return load_policy(policy_path)


def get_active_policy_path(
    active_path: Path, safe_path: Optional[Path], safety_state: Optional[SafetyState]
) -> Path:
    """Return the path to load based on current safety state.

    If SAFE_MODE and a separate safe_path is provided, use it; otherwise fall
    back to the active_path. This function is deterministic and side-effect free.
    """
    if safety_state and safety_state.current_state == "SAFE_MODE" and safe_path:
        return safe_path
    return active_path
