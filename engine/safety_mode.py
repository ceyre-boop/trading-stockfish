"""Safety mode controller.

Deterministic SAFE_MODE entry/exit based on drift, instability, and repeated
gate failures. Provides policy adjustments (revert or dampen) without any
randomness. Identical inputs yield identical outputs.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass
class SafetyConfig:
    drift_feature_threshold: int = 1
    gate_fail_window: int = 3
    gate_fail_threshold: int = 2
    multiplier_dampen: float = 0.5
    stability_variance_threshold: float = 1.0e9
    stability_score_min: float = 0.0
    manual_override: Optional[str] = None  # "SAFE_MODE" or "NORMAL"


@dataclass
class SafetyState:
    current_state: str = "NORMAL"
    last_good_policy_path: Optional[str] = None
    last_good_policy_version: Optional[str] = None
    timestamp_utc: str = field(default_factory=_utc_now)
    mode: Optional[str] = None  # legacy alias

    def __post_init__(self) -> None:
        if self.mode:
            self.current_state = self.mode


@dataclass
class SafetyDecision:
    new_state: str = "NORMAL"
    reason: str = ""
    timestamp_utc: str = field(default_factory=_utc_now)
    action: str = "NONE"
    fallback_policy: Optional[Dict[str, Any]] = None
    multiplier_scale: float = 1.0
    reasons: List[str] = field(default_factory=list)
    mode: Optional[str] = None  # legacy alias
    multiplier_dampen: Optional[float] = None  # legacy alias

    def __post_init__(self) -> None:
        if self.mode:
            self.new_state = self.mode
        if self.multiplier_dampen is not None:
            self.multiplier_scale = float(self.multiplier_dampen)
        if not self.reason:
            self.reason = self.reasons[0] if self.reasons else "stable"
        # keep mode attribute aligned
        self.mode = self.new_state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "new_state": self.new_state,
            "reason": self.reason,
            "timestamp_utc": self.timestamp_utc,
            "action": self.action,
            "multiplier_scale": self.multiplier_scale,
            "reasons": list(self.reasons) if self.reasons else [self.reason],
            "mode": self.new_state,
        }


def _count_recent_failures(history: List[str], window: int) -> int:
    recent = history[-window:]
    return sum(1 for h in recent if str(h).upper() == "FAIL")


def _variance_from_stability(stability: Dict[str, Any]) -> Optional[float]:
    for key in ("variance", "return_variance"):
        if key in stability:
            try:
                return float(stability[key])
            except Exception:
                return None
    return None


def _score_from_stability(stability: Dict[str, Any]) -> Optional[float]:
    for key in ("stability", "stability_score"):
        if key in stability:
            try:
                return float(stability[key])
            except Exception:
                return None
    return None


def check_and_update_safety(
    drift_result: Dict[str, Any],
    gating_history: List[str],
    config: Optional[SafetyConfig] = None,
    state: Optional[SafetyState] = None,
    stability_metrics: Optional[Dict[str, Any]] = None,
) -> SafetyDecision:
    cfg = config or SafetyConfig()
    current_state = state or SafetyState()
    reasons: List[str] = []

    # Manual override takes precedence
    if cfg.manual_override in {"SAFE_MODE", "NORMAL"}:
        target = cfg.manual_override
        reason = f"manual_override_{target}"
        ts = _utc_now()
        return SafetyDecision(
            new_state=target,
            reason=reason,
            reasons=[reason],
            timestamp_utc=ts,
            action="RESTORE" if target == "NORMAL" else "DAMPEN_MULTIPLIERS",
            multiplier_scale=cfg.multiplier_dampen,
        )

    features_flagged = 0
    try:
        features_flagged = int(
            (drift_result or {}).get("aggregates", {}).get("features_flagged", 0)
        )
    except Exception:
        features_flagged = 0

    stability = stability_metrics or (drift_result or {}).get("stability", {}) or {}
    variance_val = _variance_from_stability(stability)
    stability_score = _score_from_stability(stability)

    gate_failures = _count_recent_failures(gating_history, cfg.gate_fail_window)

    drift_breach = features_flagged >= cfg.drift_feature_threshold
    instability_breach = False
    if variance_val is not None and variance_val > cfg.stability_variance_threshold:
        instability_breach = True
        reasons.append(
            f"variance_exceeded:{variance_val:.6f}>{cfg.stability_variance_threshold:.6f}"
        )
    if stability_score is not None and stability_score < cfg.stability_score_min:
        instability_breach = True
        reasons.append(
            f"stability_score_below_min:{stability_score:.6f}<{cfg.stability_score_min:.6f}"
        )
    gate_breach = gate_failures >= cfg.gate_fail_threshold

    enter_safe = False
    exit_safe = False

    if current_state.current_state != "SAFE_MODE":
        if drift_breach:
            reasons.append(
                f"drift_features_flagged={features_flagged}>={cfg.drift_feature_threshold}"
            )
        if gate_breach:
            reasons.append(f"gate_failures={gate_failures}>={cfg.gate_fail_threshold}")
        if drift_breach or gate_breach or instability_breach:
            enter_safe = True
    else:
        if not drift_breach and gate_failures == 0 and not instability_breach:
            exit_safe = True
            reasons.append("drift_and_gate_stable")

    ts = _utc_now()

    if enter_safe:
        current_state.current_state = "SAFE_MODE"
        reason_str = ";".join(reasons) if reasons else "enter_safe_mode"
        return SafetyDecision(
            new_state="SAFE_MODE",
            reason=reason_str,
            reasons=reasons or [reason_str],
            timestamp_utc=ts,
            action="DAMPEN_MULTIPLIERS",
            multiplier_scale=cfg.multiplier_dampen,
        )

    if exit_safe:
        current_state.current_state = "NORMAL"
        reason_str = ";".join(reasons) if reasons else "exit_safe_mode"
        return SafetyDecision(
            new_state="NORMAL",
            reason=reason_str,
            reasons=reasons or [reason_str],
            timestamp_utc=ts,
            action="RESTORE",
            multiplier_scale=1.0,
        )

    reason_str = ";".join(reasons) if reasons else "stable"
    return SafetyDecision(
        new_state=current_state.current_state,
        reason=reason_str,
        reasons=reasons or [reason_str],
        timestamp_utc=ts,
        action="NONE",
        multiplier_scale=1.0,
    )


def apply_safety_decision(
    policy: Dict[str, Any], decision: SafetyDecision
) -> Dict[str, Any]:
    if decision.new_state != "SAFE_MODE":
        return policy

    # Prefer explicit fallback policy when provided
    if decision.fallback_policy is not None:
        safe_policy = copy.deepcopy(decision.fallback_policy)
    else:
        safe_policy = copy.deepcopy(policy)
        damp = decision.multiplier_scale or 1.0
        regime_mult = safe_policy.get("regime_multipliers", {})
        if isinstance(regime_mult, dict):
            for regime, fmap in regime_mult.items():
                if not isinstance(fmap, dict):
                    continue
                for feat, val in list(fmap.items()):
                    try:
                        fmap[feat] = float(val) * damp
                    except Exception:
                        fmap[feat] = 0.0
        safe_policy["regime_multipliers"] = regime_mult
        # Risk scale clamp
        existing_risk = safe_policy.get("risk_scale", 1.0)
        try:
            existing_risk = float(existing_risk)
        except Exception:
            existing_risk = 1.0
        safe_policy["risk_scale"] = min(existing_risk, damp)

    safety_meta = {
        "state": decision.new_state,
        "reason": decision.reason,
        "timestamp_utc": decision.timestamp_utc,
        "multiplier_scale": decision.multiplier_scale,
    }
    safe_policy["safety_mode"] = safety_meta
    return safe_policy
