from __future__ import annotations

from typing import Any, Dict

from .decision_actions import ActionType


class InvariantViolation(Exception):
    pass


class InvariantChecker:
    def __init__(self) -> None:
        pass

    def assert_no_order_routing(
        self, intent: Dict[str, Any], mode_info: Dict[str, Any]
    ) -> None:
        allow = _mode_allows_routing(mode_info)
        if allow:
            return
        # routing_allowed flag should never be true in disallowed modes
        if intent.get("routing_allowed") is True:
            raise InvariantViolation("routing_allowed True in non-routing mode")
        # presence of any routing payloads is forbidden
        for key in ("order_request", "execution_request", "route_order"):
            if key in intent and intent[key]:
                raise InvariantViolation(
                    f"routing field {key} present in non-routing mode"
                )

    def assert_safety_enforced(self, intent: Dict[str, Any]) -> None:
        safety = intent.get("safety_decision") or {}
        if not safety:
            return
        final_action_type = _action_type(_action_payload(safety.get("final_action")))
        if final_action_type == "NO_TRADE":
            chosen_type = _action_type(_action_payload(intent.get("chosen_action")))
            if chosen_type != "NO_TRADE":
                raise InvariantViolation(
                    "chosen_action not downgraded to NO_TRADE after safety veto"
                )

    def assert_environment_safe(self, intent: Dict[str, Any]) -> None:
        env = intent.get("environment") or {}
        anomaly = env.get("anomaly_decision") or {}
        if anomaly.get("triggered"):
            final_action_payload = _action_payload(
                (intent.get("safety_decision") or {}).get("final_action")
            )
            if final_action_payload is None:
                final_action_payload = _action_payload(intent.get("chosen_action"))
            final_action = _action_type(final_action_payload)
            if final_action != "NO_TRADE":
                raise InvariantViolation(
                    "anomaly triggered but final_action not NO_TRADE"
                )

    def assert_brain_artifacts_valid(self, intent: Dict[str, Any]) -> None:
        scores = intent.get("scores") or {}
        ev_brain = scores.get("EV_brain")
        unified = scores.get("unified_score")
        mcr = scores.get("MCR") or {}
        mean_ev = mcr.get("mean_EV")
        variance_ev = mcr.get("variance_EV")

        for name, val in ("EV_brain", ev_brain), ("unified_score", unified):
            if val is None:
                raise InvariantViolation(f"missing score: {name}")
            try:
                float(val)
            except Exception:
                raise InvariantViolation(f"non-numeric score: {name}")

        mcr_value = mean_ev if mean_ev is not None else variance_ev
        if mcr_value is None:
            raise InvariantViolation("missing score: MCR.mean_EV or MCR.variance_EV")
        try:
            float(mcr_value)
        except Exception:
            raise InvariantViolation("non-numeric score: MCR metric")

    def assert_mode_transition_valid(self, old_mode: Any, new_mode: Any) -> None:
        from .live_modes import VALID_MODE_TRANSITIONS

        allowed = VALID_MODE_TRANSITIONS.get(old_mode, set())
        if new_mode not in allowed:
            raise InvariantViolation(
                f"invalid mode transition {old_mode} -> {new_mode}"
            )


def _mode_allows_routing(mode_info: Dict[str, Any]) -> bool:
    if not isinstance(mode_info, dict):
        return False
    if "allow_order_routing" in mode_info:
        return bool(mode_info.get("allow_order_routing"))
    capabilities = mode_info.get("capabilities") or {}
    return bool(capabilities.get("allow_order_routing"))


def _action_type(val: Any) -> str:
    if hasattr(val, "value"):
        return str(val.value)
    if isinstance(val, ActionType):
        return str(val.value)
    return str(val) if val is not None else "UNKNOWN"


def _action_payload(obj: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get("action_type")
    if hasattr(obj, "action_type"):
        return getattr(obj, "action_type")
    return obj
