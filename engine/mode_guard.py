from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .live_modes import LiveMode, MODE_CAPABILITIES


class ModeGuard:
    def __init__(self, mode: LiveMode):
        self.mode = mode
        self.capabilities = MODE_CAPABILITIES[mode]

    def enforce(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        intent_copy = dict(intent or {})

        routing_allowed = self.capabilities.allow_order_routing
        position_updates_allowed = self.capabilities.allow_position_updates

        intent_copy["routing_allowed"] = routing_allowed
        if not routing_allowed:
            intent_copy["routing_block_reason"] = "mode_disallows_order_routing"

        intent_copy["position_updates_allowed"] = position_updates_allowed
        if not position_updates_allowed:
            intent_copy["position_block_reason"] = "mode_disallows_position_updates"

        intent_copy["mode_info"] = {
            "mode": self.mode.value,
            "capabilities": asdict(self.capabilities),
            "allow_order_routing": self.capabilities.allow_order_routing,
            "allow_position_updates": self.capabilities.allow_position_updates,
        }

        return intent_copy
