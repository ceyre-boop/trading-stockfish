import hashlib
import json
from collections import OrderedDict
from typing import Any

from .decision_actions import DecisionAction
from .decision_frame import DecisionFrame


class TranspositionTable:
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.table: OrderedDict[str, Any] = OrderedDict()

    def compute_state_hash(
        self,
        frame: DecisionFrame,
        position_state: Any,
        action: DecisionAction,
    ) -> str:
        def _safe_dict(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, (int, float, str, bool)):
                return obj
            if isinstance(obj, dict):
                return {k: _safe_dict(obj[k]) for k in sorted(obj.keys())}
            if hasattr(obj, "__dict__"):
                return _safe_dict(vars(obj))
            return str(obj)

        state_payload = {
            "market_profile_state": getattr(frame, "market_profile_state", None),
            "vol_regime": getattr(frame, "vol_regime", None),
            "trend_regime": getattr(frame, "trend_regime", None),
            "liquidity_bias": (
                None
                if frame.liquidity_frame is None
                else frame.liquidity_frame.get("bias")
            ),
            "session_profile": getattr(frame, "session_profile", None),
            "position_state": _safe_dict(
                position_state if isinstance(position_state, dict) else {}
            ),
            "action": {
                "action_type": (
                    getattr(action.action_type, "value", None) if action else None
                ),
                "entry_model_id": (
                    getattr(action, "entry_model_id", None) if action else None
                ),
                "direction": getattr(action, "direction", None) if action else None,
                "size_bucket": getattr(action, "size_bucket", None) if action else None,
                "stop_structure": _safe_dict(
                    getattr(action, "stop_structure", None) if action else None
                ),
                "tp_structure": _safe_dict(
                    getattr(action, "tp_structure", None) if action else None
                ),
            },
        }

        serialized = json.dumps(state_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def lookup(self, key: str):
        return self.table.get(key)

    def store(self, key: str, value: Any):
        if key in self.table:
            self.table.move_to_end(key)
            self.table[key] = value
            return
        self.table[key] = value
        if len(self.table) > self.max_size:
            self.table.popitem(last=False)
