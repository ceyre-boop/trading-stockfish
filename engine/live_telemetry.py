from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class LiveDecisionTelemetry:
    timestamp: datetime
    decision_frame: Dict[str, Any]
    ranked_actions: List[Dict[str, Any]]
    chosen_action: Dict[str, Any]
    safety_decision: Optional[Dict[str, Any]]
    search_diagnostics: Dict[str, Any]
    mode: str
    metadata: Dict[str, Any]


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return _to_json_safe(obj.__dict__)
    return obj


def emit_live_telemetry(telemetry: LiveDecisionTelemetry) -> None:
    """
    Write telemetry to a structured log sink.
    Must be JSON-safe, deterministic, and append-only.
    No mutation of existing logs. For now, validation-only.
    """

    try:
        payload = asdict(telemetry)
        payload = _to_json_safe(payload)
        payload["timestamp"] = telemetry.timestamp.isoformat()
        # Serialize to ensure JSON-safe; discard output (sink can be added later)
        json.dumps(payload, ensure_ascii=False)
    except Exception:
        return
