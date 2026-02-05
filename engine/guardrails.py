from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from engine.modes import Mode, get_adapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUARDRAIL_LOG_DIR = PROJECT_ROOT / "logs" / "guardrails"
SAFE_MODE_STATE = PROJECT_ROOT / "logs" / "safe_mode_state.txt"
GUARDRAIL_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


@dataclass
class GuardrailDecision:
    triggered: bool
    reason: str
    guardrail_type: str
    safe_mode_required: bool
    timestamp: datetime

    def to_json(self) -> Dict[str, object]:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "guardrail_type": self.guardrail_type,
            "safe_mode_required": self.safe_mode_required,
            "timestamp": self.timestamp.isoformat(),
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _as_float(source: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(source.get(key, default) or 0.0)
    except Exception:
        return float(default)


def _as_int(source: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(source.get(key, default) or 0)
    except Exception:
        return int(default)


def _as_bool(source: Dict[str, Any], key: str, default: bool = False) -> bool:
    val = source.get(key, default)
    if isinstance(val, bool):
        return val
    try:
        return str(val).lower() in {"true", "1", "yes", "on"}
    except Exception:
        return bool(default)


def _violation(
    guardrail_type: str, reason: str, safe_mode_required: bool
) -> GuardrailDecision:
    return GuardrailDecision(
        triggered=True,
        reason=reason,
        guardrail_type=guardrail_type,
        safe_mode_required=safe_mode_required,
        timestamp=_utc_now(),
    )


def _safe_mode_state() -> str:
    try:
        if SAFE_MODE_STATE.exists():
            return SAFE_MODE_STATE.read_text(encoding="utf-8").strip() or "UNKNOWN"
    except Exception:
        pass
    return "UNKNOWN"


def _activate_safe_mode() -> str:
    try:
        SAFE_MODE_STATE.parent.mkdir(parents=True, exist_ok=True)
        SAFE_MODE_STATE.write_text("ON", encoding="utf-8")
        return "ON"
    except Exception:
        return _safe_mode_state()


def check_runtime_limits(
    state: Dict[str, Any], metrics: Dict[str, Any]
) -> GuardrailDecision:
    loss_threshold = _as_float(
        state, "max_daily_loss", _as_float(metrics, "max_daily_loss")
    )
    realized_loss = _as_float(
        metrics, "realized_loss", max(-_as_float(metrics, "realized_pnl"), 0.0)
    )
    unrealized_loss = _as_float(
        metrics, "unrealized_loss", max(-_as_float(metrics, "unrealized_pnl"), 0.0)
    )
    total_loss = realized_loss + unrealized_loss
    if loss_threshold and total_loss > loss_threshold:
        return _violation(
            "max_daily_loss",
            f"loss {total_loss:.4f}>{loss_threshold:.4f}",
            safe_mode_required=True,
        )

    pos_threshold = _as_float(
        state, "max_position_size", _as_float(metrics, "max_position_size")
    )
    position = _as_float(metrics, "position_size")
    if pos_threshold and abs(position) > abs(pos_threshold):
        return _violation(
            "max_position_size",
            f"position {position:.4f}>{pos_threshold:.4f}",
            safe_mode_required=False,
        )

    lev_threshold = _as_float(state, "max_leverage", _as_float(metrics, "max_leverage"))
    leverage = _as_float(metrics, "leverage")
    if lev_threshold and abs(leverage) > abs(lev_threshold):
        return _violation(
            "max_leverage",
            f"leverage {leverage:.4f}>{lev_threshold:.4f}",
            safe_mode_required=True,
        )

    freq_threshold = _as_float(
        state, "max_orders_per_min", _as_float(metrics, "max_orders_per_min")
    )
    orders_per_min = _as_float(metrics, "orders_per_min")
    if freq_threshold and orders_per_min > freq_threshold:
        return _violation(
            "max_order_frequency",
            f"orders_per_min {orders_per_min:.4f}>{freq_threshold:.4f}",
            safe_mode_required=False,
        )

    slip_threshold = _as_float(
        state, "max_slippage", _as_float(metrics, "max_slippage")
    )
    slippage = _as_float(metrics, "estimated_slippage")
    if slip_threshold and slippage > slip_threshold:
        return _violation(
            "max_slippage",
            f"slippage {slippage:.4f}>{slip_threshold:.4f}",
            safe_mode_required=False,
        )

    hb_threshold = _as_float(
        state, "max_heartbeat_age_sec", _as_float(metrics, "max_heartbeat_age_sec")
    )
    heartbeat_age = _as_float(metrics, "heartbeat_age_sec")
    if hb_threshold and heartbeat_age > hb_threshold:
        return _violation(
            "heartbeat_timeout",
            f"heartbeat_age {heartbeat_age:.4f}s>{hb_threshold:.4f}s",
            safe_mode_required=True,
        )

    failure_threshold = _as_int(
        state, "max_connector_failures", _as_int(metrics, "max_connector_failures")
    )
    connector_failures = _as_int(metrics, "connector_failures")
    if failure_threshold and connector_failures > failure_threshold:
        return _violation(
            "connector_failure",
            f"connector_failures {connector_failures}>{failure_threshold}",
            safe_mode_required=True,
        )

    if _as_bool(metrics, "safe_mode_active"):
        return _violation(
            "safe_mode_active",
            "SAFE_MODE already active",
            safe_mode_required=True,
        )

    if _as_bool(metrics, "kill_switch_active"):
        return _violation(
            "kill_switch",
            "kill-switch active",
            safe_mode_required=True,
        )

    return GuardrailDecision(
        triggered=False,
        reason="stable",
        guardrail_type="none",
        safe_mode_required=False,
        timestamp=_utc_now(),
    )


def _log_guardrail_event(
    decision: GuardrailDecision, action: str, safe_mode_state: str
) -> Path:
    GUARDRAIL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = decision.timestamp
    log_path = GUARDRAIL_LOG_DIR / f"guardrail_{ts:%Y%m%d_%H%M%S}.json"
    payload = decision.to_json() | {
        "action": action,
        "safe_mode_state": safe_mode_state,
    }
    try:
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        logger.warning("Failed to write guardrail log: %s", log_path)
    return log_path


def apply_guardrail_decision(
    decision: GuardrailDecision, adapter=None
) -> Dict[str, Any]:
    if not decision.triggered:
        return {"applied": False, "action": "NONE"}

    adapter = adapter or get_adapter(Mode.LIVE)
    action_taken = "orders_blocked"
    safe_mode_state = _safe_mode_state()

    try:
        if hasattr(adapter, "disable_orders"):
            adapter.disable_orders()
    except Exception:
        logger.exception("Failed to disable orders via adapter")

    if decision.safe_mode_required:
        safe_mode_state = _activate_safe_mode()
        action_taken = "safe_mode_activated"

    log_path = _log_guardrail_event(decision, action_taken, safe_mode_state)
    return {
        "applied": True,
        "action": action_taken,
        "safe_mode_state": safe_mode_state,
        "log_path": log_path,
    }


def kill_switch(mode: Mode, adapter=None) -> Dict[str, object]:
    adapter = adapter or get_adapter(mode)
    adapter.disable_orders()
    return {"mode": mode.value, "adapter": adapter.name, "disabled": adapter.disabled}
