from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from .live_telemetry import LiveDecisionTelemetry


@dataclass
class DailyDecisionSummary:
    date: str
    total_decisions: int
    actions_taken: int
    actions_skipped: int
    no_trade_count: int
    safety_veto_count: int
    anomaly_trigger_count: int
    mode_usage: Dict[str, int]
    regime_breakdown: Dict[str, int]
    entry_model_usage: Dict[str, int]
    ev_vs_realized: Dict[str, float]
    metadata: Dict[str, Any]


class OperatorReportBuilder:
    def __init__(self) -> None:
        self.records: List[LiveDecisionTelemetry] = []

    def record_decision(self, telemetry: LiveDecisionTelemetry) -> None:
        # Append without mutating input to keep deterministic behavior
        self.records.append(telemetry)

    def build_daily_summary(self, date: str) -> DailyDecisionSummary:
        same_day = [t for t in self.records if _date_str(t.timestamp) == date]

        total_decisions = len(same_day)
        actions_taken = 0
        actions_skipped = 0
        no_trade_count = 0
        safety_veto_count = 0
        anomaly_trigger_count = 0
        mode_usage: Dict[str, int] = {}
        regime_breakdown: Dict[str, int] = {}
        entry_model_usage: Dict[str, int] = {}
        ev_brain_values: List[float] = []
        mcr_mean_values: List[float] = []
        realized_r_values: List[float] = []

        for t in same_day:
            chosen_type = _action_type(t.chosen_action.get("action_type"))
            safety = t.safety_decision or {}
            final_action_type = _action_type(
                (safety.get("final_action") or {}).get("action_type")
            )
            allowed = safety.get("allowed", True)

            if (
                chosen_type != "NO_TRADE"
                and allowed
                and final_action_type != "NO_TRADE"
            ):
                actions_taken += 1
            if chosen_type != "NO_TRADE" and final_action_type == "NO_TRADE":
                actions_skipped += 1
            if chosen_type == "NO_TRADE" or final_action_type == "NO_TRADE":
                no_trade_count += 1
            if safety and final_action_type == "NO_TRADE" and safety.get("reason"):
                safety_veto_count += 1

            if _anomaly_triggered(t):
                anomaly_trigger_count += 1

            mode_key = _mode_from_metadata(t) or t.mode
            mode_usage[mode_key] = mode_usage.get(mode_key, 0) + 1

            regime_key = _regime_from_frame(t.decision_frame)
            regime_breakdown[regime_key] = regime_breakdown.get(regime_key, 0) + 1

            model_id = t.chosen_action.get("entry_model_id")
            if model_id:
                entry_model_usage[model_id] = entry_model_usage.get(model_id, 0) + 1

            ev_brain = _safe_float(t.search_diagnostics.get("EV_brain"))
            if ev_brain is not None:
                ev_brain_values.append(ev_brain)

            mcr_mean = _safe_float(
                (t.search_diagnostics.get("MCR") or {}).get("mean_EV")
            )
            if mcr_mean is not None:
                mcr_mean_values.append(mcr_mean)

            realized_r = _safe_float(_realized_r_from_metadata(t.metadata))
            if realized_r is not None:
                realized_r_values.append(realized_r)

        ev_vs_realized = {
            "avg_ev_brain": _mean(ev_brain_values),
            "avg_mcr_mean_ev": _mean(mcr_mean_values),
            "avg_realized_R": _mean(realized_r_values),
        }

        return DailyDecisionSummary(
            date=date,
            total_decisions=total_decisions,
            actions_taken=actions_taken,
            actions_skipped=actions_skipped,
            no_trade_count=no_trade_count,
            safety_veto_count=safety_veto_count,
            anomaly_trigger_count=anomaly_trigger_count,
            mode_usage=mode_usage,
            regime_breakdown=regime_breakdown,
            entry_model_usage=entry_model_usage,
            ev_vs_realized=ev_vs_realized,
            metadata={},
        )


def _date_str(ts: datetime) -> str:
    try:
        return ts.date().isoformat()
    except Exception:
        return ""


def _action_type(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    if value is None:
        return "UNKNOWN"
    return str(value)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _anomaly_triggered(t: LiveDecisionTelemetry) -> bool:
    env = (t.metadata or {}).get("environment") or {}
    anomaly = env.get("anomaly_decision") or {}
    return bool(anomaly.get("triggered"))


def _mode_from_metadata(t: LiveDecisionTelemetry) -> str | None:
    meta = t.metadata or {}
    mode_info = meta.get("mode_info") or {}
    return mode_info.get("mode")


def _regime_from_frame(frame: Dict[str, Any]) -> str:
    if not isinstance(frame, dict):
        return "unknown"
    for key in ("regime", "vol_regime", "trend_regime"):
        val = frame.get(key)
        if val:
            return str(val)
    return "unknown"


def _realized_r_from_metadata(meta: Dict[str, Any]) -> Any:
    if not isinstance(meta, dict):
        return None
    if "realized_R" in meta:
        return meta.get("realized_R")
    performance = meta.get("performance") or {}
    return performance.get("realized_R")
