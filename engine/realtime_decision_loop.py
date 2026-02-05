from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from .anomaly_detector import AnomalyDetector
from .decision_frame import DecisionFrame
from .decision_logger import log_live_decision, log_realtime_decision
from .entry_brain import BrainPolicy
from .entry_models import EntryModelSpec
from .environment_health import EnvironmentHealthMonitor
from .invariants import InvariantChecker, InvariantViolation
from .live_modes import LiveMode
from .live_telemetry import LiveDecisionTelemetry
from .mode_guard import ModeGuard
from .parity_checker import ParityChecker
from .runtime_safety_envelope import RuntimeSafetyEnvelope, SafetyDecision
from .search_engine_v1 import SearchEngineV1

if TYPE_CHECKING:
    from .operator_reports import OperatorReportBuilder


def _copy_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, list):
        return list(obj)
    return obj


def _parse_ts(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime.utcnow()
    return datetime.utcnow()


class RealtimeDecisionLoop:
    def __init__(
        self,
        search_engine: SearchEngineV1,
        brain_policy: BrainPolicy,
        mode: LiveMode,
        safety_envelope: Optional[RuntimeSafetyEnvelope] = None,
        parity_checker: Optional[ParityChecker] = None,
        env_monitor: Optional[EnvironmentHealthMonitor] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        mode_guard: Optional[ModeGuard] = None,
        operator_report_builder: Optional["OperatorReportBuilder"] = None,
        invariant_checker: Optional[InvariantChecker] = None,
    ) -> None:
        self.search_engine = search_engine
        self.brain_policy = brain_policy
        self.mode = mode
        self.safety_envelope = safety_envelope
        self.parity_checker = parity_checker
        self.env_monitor = env_monitor
        self.anomaly_detector = anomaly_detector
        self.mode_guard = mode_guard
        self.operator_report_builder = operator_report_builder
        self.invariant_checker = invariant_checker

    def build_decision_frame(
        self,
        market_state: Dict[str, Any],
        position_state: Any,
        clock_state: Dict[str, Any],
    ) -> DecisionFrame:
        ms = market_state or {}
        cs = clock_state or {}

        frame = DecisionFrame(
            timestamp_utc=ms.get("timestamp_utc") or cs.get("timestamp_utc"),
            symbol=ms.get("symbol"),
            session_context=_copy_obj(ms.get("session_context")),
            condition_vector=_copy_obj(ms.get("condition_vector")),
        )

        frame.vol_regime = ms.get("vol_regime")
        frame.trend_regime = ms.get("trend_regime")
        frame.market_profile_state = ms.get("market_profile_state")
        frame.market_profile_confidence = ms.get("market_profile_confidence")
        frame.market_profile_evidence = _copy_obj(ms.get("market_profile_evidence"))
        frame.session_profile = ms.get("session_profile") or cs.get("session_profile")
        frame.session_profile_confidence = ms.get("session_profile_confidence")
        frame.session_profile_evidence = _copy_obj(ms.get("session_profile_evidence"))
        frame.liquidity_frame = _copy_obj(
            ms.get("liquidity_frame") or ms.get("liquidity")
        )

        frame.entry_signals_present = _copy_obj(ms.get("entry_signals_present"))
        frame.eligible_entry_models = _copy_obj(ms.get("eligible_entry_models"))
        frame.chosen_entry_model_id = ms.get("chosen_entry_model_id")
        frame.risk_per_trade = ms.get("risk_per_trade")
        frame.position_size = ms.get("position_size")
        frame.entry_brain_labels = _copy_obj(ms.get("entry_brain_labels"))
        frame.entry_brain_scores = _copy_obj(ms.get("entry_brain_scores"))
        frame.entry_consistency_report = _copy_obj(ms.get("entry_consistency_report"))

        # Position state linkage if needed later
        if isinstance(position_state, dict) and frame.entry_brain_scores is None:
            frame.entry_brain_scores = _copy_obj(
                position_state.get("entry_brain_scores")
            )

        return frame

    def run_once(
        self,
        market_state: Dict[str, Any],
        position_state: Any,
        clock_state: Dict[str, Any],
        risk_envelope: Any,
    ) -> Dict[str, Any]:
        frame = self.build_decision_frame(market_state, position_state, clock_state)

        health_snapshot = None
        anomaly_decision = None
        if self.env_monitor is not None:
            health_snapshot = self.env_monitor.evaluate(market_state, clock_state)
        if self.anomaly_detector is not None and health_snapshot is not None:
            anomaly_decision = self.anomaly_detector.record_and_evaluate(
                health_snapshot
            )

        entry_models: List[EntryModelSpec] = _copy_obj(
            market_state.get("entry_models") or []
        )

        ranked = self.search_engine.rank_actions(
            frame,
            position_state,
            entry_models,
            risk_envelope,
        )

        chosen_action = ranked[0][0] if ranked else None
        chosen_scores = ranked[0][1] if ranked else {}

        decision_frame_id = (
            f"{frame.timestamp_utc or ''}:{clock_state.get('bar_index', 0)}"
        )

        policy_label = None
        if chosen_action and getattr(chosen_action, "entry_model_id", None):
            try:
                policy_label = self.brain_policy.lookup(
                    chosen_action.entry_model_id, frame
                )
            except Exception:
                policy_label = None

        decision = {
            "decision_frame_id": decision_frame_id,
            "decision_frame": (
                frame.to_dict() if hasattr(frame, "to_dict") else asdict(frame)
            ),
            "chosen_action": chosen_action,
            "ranked_actions": ranked,
            "scores": chosen_scores,
            "timestamp": frame.timestamp_utc,
            "mode": self.mode.value,
            "policy_label": policy_label,
        }

        decision["environment"] = {
            "health": asdict(health_snapshot) if health_snapshot else None,
            "anomaly_decision": asdict(anomaly_decision) if anomaly_decision else None,
        }

        # If anomaly triggered, force NO_TRADE for safety evaluation
        anomaly_triggered = anomaly_decision is not None and getattr(
            anomaly_decision, "triggered", False
        )
        if anomaly_triggered:
            from .decision_actions import ActionType, DecisionAction

            decision["anomaly_forced_no_trade"] = True
            decision["anomaly_reason"] = getattr(
                anomaly_decision, "reason", "environment_anomaly"
            )
            chosen_action = DecisionAction(action_type=ActionType.NO_TRADE)
            decision["chosen_action"] = chosen_action

        if self.safety_envelope is not None:
            if anomaly_triggered:
                safety_decision = SafetyDecision(
                    allowed=True,
                    final_action=decision["chosen_action"],
                    reason="environment_anomaly",
                    details={"anomaly_reason": decision.get("anomaly_reason")},
                )
            else:
                safety_decision = self.safety_envelope.evaluate_intent(
                    frame,
                    position_state,
                    risk_envelope,
                    decision,
                )
            decision["safety_decision"] = asdict(safety_decision)

        # Sync chosen_action with safety final_action to avoid mismatches
        if decision.get("safety_decision"):
            final_action_dict = decision["safety_decision"].get("final_action")
            if final_action_dict:
                chosen_action = final_action_dict
                decision["chosen_action"] = final_action_dict

        if self.mode_guard is not None:
            decision = self.mode_guard.enforce(decision)

        # Build telemetry
        ranked_actions_serialized = []
        for a, s in ranked:
            ranked_actions_serialized.append(
                {"action": getattr(a, "__dict__", a), "scores": s}
            )

        telemetry_ts = _parse_ts(frame.timestamp_utc)
        telemetry = LiveDecisionTelemetry(
            timestamp=telemetry_ts,
            decision_frame=decision["decision_frame"],
            ranked_actions=ranked_actions_serialized,
            chosen_action=(
                getattr(chosen_action, "__dict__", chosen_action)
                if chosen_action
                else {}
            ),
            safety_decision=decision.get("safety_decision"),
            search_diagnostics=decision.get("scores", {}),
            mode=self.mode.value,
            metadata={
                "policy_label": policy_label,
                "environment": decision.get("environment"),
                "mode_info": decision.get("mode_info"),
            },
        )

        parity_report = None
        if self.parity_checker is not None and self.mode != LiveMode.SIM_REPLAY:
            parity_report = self.parity_checker.compare_live_vs_replay(
                frame,
                position_state,
                entry_models,
                risk_envelope,
                ranked,
            )
            telemetry.metadata["parity"] = parity_report

        log_live_decision(telemetry)

        if self.invariant_checker is not None:
            self.invariant_checker.assert_no_order_routing(
                decision, decision.get("mode_info", {})
            )
            self.invariant_checker.assert_safety_enforced(decision)
            self.invariant_checker.assert_environment_safe(decision)
            self.invariant_checker.assert_brain_artifacts_valid(decision)

        if self.operator_report_builder is not None:
            self.operator_report_builder.record_decision(telemetry)

        log_realtime_decision(decision)
        return decision


def run_realtime_dry_run(
    loop: RealtimeDecisionLoop,
    feed: Iterable[Any],
    position_tracker: Any,
    risk_envelope_provider: Any,
    *,
    max_ticks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []

    def _get_position_state():
        if hasattr(position_tracker, "get_state"):
            return position_tracker.get_state()
        if callable(position_tracker):
            return position_tracker()
        return position_tracker

    def _get_risk_envelope():
        if callable(risk_envelope_provider):
            return risk_envelope_provider()
        return risk_envelope_provider

    for idx, item in enumerate(feed):
        if max_ticks is not None and idx >= max_ticks:
            break

        market_state = None
        clock_state = None

        if isinstance(item, dict):
            market_state = item.get("market_state") or item
            clock_state = item.get("clock_state") or {}
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            market_state, clock_state = item
        else:
            market_state = item
            clock_state = {}

        position_state = _get_position_state()
        risk_env = _get_risk_envelope()

        decision = loop.run_once(
            market_state=market_state,
            position_state=position_state,
            clock_state=clock_state,
            risk_envelope=risk_env,
        )
        decisions.append(decision)

    return decisions
