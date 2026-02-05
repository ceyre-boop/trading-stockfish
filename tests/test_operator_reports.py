from datetime import datetime

from engine.decision_actions import ActionType, DecisionAction
from engine.entry_brain import BrainPolicy
from engine.live_modes import LiveMode
from engine.live_telemetry import LiveDecisionTelemetry
from engine.mode_guard import ModeGuard
from engine.operator_reports import OperatorReportBuilder
from engine.realtime_decision_loop import RealtimeDecisionLoop


def _telemetry(
    *,
    ts: datetime,
    regime: str,
    action_type: str,
    entry_model_id: str,
    final_action_type: str,
    safety_reason: str | None,
    anomaly_triggered: bool,
    mode_value: str,
    ev_brain: float,
    mcr_mean_ev: float,
    realized_r: float,
) -> LiveDecisionTelemetry:
    decision_frame = {"regime": regime}
    chosen_action = {"action_type": action_type, "entry_model_id": entry_model_id}
    safety_decision = {
        "allowed": final_action_type != "NO_TRADE",
        "final_action": {"action_type": final_action_type},
        "reason": safety_reason,
    }
    search_diagnostics = {"EV_brain": ev_brain, "MCR": {"mean_EV": mcr_mean_ev}}
    metadata = {
        "environment": {"anomaly_decision": {"triggered": anomaly_triggered}},
        "mode_info": {"mode": mode_value},
        "realized_R": realized_r,
    }
    return LiveDecisionTelemetry(
        timestamp=ts,
        decision_frame=decision_frame,
        ranked_actions=[],
        chosen_action=chosen_action,
        safety_decision=safety_decision,
        search_diagnostics=search_diagnostics,
        mode=mode_value,
        metadata=metadata,
    )


def test_record_decision_is_append_only():
    builder = OperatorReportBuilder()
    t = _telemetry(
        ts=datetime(2025, 1, 1, 0, 0),
        regime="NORMAL",
        action_type="OPEN_LONG",
        entry_model_id="E1",
        final_action_type="OPEN_LONG",
        safety_reason=None,
        anomaly_triggered=False,
        mode_value="sim_live_feed",
        ev_brain=1.0,
        mcr_mean_ev=0.5,
        realized_r=0.1,
    )

    builder.record_decision(t)
    builder.record_decision(t)

    assert len(builder.records) == 2
    assert builder.records[0] is t
    assert builder.records[1] is t


def test_build_daily_summary_aggregates_fields():
    builder = OperatorReportBuilder()
    day = "2025-01-01"

    t1 = _telemetry(
        ts=datetime(2025, 1, 1, 10, 0),
        regime="NORMAL",
        action_type="OPEN_LONG",
        entry_model_id="E1",
        final_action_type="OPEN_LONG",
        safety_reason=None,
        anomaly_triggered=False,
        mode_value="sim_live_feed",
        ev_brain=1.0,
        mcr_mean_ev=0.5,
        realized_r=0.1,
    )

    t2 = _telemetry(
        ts=datetime(2025, 1, 1, 11, 0),
        regime="EXTREME",
        action_type="OPEN_LONG",
        entry_model_id="E2",
        final_action_type="NO_TRADE",
        safety_reason="risk_block",
        anomaly_triggered=True,
        mode_value="paper_trading",
        ev_brain=0.5,
        mcr_mean_ev=0.2,
        realized_r=-0.2,
    )

    t_other_day = _telemetry(
        ts=datetime(2025, 1, 2, 9, 0),
        regime="CALM",
        action_type="NO_TRADE",
        entry_model_id="E3",
        final_action_type="NO_TRADE",
        safety_reason=None,
        anomaly_triggered=False,
        mode_value="sim_live_feed",
        ev_brain=2.0,
        mcr_mean_ev=0.1,
        realized_r=0.0,
    )

    for t in (t1, t2, t_other_day):
        builder.record_decision(t)

    summary = builder.build_daily_summary(day)

    assert summary.date == day
    assert summary.total_decisions == 2
    assert summary.actions_taken == 1
    assert summary.actions_skipped == 1
    assert summary.no_trade_count == 1
    assert summary.safety_veto_count == 1
    assert summary.anomaly_trigger_count == 1

    assert summary.mode_usage == {"sim_live_feed": 1, "paper_trading": 1}
    assert summary.regime_breakdown == {"NORMAL": 1, "EXTREME": 1}
    assert summary.entry_model_usage == {"E1": 1, "E2": 1}

    assert summary.ev_vs_realized["avg_ev_brain"] == (1.0 + 0.5) / 2
    assert summary.ev_vs_realized["avg_mcr_mean_ev"] == (0.5 + 0.2) / 2
    assert summary.ev_vs_realized["avg_realized_R"] == (0.1 - 0.2) / 2


def test_realtime_loop_records_operator_reports(monkeypatch):
    class StubSearchEngine:
        def rank_actions(self, frame, position_state, entry_models, risk_envelope):
            action = DecisionAction(
                ActionType.OPEN_LONG, entry_model_id="E1", direction="LONG"
            )
            return [
                (
                    action,
                    {"unified_score": 1.0, "EV_brain": 0.5, "MCR": {"mean_EV": 0.25}},
                )
            ]

    builder = OperatorReportBuilder()
    loop = RealtimeDecisionLoop(
        search_engine=StubSearchEngine(),  # type: ignore[arg-type]
        brain_policy=BrainPolicy(policy={"E1": "ALLOW"}),
        mode=LiveMode.SIM_REPLAY,
        safety_envelope=None,
        parity_checker=None,
        env_monitor=None,
        anomaly_detector=None,
        mode_guard=ModeGuard(LiveMode.SIM_REPLAY),
        operator_report_builder=builder,
    )

    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_live_decision", lambda t: None
    )
    monkeypatch.setattr(
        "engine.realtime_decision_loop.log_realtime_decision", lambda d: None
    )

    loop.run_once(
        market_state={"timestamp_utc": "2025-01-01T00:00:00Z", "entry_models": []},
        position_state={"is_open": False},
        clock_state={"timestamp_utc": "2025-01-01T00:00:00Z", "bar_index": 1},
        risk_envelope={},
    )

    assert len(builder.records) == 1
    recorded = builder.records[0]
    assert recorded.mode == LiveMode.SIM_REPLAY.value
    assert recorded.chosen_action["entry_model_id"] == "E1"
