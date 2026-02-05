from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.mcr_engine import (
    aggregate_rollout_results,
    evaluate_action_via_mcr,
    run_rollouts_for_action,
)
from engine.mcr_scenarios import MCRActionContext


def _ctx(action_type: ActionType):
    return MCRActionContext(
        decision_frame_ref="df",
        initial_position_state={},
        decision_action=DecisionAction(action_type=action_type),
        risk_envelope={},
    )


def test_run_rollouts_returns_results_and_deterministic():
    frame = DecisionFrame()
    ctx = _ctx(ActionType.OPEN_LONG)
    r1 = run_rollouts_for_action(frame, ctx, n_paths=3, horizon_bars=4, seed=123)
    r2 = run_rollouts_for_action(frame, ctx, n_paths=3, horizon_bars=4, seed=123)
    assert len(r1) == 3
    assert len(r2) == 3
    assert r1 == r2


def test_aggregate_rollout_results_metrics():
    frame = DecisionFrame()
    ctx = _ctx(ActionType.OPEN_LONG)
    rollouts = run_rollouts_for_action(frame, ctx, n_paths=5, horizon_bars=5, seed=5)
    agg = aggregate_rollout_results(rollouts)

    assert set(agg.keys()) == {
        "mean_EV",
        "variance_EV",
        "tail_risk",
        "stop_hit_rate",
        "tp_hit_rate",
        "avg_time_in_trade",
    }
    # Basic sanity
    assert isinstance(agg["mean_EV"], float)
    assert agg["stop_hit_rate"] >= 0.0
    assert agg["tp_hit_rate"] >= 0.0


def test_evaluate_action_via_mcr_end_to_end_deterministic():
    frame = DecisionFrame()
    ctx = _ctx(ActionType.OPEN_SHORT)
    agg1 = evaluate_action_via_mcr(frame, ctx, n_paths=4, horizon_bars=3, seed=77)
    agg2 = evaluate_action_via_mcr(frame, ctx, n_paths=4, horizon_bars=3, seed=77)
    assert agg1 == agg2
