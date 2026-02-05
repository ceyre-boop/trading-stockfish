import numpy as np

from .decision_frame import DecisionFrame
from .mcr_pathgen import generate_price_paths
from .mcr_scenarios import MCRActionContext, MCRRolloutResult
from .mcr_simulator import simulate_trade_on_path


def run_rollouts_for_action(
    frame: DecisionFrame,
    action_ctx: MCRActionContext,
    *,
    n_paths: int,
    horizon_bars: int,
    seed: int,
) -> list[MCRRolloutResult]:
    paths = generate_price_paths(
        frame, n_paths=n_paths, horizon_bars=horizon_bars, seed=seed
    )
    results: list[MCRRolloutResult] = []
    for path in paths:
        results.append(simulate_trade_on_path(path, action_ctx))
    return results


def aggregate_rollout_results(rollouts: list[MCRRolloutResult]) -> dict:
    if not rollouts:
        return {
            "mean_EV": 0.0,
            "variance_EV": 0.0,
            "tail_risk": 0.0,
            "stop_hit_rate": 0.0,
            "tp_hit_rate": 0.0,
            "avg_time_in_trade": 0.0,
        }

    realized = np.array([r.realized_R for r in rollouts], dtype=float)
    mean_ev = float(np.mean(realized))
    variance_ev = float(np.var(realized))
    tail_risk = float(np.percentile(realized, 5))
    stop_hit_rate = float(np.mean([1.0 if r.hit_stop else 0.0 for r in rollouts]))
    tp_hit_rate = float(np.mean([1.0 if r.hit_tp else 0.0 for r in rollouts]))
    avg_time = float(np.mean([r.time_in_trade_bars for r in rollouts]))

    return {
        "mean_EV": mean_ev,
        "variance_EV": variance_ev,
        "tail_risk": tail_risk,
        "stop_hit_rate": stop_hit_rate,
        "tp_hit_rate": tp_hit_rate,
        "avg_time_in_trade": avg_time,
    }


def evaluate_action_via_mcr(
    frame: DecisionFrame,
    action_ctx: MCRActionContext,
    *,
    n_paths: int,
    horizon_bars: int,
    seed: int,
) -> dict:
    rollouts = run_rollouts_for_action(
        frame, action_ctx, n_paths=n_paths, horizon_bars=horizon_bars, seed=seed
    )
    return aggregate_rollout_results(rollouts)
