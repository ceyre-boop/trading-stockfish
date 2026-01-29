"""
Deterministic line search engine v1.

Explores a small, depth-limited tree of future MarketStates using the existing
evaluator and a lightweight execution simulation to produce a principal
variation (best line) ranked by a deterministic EV metric.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.evaluator import evaluate_state
from engine.execution_simulator import ExecutionSimulator, PositionState
from engine.types import MarketState


@dataclass(frozen=True)
class SearchConfig:
    max_depth: int = 2
    branching_limit: int = 3
    weight_pnl: float = 0.7
    weight_eval: float = 0.3


@dataclass(frozen=True)
class LineNode:
    state_index: int
    action: str
    position_side: str
    position_size: float
    entry_price: float
    step_pnl: float
    cumulative_pnl: float
    evaluator_score: float


@dataclass(frozen=True)
class Line:
    nodes: List[LineNode] = field(default_factory=list)
    total_pnl: float = 0.0
    total_evaluator_score: float = 0.0
    combined_ev_metric: float = 0.0


@dataclass(frozen=True)
class LineSearchResult:
    principal_variation: Line
    all_lines: List[Line]


def _candidate_actions(position_side: str) -> List[str]:
    if position_side == "flat":
        return ["HOLD", "ENTER", "EXIT"]
    return ["HOLD", "ADD", "REDUCE", "EXIT"]


def _compute_unrealized(position: Optional[PositionState], price: float) -> float:
    if position is None or position.side == "flat":
        return 0.0
    direction = 1.0 if position.side == "long" else -1.0
    return (price - position.entry_price) * position.quantity * direction


def _normalize(value: float, scale: float) -> float:
    if scale <= 0:
        scale = 1.0
    return value / scale


def _score_line(
    total_pnl: float, total_eval: float, cfg: SearchConfig, price_scale: float
) -> float:
    pnl_component = cfg.weight_pnl * _normalize(total_pnl, max(price_scale, 1.0))
    eval_component = cfg.weight_eval * total_eval
    return pnl_component + eval_component


def evaluate_lines(
    initial_state: MarketState,
    future_states: Sequence[MarketState],
    search_config: Optional[SearchConfig] = None,
) -> LineSearchResult:
    cfg = search_config or SearchConfig()
    depth_limit = max(1, cfg.max_depth)
    simulator = ExecutionSimulator()

    initial_position: Optional[PositionState] = None
    initial_price = float(getattr(initial_state, "current_price", 0.0) or 0.0)
    price_scale = abs(initial_price) if initial_price else 1.0

    lines: List[Line] = []

    def dfs(
        depth: int,
        position: Optional[PositionState],
        cumulative_pnl: float,
        eval_sum: float,
        last_price: float,
        nodes: List[LineNode],
    ):
        if depth >= depth_limit or depth >= len(future_states):
            combined = _score_line(cumulative_pnl, eval_sum, cfg, price_scale)
            lines.append(
                Line(
                    nodes=list(nodes),
                    total_pnl=cumulative_pnl,
                    total_evaluator_score=eval_sum,
                    combined_ev_metric=combined,
                )
            )
            return

        state = future_states[depth]
        price = float(getattr(state, "current_price", 0.0) or 0.0)
        eval_out = evaluate_state(state)

        actions = _candidate_actions(position.side if position else "flat")
        # Prioritize action consistent with evaluator sign before truncation
        if eval_out.score > 0.1 and "ENTER" in actions:
            actions = ["ENTER"] + [a for a in actions if a != "ENTER"]
        elif eval_out.score < -0.1 and "EXIT" in actions:
            actions = ["EXIT"] + [a for a in actions if a != "EXIT"]
        actions = actions[: max(1, cfg.branching_limit)]

        for act in actions:
            action_lower = act.lower()
            target_size = 1.0
            if act == "HOLD":
                new_position = position
                prev_unreal = _compute_unrealized(position, last_price)
                new_unreal = _compute_unrealized(position, price)
                step_pnl = new_unreal - prev_unreal
            else:
                sim_result = simulator.simulate_execution_v4(
                    action=action_lower,
                    target_size=target_size,
                    mid_price=price,
                    liquidity_state=None,
                    volatility_state=None,
                    symbol="SIM",
                    current_position=position,
                    state={"spread": getattr(state, "spread", 0.0)},
                    enable_microstructure=False,
                )
                prev_unreal = _compute_unrealized(position, last_price)
                new_position = sim_result.updated_position
                new_unreal = _compute_unrealized(new_position, price)
                realized = getattr(new_position, "realized_pnl", 0.0)
                step_pnl = (new_unreal - prev_unreal) + realized

            new_cum_pnl = cumulative_pnl + step_pnl
            new_eval_sum = eval_sum + float(eval_out.score)
            node = LineNode(
                state_index=depth,
                action=act,
                position_side=new_position.side if new_position else "flat",
                position_size=new_position.quantity if new_position else 0.0,
                entry_price=new_position.entry_price if new_position else 0.0,
                step_pnl=step_pnl,
                cumulative_pnl=new_cum_pnl,
                evaluator_score=float(eval_out.score),
            )
            nodes.append(node)
            dfs(depth + 1, new_position, new_cum_pnl, new_eval_sum, price, nodes)
            nodes.pop()

    dfs(0, initial_position, 0.0, 0.0, initial_price, [])

    if not lines:
        empty = Line()
        return LineSearchResult(principal_variation=empty, all_lines=[])

    best = max(lines, key=lambda l: l.combined_ev_metric)
    return LineSearchResult(principal_variation=best, all_lines=lines)


def run_line_search(
    current_state: MarketState,
    future_states: Sequence[MarketState],
    search_config: Optional[SearchConfig] = None,
) -> LineSearchResult:
    return evaluate_lines(current_state, future_states, search_config)
