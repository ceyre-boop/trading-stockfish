"""
Institutional-grade real-time engine loop for Trading Stockfish
- Subscribes to event bus
- Builds market state
- Evaluates state
- Generates candidate actions
- Applies meta-policy (v3.2)
- Applies adaptive weighting (v3.3)
- Simulates execution
- Updates portfolio/risk
- Emits diagnostics
- Deterministic and replay-compatible
"""

"""
Institutional-grade real-time engine loop for Trading Stockfish
- Subscribes to event bus
- Builds market state
- Evaluates state
- Generates candidate actions
- Applies meta-policy (v3.2)
- Applies adaptive weighting (v3.3)
- Simulates execution
- Updates portfolio/risk
- Emits diagnostics
- Deterministic and replay-compatible
"""

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Optional

from data.unified_feed import UnifiedFeed

logger = logging.getLogger(__name__)


class EngineLoop:

    def __init__(
        self,
        event_bus: Any,
        state_builder: Any,
        evaluator: Any,
        meta_policy: Any,
        adaptive_weighting: Any,
        executor: Any,
        portfolio_manager: Any,
        diagnostics: Any,
        experiment_one_shot: bool = False,
        experiment_id: str = "first_one_shot_live_paper",
    ):
        self.event_bus = event_bus
        self.state_builder = state_builder
        self.evaluator = evaluator
        self.meta_policy = meta_policy
        self.adaptive_weighting = adaptive_weighting
        self.executor = executor
        self.portfolio_manager = portfolio_manager
        self.diagnostics = diagnostics
        self.current_state = None
        self.last_diagnostics = None
        self.experiment_one_shot = experiment_one_shot
        self.experiment_id = experiment_id
        self.has_traded_today = False
        self.trade_opened = False
        self.trade_closed = False
        self.experiment_log_path = f"logs/{self.experiment_id}.jsonl"

    def subscribe(self) -> None:
        self.event_bus.subscribe(self.on_event)

    def on_event(self, event: Dict[str, Any]) -> None:
        self.run_step(event)

    def run_step(self, event: Dict[str, Any]) -> Any:
        _ensure_real_event(event)

        # 1. Build market state
        self.current_state = self.state_builder.build(event)
        _ensure_valid_state(self.state_builder, self.current_state)
        # 2. Evaluate state
        evaluation = self.evaluator.evaluate(self.current_state)
        _ensure_valid_evaluation(evaluation)
        # 3. Generate candidate actions
        candidates = self.evaluator.generate_candidates(self.current_state, evaluation)
        # 4. Apply meta-policy (v3.2)
        meta_actions = self.meta_policy.apply(candidates, self.current_state)
        # 5. Apply adaptive weighting (v3.3)
        weighted_actions = self.adaptive_weighting.apply(
            meta_actions, self.current_state
        )
        _ensure_valid_policy_output(weighted_actions)
        # 6. Simulate execution
        execution_result = self.executor.simulate(weighted_actions, self.current_state)
        # 7. Update portfolio/risk
        self.portfolio_manager.update(execution_result)

        # --- Guarantee diagnostics dict with all required fields and placeholders ---
        # evaluator_output
        evaluator_output = (
            evaluation
            if evaluation is not None
            else {"score": 0, "confidence": 0, "factors": {}}
        )
        if not isinstance(evaluator_output, dict):
            evaluator_output = {"score": 0, "confidence": 0, "factors": {}}
        evaluator_output.setdefault("score", 0)
        evaluator_output.setdefault("confidence", 0)
        evaluator_output.setdefault("factors", {})

        # policy_decision
        if (
            weighted_actions
            and isinstance(weighted_actions, list)
            and weighted_actions[0]
        ):
            policy_decision = weighted_actions[0]
        else:
            policy_decision = {"action": "FLAT", "size": 0}
        policy_decision.setdefault("action", "FLAT")
        policy_decision.setdefault("size", 0)

        # governance_decision
        governance_decision = getattr(self, "last_governance_decision", None)
        if governance_decision is None:
            governance_decision = {"approved": True, "reason": "NONE"}
        governance_decision.setdefault("approved", True)
        governance_decision.setdefault("reason", "NONE")

        # execution_result
        exec_result = (
            execution_result if execution_result is not None else {"filled": False}
        )
        if not isinstance(exec_result, dict):
            exec_result = {"filled": False}
        exec_result.setdefault("filled", False)

        # --- Diagnostics dict ---
        diagnostics = {
            "market_state": self.current_state,
            "evaluator_output": evaluator_output,
            "policy_decision": policy_decision,
            "governance_decision": governance_decision,
            "execution_result": exec_result,
        }
        self.last_diagnostics = diagnostics

        # === EXPERIMENT ONE-SHOT LOGIC ===
        if self.experiment_one_shot:
            os.makedirs(os.path.dirname(self.experiment_log_path), exist_ok=True)
            with open(self.experiment_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(diagnostics) + "\n")
                f.flush()

        # After trade closed, veto all new entries (legacy logic)
        if hasattr(self, "trade_closed") and self.trade_closed:
            for wa in weighted_actions:
                if wa["action"] in ["ENTER_LONG", "ENTER_SHORT"]:
                    wa["action"] = "HOLD"
                    wa["size"] = 0.0
        return execution_result

    def get_last_diagnostics(self) -> Any:
        return self.last_diagnostics


def _ensure_real_event(event: Dict[str, Any]) -> None:
    tick = event.get("tick", {}) if isinstance(event, dict) else {}
    if tick.get("synthetic") or event.get("synthetic"):
        raise RuntimeError("Synthetic ticks are forbidden")
    required_fields = ["timestamp", "bid", "ask", "mid", "volume"]
    for field in required_fields:
        if tick.get(field) is None:
            raise RuntimeError(
                f"Missing required tick field {field}; synthetic fallback is forbidden"
            )
    book = event.get("book", {}) or {}
    if book:
        if not book.get("bids") or not book.get("asks"):
            raise RuntimeError(
                "Order book entries must include bids and asks; synthetic depth is forbidden"
            )


def _ensure_valid_state(state_builder: Any, state: Any) -> None:
    if state is None:
        raise RuntimeError("State builder returned None; aborting run")
    if hasattr(state_builder, "validate_state"):
        ok, errors = state_builder.validate_state(state)
        if not ok:
            raise RuntimeError(f"Invalid market state: {errors}")


def _ensure_valid_evaluation(evaluation: Any) -> None:
    if evaluation is None:
        raise RuntimeError("Evaluator returned None")
    if not isinstance(evaluation, dict):
        raise RuntimeError("Evaluator output must be a dict")
    for field in ["score", "confidence"]:
        if field not in evaluation:
            raise RuntimeError(f"Evaluator missing field {field}")
        val = evaluation[field]
        if val is None or not isinstance(val, (int, float)) or math.isnan(val):
            raise RuntimeError(f"Evaluator field {field} malformed")


def _ensure_valid_policy_output(weighted_actions: Any) -> None:
    if weighted_actions is None:
        raise RuntimeError("Policy output is None")
    if not isinstance(weighted_actions, list):
        raise RuntimeError("Policy output must be a list")
    for action in weighted_actions:
        if not isinstance(action, dict):
            raise RuntimeError("Policy action must be dict")
        if action.get("action") is None:
            raise RuntimeError("Policy action missing action field")
        if "size" in action:
            size = action.get("size")
            if size is None or not isinstance(size, (int, float)) or math.isnan(size):
                raise RuntimeError("Policy action size malformed")


# Factory for replay/live compatibility
def create_engine_loop(deps: Dict[str, Any]) -> EngineLoop:
    return EngineLoop(
        event_bus=deps["event_bus"],
        state_builder=deps["state_builder"],
        evaluator=deps["evaluator"],
        meta_policy=deps["meta_policy"],
        adaptive_weighting=deps["adaptive_weighting"],
        executor=deps["executor"],
        portfolio_manager=deps["portfolio_manager"],
        diagnostics=deps["diagnostics"],
        experiment_one_shot=deps.get("experiment_one_shot", False),
        experiment_id=deps.get("experiment_id", "first_one_shot_live_paper"),
    )


# Explicit subclass for orchestrator compatibility
class RealTimeEngineLoop(EngineLoop):
    pass


def _log_diagnostics(diagnostics: List[Dict[str, Any]], experiment_id: str) -> None:
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", f"{experiment_id}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for diag in diagnostics:
            f.write(json.dumps(diag) + "\n")


def run_engine_mode(
    mode: str,
    symbol: str,
    date: Optional[str] = None,
    scenario_path: Optional[str] = None,
    timespan: str = "minute",
    max_ticks: Optional[int] = None,
    experiment_id: str = "engine_mode_run",
    deps: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run engine in specified mode with unified feed.

    Modes:
    - polygon_historical: load bars and replay deterministically
    - polygon_live: stream live via polling
    - mt5_live: stream live from MT5
    - scenario_replay: deterministic replay from scenario file
    """

    if deps is None:
        raise RuntimeError(
            "Real dependencies must be provided; mock/synthetic pipelines are disabled"
        )
    engine = create_engine_loop(
        {
            **deps,
            "experiment_one_shot": True,
            "experiment_id": experiment_id,
        }
    )

    if mode == "polygon_historical":
        feed = UnifiedFeed("polygon", symbol)
        events = feed.load_historical({"date": date, "timespan": timespan})
        event_iter: Iterable[Dict[str, Any]] = events
    elif mode == "polygon_live":
        feed = UnifiedFeed("polygon", symbol)
        event_iter = feed.stream()
    elif mode == "mt5_live":
        feed = UnifiedFeed("mt5", symbol)
        event_iter = feed.stream()
    elif mode == "scenario_replay":
        feed = UnifiedFeed("scenario", symbol, scenario_path=scenario_path)
        events = feed.load_historical()
        event_iter = events
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    diagnostics: List[Dict[str, Any]] = []
    cumulative_pnl = 0.0
    for idx, event in enumerate(event_iter):
        try:
            _ensure_real_event(event)
            result = engine.run_step(event)
        except Exception as exc:
            logger.critical(f"Aborting engine run due to invalid event/state: {exc}")
            raise
        diag = engine.get_last_diagnostics() or {}
        pnl_delta = 0.0
        exec_res = diag.get("execution_result", {}) if isinstance(diag, dict) else {}
        if isinstance(exec_res, dict) and exec_res.get("filled") and "pnl" in exec_res:
            pnl_delta = exec_res.get("pnl", 0.0)
        cumulative_pnl += pnl_delta
        diag["cumulative_pnl"] = cumulative_pnl
        diagnostics.append(diag)
        if max_ticks is not None and idx + 1 >= max_ticks:
            break

    _log_diagnostics(diagnostics, experiment_id)
    print(
        f"Mode {mode} complete: ticks processed={len(diagnostics)}, cumulative_pnl={cumulative_pnl}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Trading Stockfish engine in various modes"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["polygon_historical", "polygon_live", "mt5_live", "scenario_replay"],
    )
    parser.add_argument("--symbol", required=True, help="Symbol/ticker")
    parser.add_argument("--date", help="Date for polygon historical YYYY-MM-DD")
    parser.add_argument("--scenario-path", help="Path to scenario file for replay")
    parser.add_argument(
        "--timespan",
        default="minute",
        help="Timespan for polygon historical (default: minute)",
    )
    parser.add_argument(
        "--max-ticks", type=int, help="Optional max ticks to process (for live modes)"
    )
    parser.add_argument(
        "--experiment-id",
        default="engine_mode_run",
        help="Experiment identifier for logs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_engine_mode(
        mode=args.mode,
        symbol=args.symbol,
        date=args.date,
        scenario_path=args.scenario_path,
        timespan=args.timespan,
        max_ticks=args.max_ticks,
        experiment_id=args.experiment_id,
    )


__all__ = [
    "EngineLoop",
    "RealTimeEngineLoop",
    "create_engine_loop",
    "run_engine_mode",
    "main",
]


if __name__ == "__main__":
    main()
