"""
Golden Trace facility for deterministic pipeline regression.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple

from engine.causal_evaluator import evaluate_state as causal_evaluate
from engine.execution_simulator import ExecutionSimulator
from engine.governance_engine import GovernanceEngine
from engine.policy_engine import select_action_regime
from engine.scenario_simulator import ScenarioSimulator
from state.schema import MarketState


@dataclass(frozen=True)
class GoldenTraceRecord:
    index: int
    timestamp: float
    market_state_id: str
    evaluator_score: float
    evaluator_confidence: float
    regime: Tuple[Tuple[str, Any], ...]
    scenario_ev: float
    scenario_alignment: float
    policy_action: str
    policy_size: float
    governance_veto: bool
    governance_reason: str
    execution_result: Tuple[Tuple[str, Any], ...]

    def to_json(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "market_state_id": self.market_state_id,
            "evaluator_score": round(self.evaluator_score, 8),
            "evaluator_confidence": round(self.evaluator_confidence, 8),
            "regime": _to_json(self.regime),
            "scenario_ev": round(self.scenario_ev, 8),
            "scenario_alignment": round(self.scenario_alignment, 8),
            "policy_action": self.policy_action,
            "policy_size": round(self.policy_size, 8),
            "governance_veto": self.governance_veto,
            "governance_reason": self.governance_reason,
            "execution_result": _to_json(self.execution_result),
        }


def _freeze_mapping(mapping: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(mapping.items()))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, float):
        return round(float(value), 8)
    return value


def _to_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_json(v) for v in value]
    if isinstance(value, list):
        return [_to_json(v) for v in value]
    return value


def _market_state_hash(state: MarketState) -> str:
    # Convert MarketState to JSON string deterministically for hashing
    def _to_serializable(obj: Any) -> Any:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, MarketState):
            return asdict(obj)
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        return obj

    payload = _to_serializable(state)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _mid_price(state: MarketState) -> float:
    if state.price.mid:
        return float(state.price.mid)
    if state.price.bid or state.price.ask:
        return float((state.price.bid + state.price.ask) / 2)
    return 0.0


def _scenario_eval(
    state: MarketState, eval_output: Dict[str, Any], simulator: ScenarioSimulator
) -> Tuple[float, float]:
    mid = _mid_price(state) or 1.0
    bandwidth = max(state.volatility.intraday_band_width, 0.0)
    expected_move = max(bandwidth * mid, 0.0)
    session_high = mid * (1.0 + bandwidth)
    session_low = mid * (1.0 - bandwidth)
    result = simulator.simulate_scenarios(
        current_price=mid,
        vwap=mid,
        session_high=session_high,
        session_low=session_low,
        expected_move=expected_move,
        volatility=max(state.volatility.realized_vol, 1e-6),
        regime_label=eval_output.get("regime", {}).get("vol", ""),
        regime_confidence=eval_output.get("regime", {}).get("confidence", 0.0),
        eval_score=eval_output.get("score", 0.0),
    )
    alignment = getattr(result, "regime_alignment", 0.0) or 0.0
    return round(float(result.expected_price), 8), round(float(alignment), 8)


class GoldenTraceCollector:
    def __init__(self):
        self.governance = GovernanceEngine()
        self.executor = ExecutionSimulator()
        self.scenario_sim = ScenarioSimulator()

    def collect(self, market_states: Iterable[MarketState]) -> List[GoldenTraceRecord]:
        traces: List[GoldenTraceRecord] = []
        position = None
        for idx, state in enumerate(market_states):
            eval_out = causal_evaluate(state)
            scenario_ev, scenario_align = _scenario_eval(
                state, eval_out, self.scenario_sim
            )
            policy_decision = select_action_regime(state)

            governance_decision = self.governance.decide(
                market_state={
                    "volatility_state": state.volatility.__dict__,
                    "liquidity_state": state.liquidity.__dict__,
                    "regime_state": eval_out.get("regime", {}),
                    "trend_direction": state.trend_direction,
                    "trend_strength": state.trend_strength,
                    "swing_structure": state.swing_structure,
                    "timestamp": state.price.timestamp,
                    "amd_state": state.amd.__dict__,
                },
                eval_output={"eval_score": eval_out["score"]},
                policy_decision=policy_decision,
                execution={"unrealized_pnl": getattr(position, "unrealized_pnl", 0.0)},
            )

            action = governance_decision.adjusted_action or policy_decision.get(
                "action", "EXIT"
            )
            target_size = governance_decision.adjusted_size
            if target_size is None:
                target_size = policy_decision.get(
                    "target_size", policy_decision.get("size", 0.0)
                )

            exec_result = self.executor.simulate_execution_v4(
                action=str(action).lower(),
                target_size=float(target_size or 0.0),
                mid_price=_mid_price(state),
                liquidity_state=None,
                volatility_state=None,
                symbol="TEST",
                current_position=position,
                state={
                    "spread": state.price.spread
                    or abs(state.price.ask - state.price.bid)
                    or 0.0,
                    "liquidity_score": float(
                        state.liquidity.cumulative_depth_bid
                        + state.liquidity.cumulative_depth_ask
                    )
                    / 1_000_000.0,
                    "order_flow_features": {
                        "quote_pulling_score": getattr(
                            state.order_flow, "quote_pulling_score", 0.0
                        ),
                        "spoofing_score": getattr(
                            state.order_flow, "quote_pulling_score", 0.0
                        ),
                    },
                },
            )
            position = exec_result.updated_position or position

            record = GoldenTraceRecord(
                index=idx,
                timestamp=round(
                    float(getattr(state.price, "timestamp", 0.0) or 0.0), 8
                ),
                market_state_id=_market_state_hash(state),
                evaluator_score=round(float(eval_out["score"]), 8),
                evaluator_confidence=round(float(eval_out["confidence"]), 8),
                regime=_freeze(eval_out.get("regime", {})),
                scenario_ev=scenario_ev,
                scenario_alignment=scenario_align,
                policy_action=str(action),
                policy_size=round(float(target_size or 0.0), 8),
                governance_veto=bool(governance_decision.adjusted_action is not None),
                governance_reason=str(governance_decision.reason),
                execution_result=_freeze(asdict(exec_result)),
            )
            traces.append(record)
        return traces


def save_golden_trace(path: str, trace: List[GoldenTraceRecord]) -> None:
    data = [rec.to_json() for rec in trace]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2)


def load_golden_trace(path: str) -> List[GoldenTraceRecord]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records: List[GoldenTraceRecord] = []
    for item in data:
        records.append(
            GoldenTraceRecord(
                index=item["index"],
                timestamp=float(item.get("timestamp", 0.0)),
                market_state_id=item["market_state_id"],
                evaluator_score=round(float(item["evaluator_score"]), 8),
                evaluator_confidence=round(float(item["evaluator_confidence"]), 8),
                regime=_freeze(item.get("regime", ())),
                scenario_ev=round(float(item["scenario_ev"]), 8),
                scenario_alignment=round(float(item.get("scenario_alignment", 0.0)), 8),
                policy_action=item["policy_action"],
                policy_size=round(float(item["policy_size"]), 8),
                governance_veto=bool(item.get("governance_veto", False)),
                governance_reason=item.get("governance_reason", ""),
                execution_result=_freeze(item.get("execution_result", ())),
            )
        )
    return records


def compare_golden_traces(
    a: List[GoldenTraceRecord], b: List[GoldenTraceRecord]
) -> Tuple[bool, str]:
    if len(a) != len(b):
        return False, f"Length mismatch: {len(a)} vs {len(b)}"
    for idx, (ra, rb) in enumerate(zip(a, b)):
        if ra != rb:
            return False, f"Mismatch at index {idx}: {ra} != {rb}"
    return True, ""


__all__ = [
    "GoldenTraceRecord",
    "GoldenTraceCollector",
    "save_golden_trace",
    "load_golden_trace",
    "compare_golden_traces",
]
