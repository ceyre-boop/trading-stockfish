"""
Tournament Harness for cross-mode determinism validation.

Runs a fixed sequence of MarketState objects through the canonical pipeline
(evaluator → scenario simulator → policy → governance → execution) across
multiple modes and asserts identical traces.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from engine.causal_evaluator import evaluate_state as causal_evaluate
from engine.execution_simulator import ExecutionSimulator
from engine.governance_engine import GovernanceEngine
from engine.policy_engine import select_action_regime
from engine.scenario_simulator import ScenarioSimulator
from golden_trace import (
    GoldenTraceCollector,
    compare_golden_traces,
    load_golden_trace,
    save_golden_trace,
)
from state.regime_engine import RegimeSignal
from state.schema import (
    AMDState,
    ExecutionContext,
    LiquidityState,
    MacroNewsState,
    MarketState,
    OrderFlowState,
    PriceState,
    VolatilityState,
)

MODES = ["OFFICIAL_MODE", "SANDBOX_MODE", "TEST_MODE", "REPLAY_MODE"]


@dataclass(frozen=True)
class TraceEntry:
    index: int
    mode: str
    eval_score: float
    eval_confidence: float
    regime: Dict[str, Any]
    scenario_ev: float
    policy_action: str
    policy_size: float
    governance_reason: str
    governance_adjusted_action: Any
    execution_fill_price: float
    execution_filled: float

    def as_tuple(self) -> Tuple:
        return (
            self.index,
            round(self.eval_score, 8),
            round(self.eval_confidence, 8),
            self._frozen_regime(),
            round(self.scenario_ev, 8),
            self.policy_action,
            round(self.policy_size, 8),
            self.governance_reason,
            self.governance_adjusted_action,
            round(self.execution_fill_price, 8),
            round(self.execution_filled, 8),
        )

    def _frozen_regime(self) -> Tuple:
        return tuple(sorted((self.regime or {}).items()))


def _mid_price(state: MarketState) -> float:
    if state.price.mid:
        return float(state.price.mid)
    if state.price.bid or state.price.ask:
        return float((state.price.bid + state.price.ask) / 2)
    return 0.0


def _scenario_ev(
    state: MarketState, eval_output: Dict[str, Any], simulator: ScenarioSimulator
) -> float:
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
        regime_label=eval_output["regime"].get("vol", ""),
        regime_confidence=eval_output["regime"].get("confidence", 0.0),
        eval_score=eval_output["score"],
    )
    return float(result.expected_price)


def _execution_state_from_market(state: MarketState) -> Dict[str, Any]:
    spread = state.price.spread or abs(state.price.ask - state.price.bid) or 0.0
    liquidity_score = (
        float(
            state.liquidity.cumulative_depth_bid + state.liquidity.cumulative_depth_ask
        )
        / 1_000_000.0
    )
    order_flow_features = {
        "quote_pulling_score": getattr(state.order_flow, "quote_pulling_score", 0.0),
        "spoofing_score": getattr(state.order_flow, "quote_pulling_score", 0.0),
    }
    return {
        "spread": spread,
        "liquidity_score": liquidity_score,
        "order_flow_features": order_flow_features,
    }


def _normalize_action(action: str) -> str:
    normalized = str(action or "").upper()
    if normalized in {"FLAT", "EXIT"}:
        return "exit"
    if normalized.startswith("ENTER"):
        return "enter"
    if normalized == "ADD":
        return "add"
    if normalized == "REDUCE":
        return "reduce"
    return normalized.lower() or "exit"


class _ModeEnv:
    def __init__(self, mode: str):
        self.mode = mode
        self.prev = os.environ.get("OFFICIAL_MODE")

    def __enter__(self):
        if self.mode == "OFFICIAL_MODE":
            os.environ["OFFICIAL_MODE"] = "1"
        else:
            os.environ.pop("OFFICIAL_MODE", None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prev is None:
            os.environ.pop("OFFICIAL_MODE", None)
        else:
            os.environ["OFFICIAL_MODE"] = self.prev
        return False


def run_mode(mode: str, market_states: Iterable[MarketState]) -> List[TraceEntry]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")

    governance = GovernanceEngine()
    executor = ExecutionSimulator()
    scenario_sim = ScenarioSimulator()

    traces: List[TraceEntry] = []
    position = None

    with _ModeEnv(mode):
        for idx, state in enumerate(market_states):
            eval_out = causal_evaluate(state)
            scenario_ev = _scenario_ev(state, eval_out, scenario_sim)
            policy_decision = select_action_regime(state)

            governance_decision = governance.decide(
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

            exec_state = _execution_state_from_market(state)
            exec_result = executor.simulate_execution_v4(
                action=_normalize_action(action),
                target_size=float(target_size or 0.0),
                mid_price=_mid_price(state),
                liquidity_state=None,
                volatility_state=None,
                symbol="TEST",
                current_position=position,
                state=exec_state,
            )
            position = exec_result.updated_position or position

            traces.append(
                TraceEntry(
                    index=idx,
                    mode=mode,
                    eval_score=float(eval_out["score"]),
                    eval_confidence=float(eval_out["confidence"]),
                    regime=eval_out.get("regime", {}),
                    scenario_ev=scenario_ev,
                    policy_action=str(action),
                    policy_size=float(target_size or 0.0),
                    governance_reason=str(governance_decision.reason),
                    governance_adjusted_action=governance_decision.adjusted_action,
                    execution_fill_price=float(exec_result.fill_price),
                    execution_filled=float(exec_result.actual_filled_size),
                )
            )
    return traces


def compare_traces(
    reference: List[TraceEntry], candidate: List[TraceEntry]
) -> Tuple[bool, str]:
    if len(reference) != len(candidate):
        return False, f"Trace length mismatch: {len(reference)} vs {len(candidate)}"
    for idx, (a, b) in enumerate(zip(reference, candidate)):
        if a.as_tuple() != b.as_tuple():
            return False, f"Mismatch at index {idx}: {a.as_tuple()} != {b.as_tuple()}"
    return True, ""


def run_tournament(market_states: Iterable[MarketState]) -> Dict[str, List[TraceEntry]]:
    states = list(market_states)
    traces: Dict[str, List[TraceEntry]] = {}
    for mode in MODES:
        traces[mode] = run_mode(mode, states)
    return traces


def assert_cross_mode_determinism(market_states: Iterable[MarketState]) -> None:
    traces = run_tournament(market_states)
    reference = traces["OFFICIAL_MODE"]
    for mode in MODES:
        if mode == "OFFICIAL_MODE":
            continue
        ok, reason = compare_traces(reference, traces[mode])
        if not ok:
            raise AssertionError(f"Determinism failed for {mode}: {reason}")


def generate_golden_trace(market_states: Iterable[MarketState]) -> List:
    collector = GoldenTraceCollector()
    return collector.collect(market_states)


def compare_against_golden(path: str, market_states: Iterable[MarketState]) -> None:
    current = generate_golden_trace(market_states)
    golden = load_golden_trace(path)
    ok, reason = compare_golden_traces(golden, current)
    if not ok:
        raise AssertionError(f"Golden trace mismatch: {reason}")


def save_official_golden(path: str, market_states: Iterable[MarketState]) -> None:
    trace = generate_golden_trace(market_states)
    save_golden_trace(path, trace)


def sample_market_states() -> List[MarketState]:
    seq: List[MarketState] = []
    mids = [100.0, 100.3, 100.6, 100.4]
    regimes = ["NORMAL", "HIGH", "NORMAL", "LOW"]
    for idx, (mid, reg) in enumerate(zip(mids, regimes)):
        regime = RegimeSignal(
            vol=reg,
            liq="NORMAL",
            macro="NEUTRAL",
            confidence=0.8,
            volatility_shock=False,
            volatility_shock_strength=0.0,
            trend_direction="UP" if idx % 2 == 0 else "DOWN",
            trend_strength=0.2,
            swing_structure="NEUTRAL",
        )
        seq.append(
            MarketState(
                price=PriceState(
                    timestamp=float(idx),
                    mid=mid,
                    bid=mid - 0.01,
                    ask=mid + 0.01,
                    spread=0.02,
                ),
                order_flow=OrderFlowState(),
                liquidity=LiquidityState(
                    top_depth_bid=5_000,
                    top_depth_ask=5_000,
                    cumulative_depth_bid=10_000,
                    cumulative_depth_ask=10_000,
                    depth_imbalance=0.0,
                    liquidity_resilience=0.1,
                    liquidity_pressure=0.0,
                    liquidity_shock=False,
                    regime="NORMAL",
                ),
                volatility=VolatilityState(
                    realized_vol=0.5,
                    intraday_band_width=0.02,
                    vol_of_vol=0.1,
                    vol_regime=reg,
                    volatility_shock=False,
                    volatility_shock_strength=0.0,
                ),
                macro=MacroNewsState(
                    risk_sentiment=0.1,
                    hawkishness=0.0,
                    surprise_score=0.0,
                    macro_regime="RISK_OFF",
                ),
                execution=ExecutionContext(
                    position_size=0.0,
                    avg_entry_price=None,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                ),
                amd=AMDState(),
                swing_high=mid * 1.01,
                swing_low=mid * 0.99,
                swing_structure="NEUTRAL",
                trend_direction="UP" if idx % 2 == 0 else "DOWN",
                trend_strength=0.2,
                raw={"regime": regime},
            )
        )
    return seq


__all__ = [
    "TraceEntry",
    "run_mode",
    "compare_traces",
    "run_tournament",
    "assert_cross_mode_determinism",
    "sample_market_states",
]
