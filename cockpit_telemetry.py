"""
Cockpit Telemetry Layer (read-only, deterministic, UI-safe).

Provides frozen DTOs that expose the engine's current view without mutating
core logic. It runs the deterministic pipeline (evaluator → regime bundle →
scenario simulator → lightweight policy/governance/execution summaries) and
returns a single immutable snapshot suitable for UI or logging.
"""

import json
from dataclasses import asdict
from typing import Any, Dict, Tuple

from engine.evaluator import evaluate_state
from engine.evaluator_probabilities import (
    compute_probability_tilts,
    extract_pattern_probabilities,
)
from engine.regime_engine import compute_regime_bundle
from engine.scenario_simulator import ScenarioSimulator
from engine.types import MarketState as CoreMarketState
from state.schema import (
    CockpitSnapshot,
    EvaluatorTelemetry,
    ExecutionTelemetry,
    GovernanceTelemetry,
    ICTSMCTelemetry,
    LineSearchSummary,
    LiquidityTelemetry,
    MarketStructureTelemetry,
    OrderflowTelemetry,
    PolicyTelemetry,
    QuantTelemetry,
    RiskTelemetry,
    ScenarioTelemetry,
    TrendIndicatorTelemetry,
)


def _coerce_market_state(state: Any) -> CoreMarketState:
    if isinstance(state, CoreMarketState):
        return state
    if isinstance(state, dict):
        # Create CoreMarketState while tolerating missing keys
        kwargs: Dict[str, Any] = {}
        for field_name in CoreMarketState.__dataclass_fields__:  # type: ignore[attr-defined]
            kwargs[field_name] = state.get(
                field_name, getattr(CoreMarketState, field_name, 0.0)
            )
        return CoreMarketState(**kwargs)  # type: ignore[arg-type]
    raise TypeError("market_state must be engine.types.MarketState or dict")


def _safe_number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _build_evaluator_telemetry(
    state: CoreMarketState,
) -> Tuple[EvaluatorTelemetry, Dict[str, Any]]:
    eval_out = evaluate_state(state)
    regime_bundle = compute_regime_bundle(state)
    telemetry = EvaluatorTelemetry(
        raw_score=_safe_number(eval_out.score),
        adjusted_score=_safe_number(eval_out.score),
        confidence=_safe_number(eval_out.confidence),
        trend_regime=regime_bundle.get("trend_regime", ""),
        volatility_regime=regime_bundle.get("volatility_regime", ""),
        AMD_phase=regime_bundle.get("amd_regime", ""),
        session_regime=regime_bundle.get("session_regime", ""),
        momentum_5=_safe_number(getattr(state, "momentum_5", 0.0)),
        momentum_10=_safe_number(getattr(state, "momentum_10", 0.0)),
        momentum_20=_safe_number(getattr(state, "momentum_20", 0.0)),
        roc_5=_safe_number(getattr(state, "roc_5", 0.0)),
        roc_10=_safe_number(getattr(state, "roc_10", 0.0)),
        roc_20=_safe_number(getattr(state, "roc_20", 0.0)),
    )
    return telemetry, regime_bundle


def _build_scenario_telemetry(
    state: CoreMarketState, regime_bundle: Dict[str, Any], eval_score: float
) -> ScenarioTelemetry:
    sim = ScenarioSimulator()
    price = _safe_number(getattr(state, "current_price", 0.0))
    vwap = _safe_number(getattr(state, "ma_long", price)) or price
    session_high = _safe_number(getattr(state, "swing_high", price)) or price
    session_low = _safe_number(getattr(state, "swing_low", price)) or price
    expected_move = abs(_safe_number(getattr(state, "volatility", 0.0))) or 0.0001
    vol = _safe_number(getattr(state, "volatility", 0.0))
    result = sim.simulate_scenarios(
        current_price=price,
        vwap=vwap,
        session_high=session_high,
        session_low=session_low,
        expected_move=expected_move,
        volatility=vol,
        regime_label=regime_bundle.get("trend_regime", ""),
        regime_confidence=_safe_number(regime_bundle.get("trend_strength", 0.0)),
        eval_score=eval_score,
    )
    best_scenario = max(result.scenarios, key=lambda s: s.probability)
    scenario_ev = _safe_number(result.expected_price) - price
    return ScenarioTelemetry(
        scenario_ev=scenario_ev,
        alignment_score=_safe_number(result.regime_alignment),
        selected_scenario_type=best_scenario.scenario_type.value,
    )


def _build_policy_telemetry(eval_score: float, confidence: float) -> PolicyTelemetry:
    if eval_score > 0.2:
        action = "BUY"
    elif eval_score < -0.2:
        action = "SELL"
    else:
        action = "HOLD"
    size = min(1.0, max(0.0, abs(eval_score) * confidence))
    return PolicyTelemetry(action=action, size=size, confidence=confidence)


def _build_governance_telemetry() -> GovernanceTelemetry:
    # Governance engine not mutated here; surface a deterministic, read-only view.
    return GovernanceTelemetry(veto_applied=False, veto_reason="APPROVED")


def _build_execution_telemetry(state: CoreMarketState) -> ExecutionTelemetry:
    price = _safe_number(getattr(state, "current_price", 0.0))
    spread = _safe_number(getattr(state, "spread", 0.0))
    simulated_fill = price + spread / 2.0
    return ExecutionTelemetry(
        simulated_fill_price=simulated_fill,
        slippage=0.0,
        shock_penalties=0.0,
        microstructure_flags={
            "quote_pulling": False,
            "spoofing": False,
            "aggressive_trade": False,
        },
    )


def _build_market_structure_telemetry(
    state: CoreMarketState,
) -> MarketStructureTelemetry:
    spread = _safe_number(getattr(state, "spread", 0.0))
    mid = _safe_number(getattr(state, "current_price", 0.0))
    best_bid = mid - spread / 2.0
    best_ask = mid + spread / 2.0
    return MarketStructureTelemetry(
        best_bid=best_bid,
        best_ask=best_ask,
        spread=spread,
        bid_depth=_safe_number(getattr(state, "bid_depth", 0.0)),
        ask_depth=_safe_number(getattr(state, "ask_depth", 0.0)),
        depth_imbalance=_safe_number(getattr(state, "depth_imbalance", 0.0)),
        swing_tag=str(getattr(state, "swing_tag", "NONE")),
        leg_type=str(getattr(state, "current_leg_type", "CORRECTION")),
        last_bos_direction=str(getattr(state, "last_bos_direction", "NONE")),
        last_choch_direction=str(getattr(state, "last_choch_direction", "NONE")),
        session_high=_safe_number(getattr(state, "session_high", 0.0)),
        session_low=_safe_number(getattr(state, "session_low", 0.0)),
        day_high=_safe_number(getattr(state, "day_high", 0.0)),
        day_low=_safe_number(getattr(state, "day_low", 0.0)),
        previous_day_high=_safe_number(getattr(state, "previous_day_high", 0.0)),
        previous_day_low=_safe_number(getattr(state, "previous_day_low", 0.0)),
        previous_day_close=_safe_number(getattr(state, "previous_day_close", 0.0)),
        vwap_price=_safe_number(getattr(state, "vwap_price", 0.0)),
        distance_from_vwap=_safe_number(getattr(state, "distance_from_vwap", 0.0)),
    )


def _build_liquidity_telemetry(state: CoreMarketState) -> LiquidityTelemetry:
    return LiquidityTelemetry(
        has_equal_highs=bool(getattr(state, "has_equal_highs", False)),
        has_equal_lows=bool(getattr(state, "has_equal_lows", False)),
        bsl_zone_price=_safe_number(getattr(state, "bsl_zone_price", 0.0)),
        ssl_zone_price=_safe_number(getattr(state, "ssl_zone_price", 0.0)),
        nearest_bsl_pool_above=_safe_number(
            getattr(state, "nearest_bsl_pool_above", 0.0)
        ),
        nearest_ssl_pool_below=_safe_number(
            getattr(state, "nearest_ssl_pool_below", 0.0)
        ),
        has_liquidity_void=bool(getattr(state, "has_liquidity_void", False)),
        void_upper=_safe_number(getattr(state, "void_upper", 0.0)),
        void_lower=_safe_number(getattr(state, "void_lower", 0.0)),
        stop_cluster_above=_safe_number(getattr(state, "stop_cluster_above", 0.0)),
        stop_cluster_below=_safe_number(getattr(state, "stop_cluster_below", 0.0)),
        last_sweep_direction=str(getattr(state, "last_sweep_direction", "NONE")),
        swept_bsl=bool(getattr(state, "swept_bsl", False)),
        swept_ssl=bool(getattr(state, "swept_ssl", False)),
    )


def _build_ict_smc_telemetry(state: CoreMarketState) -> ICTSMCTelemetry:
    return ICTSMCTelemetry(
        current_bullish_ob_low=_safe_number(
            getattr(state, "current_bullish_ob_low", 0.0)
        ),
        current_bullish_ob_high=_safe_number(
            getattr(state, "current_bullish_ob_high", 0.0)
        ),
        current_bearish_ob_low=_safe_number(
            getattr(state, "current_bearish_ob_low", 0.0)
        ),
        current_bearish_ob_high=_safe_number(
            getattr(state, "current_bearish_ob_high", 0.0)
        ),
        last_touched_ob_type=str(getattr(state, "last_touched_ob_type", "NONE")),
        has_mitigation=bool(getattr(state, "has_mitigation", False)),
        has_flip_zone=bool(getattr(state, "has_flip_zone", False)),
        mitigation_low=_safe_number(getattr(state, "mitigation_low", 0.0)),
        mitigation_high=_safe_number(getattr(state, "mitigation_high", 0.0)),
        flip_low=_safe_number(getattr(state, "flip_low", 0.0)),
        flip_high=_safe_number(getattr(state, "flip_high", 0.0)),
        has_fvg=bool(getattr(state, "has_fvg", False)),
        fvg_upper=_safe_number(getattr(state, "fvg_upper", 0.0)),
        fvg_lower=_safe_number(getattr(state, "fvg_lower", 0.0)),
        has_ifvg=bool(getattr(state, "has_ifvg", False)),
        ifvg_upper=_safe_number(getattr(state, "ifvg_upper", 0.0)),
        ifvg_lower=_safe_number(getattr(state, "ifvg_lower", 0.0)),
        premium_discount_state=str(getattr(state, "premium_discount_state", "EQ")),
        equilibrium_level=_safe_number(getattr(state, "equilibrium_level", 0.0)),
        in_london_killzone=bool(getattr(state, "in_london_killzone", False)),
        in_ny_killzone=bool(getattr(state, "in_ny_killzone", False)),
    )


def _build_orderflow_telemetry(state: CoreMarketState) -> OrderflowTelemetry:
    return OrderflowTelemetry(
        bar_delta=_safe_number(getattr(state, "bar_delta", 0.0)),
        cumulative_delta=_safe_number(getattr(state, "cumulative_delta", 0.0)),
        footprint_imbalance=_safe_number(getattr(state, "footprint_imbalance", 0.0)),
        has_absorption=bool(getattr(state, "has_absorption", False)),
        absorption_side=str(getattr(state, "absorption_side", "NONE")),
        has_exhaustion=bool(getattr(state, "has_exhaustion", False)),
        exhaustion_side=str(getattr(state, "exhaustion_side", "NONE")),
    )


def _build_quant_telemetry(state: CoreMarketState) -> QuantTelemetry:
    probs = extract_pattern_probabilities(state)
    tilts = compute_probability_tilts(state, probs)
    return QuantTelemetry(
        p_sweep_reversal=probs.p_sweep_reversal,
        p_sweep_continuation=probs.p_sweep_continuation,
        p_ob_hold=probs.p_ob_hold,
        p_ob_fail=probs.p_ob_fail,
        p_fvg_fill=probs.p_fvg_fill,
        expected_move_after_sweep=probs.expected_move_after_sweep,
        expected_move_after_ob_touch=probs.expected_move_after_ob_touch,
        expected_move_after_fvg_fill=probs.expected_move_after_fvg_fill,
        total_probability_tilt=tilts["total_probability_tilt"],
    )


def _build_trend_indicator_telemetry(
    state: CoreMarketState,
) -> TrendIndicatorTelemetry:
    return TrendIndicatorTelemetry(
        ema_9=_safe_number(getattr(state, "ema_9", 0.0)),
        ema_20=_safe_number(getattr(state, "ema_20", 0.0)),
        ema_50=_safe_number(getattr(state, "ema_50", 0.0)),
        ema_200=_safe_number(getattr(state, "ema_200", 0.0)),
        sma_20=_safe_number(getattr(state, "sma_20", 0.0)),
        sma_50=_safe_number(getattr(state, "sma_50", 0.0)),
        sma_200=_safe_number(getattr(state, "sma_200", 0.0)),
        ma_stack_state=str(getattr(state, "ma_stack_state", "NEUTRAL")),
        distance_from_ema_20=_safe_number(getattr(state, "distance_from_ema_20", 0.0)),
        distance_from_ema_50=_safe_number(getattr(state, "distance_from_ema_50", 0.0)),
        trend_strength=_safe_number(getattr(state, "trend_strength", 0.0)),
        trend_strength_state=str(getattr(state, "trend_strength_state", "WEAK")),
    )


def _build_risk_telemetry(state: CoreMarketState) -> RiskTelemetry:
    return RiskTelemetry(
        current_equity=_safe_number(getattr(state, "risk_current_equity", 0.0)),
        open_risk=_safe_number(getattr(state, "risk_open_risk", 0.0)),
        realized_pnl_today=_safe_number(getattr(state, "risk_realized_pnl_today", 0.0)),
        risk_used_today=_safe_number(getattr(state, "risk_used_today", 0.0)),
        peak_equity=_safe_number(getattr(state, "risk_peak_equity", 0.0)),
        current_drawdown=_safe_number(getattr(state, "risk_current_drawdown", 0.0)),
        last_risk_veto_reason=str(getattr(state, "risk_last_veto_reason", "")),
    )


def _build_line_search_summary(state: CoreMarketState) -> LineSearchSummary:
    return LineSearchSummary(
        best_line_total_pnl=_safe_number(
            getattr(state, "line_search_best_total_pnl", 0.0)
        ),
        best_line_total_eval=_safe_number(
            getattr(state, "line_search_best_total_eval", 0.0)
        ),
        best_line_depth=int(getattr(state, "line_search_best_depth", 0) or 0),
        best_line_actions=str(getattr(state, "line_search_best_actions", "")),
    )


def build_cockpit_snapshot(market_state: Any) -> CockpitSnapshot:
    state = _coerce_market_state(market_state)
    evaluator, regime_bundle = _build_evaluator_telemetry(state)
    scenario = _build_scenario_telemetry(state, regime_bundle, evaluator.raw_score)
    policy = _build_policy_telemetry(evaluator.raw_score, evaluator.confidence)
    governance = _build_governance_telemetry()
    execution = _build_execution_telemetry(state)
    market_structure = _build_market_structure_telemetry(state)
    liquidity = _build_liquidity_telemetry(state)
    ict_smc = _build_ict_smc_telemetry(state)
    orderflow = _build_orderflow_telemetry(state)
    quant = _build_quant_telemetry(state)
    trend = _build_trend_indicator_telemetry(state)
    risk = _build_risk_telemetry(state)
    line_search = _build_line_search_summary(state)
    timestamp = _safe_number(getattr(state, "timestamp", 0.0))
    return CockpitSnapshot(
        evaluator=evaluator,
        scenario=scenario,
        policy=policy,
        governance=governance,
        execution=execution,
        market_structure=market_structure,
        liquidity=liquidity,
        ict_smc=ict_smc,
        orderflow=orderflow,
        quant=quant,
        trend=trend,
        risk=risk,
        line_search=line_search,
        timestamp=timestamp,
    )


def to_dict(snapshot: CockpitSnapshot) -> Dict[str, Any]:
    return asdict(snapshot)


def to_json(snapshot: CockpitSnapshot) -> str:
    return json.dumps(to_dict(snapshot), sort_keys=True)
