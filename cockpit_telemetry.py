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
    BayesianTelemetry,
    CandlePatternTelemetry,
    CockpitSnapshot,
    EvaluatorTelemetry,
    ExecutionTelemetry,
    GovernanceTelemetry,
    ICTSMCTelemetry,
    LineSearchSummary,
    LiquidityTelemetry,
    MarketStructureTelemetry,
    MomentumIndicatorsTelemetryV2,
    MTFStructureTelemetry,
    NewsMacroTelemetry,
    NewsOllamaTelemetry,
    OrderbookTelemetry,
    OrderflowTelemetry,
    PolicyTelemetry,
    QuantTelemetry,
    RiskTelemetry,
    ScenarioTelemetry,
    TrendIndicatorTelemetry,
    TwitterNewsTelemetry,
    VolumeProfileTelemetry,
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


def _build_mtf_structure_telemetry(state: CoreMarketState) -> MTFStructureTelemetry:
    return MTFStructureTelemetry(
        htf_1h_trend_direction=str(getattr(state, "htf_1h_trend_direction", "RANGE")),
        htf_1h_trend_strength=_safe_number(
            getattr(state, "htf_1h_trend_strength", 0.0)
        ),
        htf_1h_last_bos_direction=str(
            getattr(state, "htf_1h_last_bos_direction", "NONE")
        ),
        htf_1h_last_choch_direction=str(
            getattr(state, "htf_1h_last_choch_direction", "NONE")
        ),
        htf_1h_current_leg_type=str(
            getattr(state, "htf_1h_current_leg_type", "CORRECTION")
        ),
        htf_1h_last_swing_high=_safe_number(
            getattr(state, "htf_1h_last_swing_high", 0.0)
        ),
        htf_1h_last_swing_low=_safe_number(
            getattr(state, "htf_1h_last_swing_low", 0.0)
        ),
        htf_4h_trend_direction=str(getattr(state, "htf_4h_trend_direction", "RANGE")),
        htf_4h_trend_strength=_safe_number(
            getattr(state, "htf_4h_trend_strength", 0.0)
        ),
        htf_4h_last_bos_direction=str(
            getattr(state, "htf_4h_last_bos_direction", "NONE")
        ),
        htf_4h_last_choch_direction=str(
            getattr(state, "htf_4h_last_choch_direction", "NONE")
        ),
        htf_4h_current_leg_type=str(
            getattr(state, "htf_4h_current_leg_type", "CORRECTION")
        ),
        htf_4h_last_swing_high=_safe_number(
            getattr(state, "htf_4h_last_swing_high", 0.0)
        ),
        htf_4h_last_swing_low=_safe_number(
            getattr(state, "htf_4h_last_swing_low", 0.0)
        ),
        htf_d_trend_direction=str(getattr(state, "htf_d_trend_direction", "RANGE")),
        htf_d_trend_strength=_safe_number(getattr(state, "htf_d_trend_strength", 0.0)),
        htf_d_last_bos_direction=str(
            getattr(state, "htf_d_last_bos_direction", "NONE")
        ),
        htf_d_last_choch_direction=str(
            getattr(state, "htf_d_last_choch_direction", "NONE")
        ),
        htf_d_current_leg_type=str(
            getattr(state, "htf_d_current_leg_type", "CORRECTION")
        ),
        htf_d_last_swing_high=_safe_number(
            getattr(state, "htf_d_last_swing_high", 0.0)
        ),
        htf_d_last_swing_low=_safe_number(getattr(state, "htf_d_last_swing_low", 0.0)),
        fractal_state=str(getattr(state, "fractal_state", "NEUTRAL")),
        fractal_score=_safe_number(getattr(state, "fractal_score", 0.0)),
        htf_ltf_alignment_score=_safe_number(
            getattr(state, "htf_ltf_alignment_score", 0.0)
        ),
        htf_bias=str(getattr(state, "htf_bias", "NEUTRAL")),
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


def _build_momentum_v2_telemetry(
    state: CoreMarketState,
) -> MomentumIndicatorsTelemetryV2:
    return MomentumIndicatorsTelemetryV2(
        rsi_value=_safe_number(getattr(state, "rsi_value", 0.0)),
        rsi_state=str(getattr(state, "rsi_state", "NEUTRAL")),
        macd_value=_safe_number(getattr(state, "macd_value", 0.0)),
        macd_signal=_safe_number(getattr(state, "macd_signal", 0.0)),
        macd_histogram=_safe_number(getattr(state, "macd_histogram", 0.0)),
        macd_state=str(getattr(state, "macd_state", "NEUTRAL")),
        stoch_k=_safe_number(getattr(state, "stoch_k", 0.0)),
        stoch_d=_safe_number(getattr(state, "stoch_d", 0.0)),
        stoch_state=str(getattr(state, "stoch_state", "NEUTRAL")),
        momentum_regime=str(getattr(state, "momentum_regime", "CHOP")),
        momentum_confidence=_safe_number(getattr(state, "momentum_confidence", 0.0)),
        rsi_bullish_divergence=bool(getattr(state, "rsi_bullish_divergence", False)),
        rsi_bearish_divergence=bool(getattr(state, "rsi_bearish_divergence", False)),
        macd_bullish_divergence=bool(getattr(state, "macd_bullish_divergence", False)),
        macd_bearish_divergence=bool(getattr(state, "macd_bearish_divergence", False)),
    )


def _build_news_macro_telemetry(state: CoreMarketState) -> NewsMacroTelemetry:
    return NewsMacroTelemetry(
        next_event_type=str(getattr(state, "next_event_type", "NONE")),
        next_event_time_delta=_safe_number(
            getattr(state, "next_event_time_delta", 0.0)
        ),
        next_event_impact=str(getattr(state, "next_event_impact", "NONE")),
        event_risk_window=str(getattr(state, "event_risk_window", "NONE")),
        expected_volatility_state=str(
            getattr(state, "expected_volatility_state", "LOW")
        ),
        expected_volatility_score=_safe_number(
            getattr(state, "expected_volatility_score", 0.0)
        ),
        macro_pressure_score=_safe_number(getattr(state, "macro_pressure_score", 0.0)),
        future_events_count=int(getattr(state, "future_events_count", 0) or 0),
        parsed_events_count=int(getattr(state, "parsed_events_count", 0) or 0),
        liquidity_withdrawal_flag=bool(
            getattr(state, "liquidity_withdrawal_flag", False)
        ),
        macro_regime=str(getattr(state, "macro_regime", "NEUTRAL")),
        macro_regime_score=_safe_number(getattr(state, "macro_regime_score", 0.0)),
        macro_entry_allowed=bool(getattr(state, "macro_entry_allowed", True)),
        macro_position_size_multiplier=_safe_number(
            getattr(state, "macro_position_size_multiplier", 1.0)
        ),
        macro_max_leverage_multiplier=_safe_number(
            getattr(state, "macro_max_leverage_multiplier", 1.0)
        ),
        macro_search_depth_multiplier=_safe_number(
            getattr(state, "macro_search_depth_multiplier", 1.0)
        ),
        macro_aggressiveness_bias=_safe_number(
            getattr(state, "macro_aggressiveness_bias", 0.0)
        ),
        ollama_unreachable=bool(getattr(state, "ollama_unreachable", False)),
    )


def _build_news_ollama_telemetry(state: CoreMarketState) -> NewsOllamaTelemetry:
    snapshot = getattr(state, "news_snapshot", {}) or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    records = snapshot.get("parsed_events", []) if isinstance(snapshot, dict) else []
    record_count = len(records) if isinstance(records, list) else 0
    future_count = int(snapshot.get("future_count", 0) or 0)
    unreachable = bool(snapshot.get("ollama_unreachable", False))
    return NewsOllamaTelemetry(
        sentiment_score=_safe_number(getattr(state, "news_sentiment_score", 0.0)),
        sentiment_volatility=_safe_number(
            getattr(state, "news_sentiment_volatility", 0.0)
        ),
        macro_impact=_safe_number(getattr(state, "news_macro_impact", 0.0)),
        impact=_safe_number(
            getattr(state, "news_impact", getattr(state, "news_macro_impact", 0.0))
        ),
        directional_bias=_safe_number(getattr(state, "news_directional_bias", 0.0)),
        confidence=_safe_number(getattr(state, "news_confidence", 0.0)),
        record_count=record_count,
        future_count=future_count,
        ollama_unreachable=unreachable,
    )


def _build_twitter_news_telemetry(state: CoreMarketState) -> TwitterNewsTelemetry:
    snapshot = getattr(state, "twitter_news_snapshot", {}) or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    score = _safe_number(snapshot.get("sentiment_score", 0.0))
    vol = _safe_number(snapshot.get("sentiment_volatility", 0.0))
    records = snapshot.get("records", []) if isinstance(snapshot, dict) else []
    record_count = len(records) if isinstance(records, list) else 0
    dominant_topic = "other"
    if records:
        topics = [r.get("topic", "other") for r in records if isinstance(r, dict)]
        if topics:
            dominant_topic = max(set(topics), key=topics.count)
    return TwitterNewsTelemetry(
        sentiment_score=score,
        sentiment_volatility=vol,
        record_count=record_count,
        dominant_topic=str(dominant_topic),
    )


def _build_bayesian_telemetry(state: CoreMarketState) -> BayesianTelemetry:
    return BayesianTelemetry(
        bayes_trend_continuation=_safe_number(
            getattr(state, "bayes_trend_continuation", 0.0)
        ),
        bayes_trend_continuation_confidence=_safe_number(
            getattr(state, "bayes_trend_continuation_confidence", 0.0)
        ),
        bayes_trend_reversal=_safe_number(getattr(state, "bayes_trend_reversal", 0.0)),
        bayes_trend_reversal_confidence=_safe_number(
            getattr(state, "bayes_trend_reversal_confidence", 0.0)
        ),
        bayes_sweep_reversal=_safe_number(getattr(state, "bayes_sweep_reversal", 0.0)),
        bayes_sweep_reversal_confidence=_safe_number(
            getattr(state, "bayes_sweep_reversal_confidence", 0.0)
        ),
        bayes_sweep_continuation=_safe_number(
            getattr(state, "bayes_sweep_continuation", 0.0)
        ),
        bayes_sweep_continuation_confidence=_safe_number(
            getattr(state, "bayes_sweep_continuation_confidence", 0.0)
        ),
        bayes_ob_respect=_safe_number(getattr(state, "bayes_ob_respect", 0.0)),
        bayes_ob_respect_confidence=_safe_number(
            getattr(state, "bayes_ob_respect_confidence", 0.0)
        ),
        bayes_ob_violation=_safe_number(getattr(state, "bayes_ob_violation", 0.0)),
        bayes_ob_violation_confidence=_safe_number(
            getattr(state, "bayes_ob_violation_confidence", 0.0)
        ),
        bayes_fvg_fill=_safe_number(getattr(state, "bayes_fvg_fill", 0.0)),
        bayes_fvg_fill_confidence=_safe_number(
            getattr(state, "bayes_fvg_fill_confidence", 0.0)
        ),
        bayes_fvg_reject=_safe_number(getattr(state, "bayes_fvg_reject", 0.0)),
        bayes_fvg_reject_confidence=_safe_number(
            getattr(state, "bayes_fvg_reject_confidence", 0.0)
        ),
        bayesian_update_strength=_safe_number(
            getattr(state, "bayesian_update_strength", 0.0)
        ),
    )


def _build_candle_pattern_telemetry(state: CoreMarketState) -> CandlePatternTelemetry:
    return CandlePatternTelemetry(
        body_size=_safe_number(getattr(state, "body_size", 0.0)),
        upper_wick_size=_safe_number(getattr(state, "upper_wick_size", 0.0)),
        lower_wick_size=_safe_number(getattr(state, "lower_wick_size", 0.0)),
        total_range=_safe_number(getattr(state, "total_range", 0.0)),
        wick_to_body_upper=_safe_number(getattr(state, "wick_to_body_upper", 0.0)),
        wick_to_body_lower=_safe_number(getattr(state, "wick_to_body_lower", 0.0)),
        wick_to_body_total=_safe_number(getattr(state, "wick_to_body_total", 0.0)),
        bullish_engulfing=bool(getattr(state, "bullish_engulfing", False)),
        bearish_engulfing=bool(getattr(state, "bearish_engulfing", False)),
        inside_bar=bool(getattr(state, "inside_bar", False)),
        outside_bar=bool(getattr(state, "outside_bar", False)),
        pin_bar_upper=bool(getattr(state, "pin_bar_upper", False)),
        pin_bar_lower=bool(getattr(state, "pin_bar_lower", False)),
        momentum_bar=bool(getattr(state, "momentum_bar", False)),
        exhaustion_bar=bool(getattr(state, "exhaustion_bar", False)),
        high_volume_candle=bool(getattr(state, "high_volume_candle", False)),
        low_volume_candle=bool(getattr(state, "low_volume_candle", False)),
        pattern_at_liquidity=bool(getattr(state, "pattern_at_liquidity", False)),
        pattern_at_structure=bool(getattr(state, "pattern_at_structure", False)),
        pattern_context_importance=str(
            getattr(state, "pattern_context_importance", "LOW")
        ),
    )


def _build_volume_profile_telemetry(state: CoreMarketState) -> VolumeProfileTelemetry:
    return VolumeProfileTelemetry(
        poc_price=_safe_number(getattr(state, "poc_price", 0.0)),
        hvn_levels=list(getattr(state, "hvn_levels", ())),
        lvn_levels=list(getattr(state, "lvn_levels", ())),
        value_area_low=_safe_number(getattr(state, "value_area_low", 0.0)),
        value_area_high=_safe_number(getattr(state, "value_area_high", 0.0)),
        value_area_coverage=_safe_number(getattr(state, "value_area_coverage", 0.0)),
        price_vs_value_area_state=str(
            getattr(state, "price_vs_value_area_state", "UNKNOWN")
        ),
        near_hvn=bool(getattr(state, "near_hvn", False)),
        near_lvn=bool(getattr(state, "near_lvn", False)),
    )


def _build_orderbook_telemetry(state: CoreMarketState) -> OrderbookTelemetry:
    return OrderbookTelemetry(
        l2_bids=list(getattr(state, "l2_bids", ())),
        l2_asks=list(getattr(state, "l2_asks", ())),
        top_level_imbalance=_safe_number(getattr(state, "top_level_imbalance", 0.0)),
        multi_level_imbalance=_safe_number(
            getattr(state, "multi_level_imbalance", 0.0)
        ),
        spread_ticks=int(getattr(state, "spread_ticks", 0) or 0),
        microstructure_shift=str(getattr(state, "microstructure_shift", "NORMAL")),
        spread_widening=bool(getattr(state, "spread_widening", False)),
        spread_tightening=bool(getattr(state, "spread_tightening", False)),
        hidden_bid_liquidity=bool(getattr(state, "hidden_bid_liquidity", False)),
        hidden_ask_liquidity=bool(getattr(state, "hidden_ask_liquidity", False)),
        queue_position_estimate=_safe_number(
            getattr(state, "queue_position_estimate", 0.0)
        ),
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
    mtf_structure = _build_mtf_structure_telemetry(state)
    candle_patterns = _build_candle_pattern_telemetry(state)
    volume_profile = _build_volume_profile_telemetry(state)
    orderbook = _build_orderbook_telemetry(state)
    momentum_v2 = _build_momentum_v2_telemetry(state)
    news_macro = _build_news_macro_telemetry(state)
    news_ollama = _build_news_ollama_telemetry(state)
    twitter_news = _build_twitter_news_telemetry(state)
    bayesian = _build_bayesian_telemetry(state)
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
        mtf_structure=mtf_structure,
        candle_patterns=candle_patterns,
        volume_profile=volume_profile,
        orderbook=orderbook,
        momentum_v2=momentum_v2,
        news_macro=news_macro,
        news_ollama=news_ollama,
        twitter_news=twitter_news,
        bayesian=bayesian,
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
