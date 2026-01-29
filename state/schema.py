from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PriceState:
    timestamp: float = 0.0
    mid: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    candles: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderFlowState:
    aggressive_buy_volume: float = 0.0
    aggressive_sell_volume: float = 0.0
    net_imbalance: float = 0.0
    sweep_flag: bool = False
    quote_pulling_score: float = 0.0


@dataclass(frozen=True)
class LiquidityState:
    top_depth_bid: float = 0.0
    top_depth_ask: float = 0.0
    cumulative_depth_bid: float = 0.0
    cumulative_depth_ask: float = 0.0
    depth_imbalance: float = 0.0
    liquidity_resilience: float = 0.0
    liquidity_pressure: float = 0.0
    liquidity_shock: bool = False
    regime: str = "NORMAL"  # DEEP, NORMAL, THIN, FRAGILE


@dataclass(frozen=True)
class VolatilityState:
    realized_vol: float = 0.0
    intraday_band_width: float = 0.0
    vol_of_vol: float = 0.0
    vol_regime: str = "NORMAL"  # LOW, NORMAL, HIGH, EXTREME
    volatility_shock: bool = False
    volatility_shock_strength: float = 0.0


@dataclass(frozen=True)
class MacroNewsState:
    hawkishness: float = 0.0
    risk_sentiment: float = 0.0
    surprise_score: float = 0.0
    macro_regime: str = "RISK_OFF"  # RISK_ON, RISK_OFF, EVENT


@dataclass(frozen=True)
class ExecutionContext:
    position_size: float = 0.0
    avg_entry_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass(frozen=True)
class AMDState:
    amd_tag: str = "NEUTRAL"
    amd_confidence: float = 0.0
    amd_price_change: float = 0.0
    amd_price_range: float = 0.0


@dataclass(frozen=True)
class MarketState:
    price: PriceState = field(default_factory=PriceState)
    order_flow: OrderFlowState = field(default_factory=OrderFlowState)
    liquidity: LiquidityState = field(default_factory=LiquidityState)
    volatility: VolatilityState = field(default_factory=VolatilityState)
    macro: MacroNewsState = field(default_factory=MacroNewsState)
    execution: ExecutionContext = field(default_factory=ExecutionContext)
    amd: AMDState = field(default_factory=AMDState)
    session: str = "UNKNOWN"
    momentum_5: float = 0.0
    momentum_10: float = 0.0
    momentum_20: float = 0.0
    roc_5: float = 0.0
    roc_10: float = 0.0
    roc_20: float = 0.0
    rsi_value: float = 0.0
    rsi_state: str = "NEUTRAL"
    macd_value: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_state: str = "NEUTRAL"
    stoch_k: float = 0.0
    stoch_d: float = 0.0
    stoch_state: str = "NEUTRAL"
    momentum_regime: str = "CHOP"
    momentum_confidence: float = 0.0
    rsi_bullish_divergence: bool = False
    rsi_bearish_divergence: bool = False
    macd_bullish_divergence: bool = False
    macd_bearish_divergence: bool = False
    next_event_type: str = "NONE"
    next_event_time_delta: float = 0.0
    next_event_impact: str = "NONE"
    event_risk_window: str = "NONE"
    expected_volatility_state: str = "LOW"
    expected_volatility_score: float = 0.0
    macro_pressure_score: float = 0.0
    macro_entry_allowed: bool = True
    macro_position_size_multiplier: float = 1.0
    macro_max_leverage_multiplier: float = 1.0
    macro_adjusted_priors: Dict[str, float] = field(default_factory=dict)
    macro_search_depth_multiplier: float = 1.0
    macro_aggressiveness_bias: float = 0.0
    ollama_unreachable: bool = False
    future_events_count: int = 0
    parsed_events_count: int = 0
    liquidity_withdrawal_flag: bool = False
    macro_regime: str = "NEUTRAL"
    macro_regime_score: float = 0.0
    bayes_trend_continuation: float = 0.0
    bayes_trend_continuation_confidence: float = 0.0
    bayes_trend_reversal: float = 0.0
    bayes_trend_reversal_confidence: float = 0.0
    bayes_sweep_reversal: float = 0.0
    bayes_sweep_reversal_confidence: float = 0.0
    bayes_sweep_continuation: float = 0.0
    bayes_sweep_continuation_confidence: float = 0.0
    bayes_ob_respect: float = 0.0
    bayes_ob_respect_confidence: float = 0.0
    bayes_ob_violation: float = 0.0
    bayes_ob_violation_confidence: float = 0.0
    bayes_fvg_fill: float = 0.0
    bayes_fvg_fill_confidence: float = 0.0
    bayes_fvg_reject: float = 0.0
    bayes_fvg_reject_confidence: float = 0.0
    bayesian_update_strength: float = 0.0
    sentiment_score: float = 0.0
    sentiment_volatility: float = 0.0
    news_sentiment_score: float = 0.0
    news_sentiment_volatility: float = 0.0
    news_macro_impact: float = 0.0
    news_impact: float = 0.0
    news_directional_bias: float = 0.0
    news_confidence: float = 0.0
    news_snapshot: Dict[str, Any] = field(default_factory=dict)
    twitter_sentiment_score: float = 0.0
    twitter_sentiment_volatility: float = 0.0
    twitter_news_snapshot: Dict[str, Any] = field(default_factory=dict)
    ema_9: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ma_stack_state: str = "NEUTRAL"
    distance_from_ema_20: float = 0.0
    distance_from_ema_50: float = 0.0
    trend_strength_state: str = "WEAK"
    bid_depth: float = 0.0
    p_sweep_reversal: float = 0.5
    p_sweep_continuation: float = 0.5
    p_ob_hold: float = 0.5
    p_ob_fail: float = 0.5
    p_fvg_fill: float = 0.5
    expected_move_after_sweep: float = 0.0
    expected_move_after_ob_touch: float = 0.0
    expected_move_after_fvg_fill: float = 0.0
    ask_depth: float = 0.0
    depth_imbalance: float = 0.0
    bar_delta: float = 0.0
    cumulative_delta: float = 0.0
    footprint_imbalance: float = 0.0
    has_absorption: bool = False
    absorption_side: str = "NONE"
    has_exhaustion: bool = False
    exhaustion_side: str = "NONE"
    swing_high: float = 0.0
    swing_low: float = 0.0
    swing_structure: str = "NEUTRAL"
    trend_direction: str = "RANGE"
    trend_strength: float = 0.0
    swing_tag: str = "NONE"
    current_leg_type: str = "CORRECTION"
    last_bos_direction: str = "NONE"
    last_choch_direction: str = "NONE"
    session_high: float = 0.0
    session_low: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    previous_day_high: float = 0.0
    previous_day_low: float = 0.0
    previous_day_close: float = 0.0
    vwap_price: float = 0.0
    distance_from_vwap: float = 0.0
    has_equal_highs: bool = False
    has_equal_lows: bool = False
    bsl_zone_price: float = 0.0
    ssl_zone_price: float = 0.0
    nearest_bsl_pool_above: float = 0.0
    nearest_ssl_pool_below: float = 0.0
    has_liquidity_void: bool = False
    void_upper: float = 0.0
    void_lower: float = 0.0
    stop_cluster_above: float = 0.0
    stop_cluster_below: float = 0.0
    last_sweep_direction: str = "NONE"
    swept_bsl: bool = False
    swept_ssl: bool = False
    htf_1h_trend_direction: str = "RANGE"
    htf_1h_trend_strength: float = 0.0
    htf_1h_last_bos_direction: str = "NONE"
    htf_1h_last_choch_direction: str = "NONE"
    htf_1h_current_leg_type: str = "CORRECTION"
    htf_1h_last_swing_high: float = 0.0
    htf_1h_last_swing_low: float = 0.0
    htf_4h_trend_direction: str = "RANGE"
    htf_4h_trend_strength: float = 0.0
    htf_4h_last_bos_direction: str = "NONE"
    htf_4h_last_choch_direction: str = "NONE"
    htf_4h_current_leg_type: str = "CORRECTION"
    htf_4h_last_swing_high: float = 0.0
    htf_4h_last_swing_low: float = 0.0
    htf_d_trend_direction: str = "RANGE"
    htf_d_trend_strength: float = 0.0
    htf_d_last_bos_direction: str = "NONE"
    htf_d_last_choch_direction: str = "NONE"
    htf_d_current_leg_type: str = "CORRECTION"
    htf_d_last_swing_high: float = 0.0
    htf_d_last_swing_low: float = 0.0
    fractal_state: str = "NEUTRAL"
    fractal_score: float = 0.0
    htf_ltf_alignment_score: float = 0.0
    htf_bias: str = "NEUTRAL"
    current_bullish_ob_low: float = 0.0
    current_bullish_ob_high: float = 0.0
    current_bearish_ob_low: float = 0.0
    current_bearish_ob_high: float = 0.0
    last_touched_ob_type: str = "NONE"
    has_mitigation: bool = False
    has_flip_zone: bool = False
    mitigation_low: float = 0.0
    mitigation_high: float = 0.0
    flip_low: float = 0.0
    flip_high: float = 0.0
    has_fvg: bool = False
    fvg_upper: float = 0.0
    fvg_lower: float = 0.0
    has_ifvg: bool = False
    ifvg_upper: float = 0.0
    ifvg_lower: float = 0.0
    premium_discount_state: str = "EQ"
    equilibrium_level: float = 0.0
    in_london_killzone: bool = False
    in_ny_killzone: bool = False
    body_size: float = 0.0
    upper_wick_size: float = 0.0
    lower_wick_size: float = 0.0
    total_range: float = 0.0
    wick_to_body_upper: float = 0.0
    wick_to_body_lower: float = 0.0
    wick_to_body_total: float = 0.0
    bullish_engulfing: bool = False
    bearish_engulfing: bool = False
    inside_bar: bool = False
    outside_bar: bool = False
    pin_bar_upper: bool = False
    pin_bar_lower: bool = False
    momentum_bar: bool = False
    exhaustion_bar: bool = False
    high_volume_candle: bool = False
    low_volume_candle: bool = False
    pattern_at_liquidity: bool = False
    pattern_at_structure: bool = False
    pattern_context_importance: str = "LOW"
    poc_price: float = 0.0
    hvn_levels: List[float] = field(default_factory=list)
    lvn_levels: List[float] = field(default_factory=list)
    value_area_low: float = 0.0
    value_area_high: float = 0.0
    value_area_coverage: float = 0.0
    price_vs_value_area_state: str = "UNKNOWN"
    near_hvn: bool = False
    near_lvn: bool = False
    l2_bids: List[Dict[str, float]] = field(default_factory=list)
    l2_asks: List[Dict[str, float]] = field(default_factory=list)
    top_level_imbalance: float = 0.0
    multi_level_imbalance: float = 0.0
    spread_ticks: int = 0
    microstructure_shift: str = "NORMAL"
    spread_widening: bool = False
    spread_tightening: bool = False
    hidden_bid_liquidity: bool = False
    hidden_ask_liquidity: bool = False
    queue_position_estimate: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)
    risk_current_equity: float = 0.0
    risk_open_risk: float = 0.0
    risk_realized_pnl_today: float = 0.0
    risk_used_today: float = 0.0
    risk_peak_equity: float = 0.0
    risk_current_drawdown: float = 0.0
    risk_last_veto_reason: str = ""


@dataclass(frozen=True)
class QuantTelemetry:
    p_sweep_reversal: float = 0.5
    p_sweep_continuation: float = 0.5
    p_ob_hold: float = 0.5
    p_ob_fail: float = 0.5
    p_fvg_fill: float = 0.5
    expected_move_after_sweep: float = 0.0
    expected_move_after_ob_touch: float = 0.0
    expected_move_after_fvg_fill: float = 0.0
    total_probability_tilt: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


# Cockpit Telemetry DTOs (read-only, UI-safe)


@dataclass(frozen=True)
class EvaluatorTelemetry:
    raw_score: float = 0.0
    adjusted_score: float = 0.0
    confidence: float = 0.0
    trend_regime: str = ""
    volatility_regime: str = ""
    AMD_phase: str = ""
    session_regime: str = ""
    momentum_5: float = 0.0
    momentum_10: float = 0.0
    momentum_20: float = 0.0
    roc_5: float = 0.0
    roc_10: float = 0.0
    roc_20: float = 0.0


@dataclass(frozen=True)
class ScenarioTelemetry:
    scenario_ev: float = 0.0
    alignment_score: float = 0.0
    selected_scenario_type: str = ""


@dataclass(frozen=True)
class PolicyTelemetry:
    action: str = ""
    size: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True)
class GovernanceTelemetry:
    veto_applied: bool = False
    veto_reason: str = ""


@dataclass(frozen=True)
class ExecutionTelemetry:
    simulated_fill_price: float = 0.0
    slippage: float = 0.0
    shock_penalties: float = 0.0
    microstructure_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketStructureTelemetry:
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    depth_imbalance: float = 0.0
    swing_tag: str = "NONE"
    leg_type: str = "CORRECTION"
    last_bos_direction: str = "NONE"
    last_choch_direction: str = "NONE"
    session_high: float = 0.0
    session_low: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    previous_day_high: float = 0.0
    previous_day_low: float = 0.0
    previous_day_close: float = 0.0
    vwap_price: float = 0.0
    distance_from_vwap: float = 0.0


@dataclass(frozen=True)
class MTFStructureTelemetry:
    htf_1h_trend_direction: str = "RANGE"
    htf_1h_trend_strength: float = 0.0
    htf_1h_last_bos_direction: str = "NONE"
    htf_1h_last_choch_direction: str = "NONE"
    htf_1h_current_leg_type: str = "CORRECTION"
    htf_1h_last_swing_high: float = 0.0
    htf_1h_last_swing_low: float = 0.0
    htf_4h_trend_direction: str = "RANGE"
    htf_4h_trend_strength: float = 0.0
    htf_4h_last_bos_direction: str = "NONE"
    htf_4h_last_choch_direction: str = "NONE"
    htf_4h_current_leg_type: str = "CORRECTION"
    htf_4h_last_swing_high: float = 0.0
    htf_4h_last_swing_low: float = 0.0
    htf_d_trend_direction: str = "RANGE"
    htf_d_trend_strength: float = 0.0
    htf_d_last_bos_direction: str = "NONE"
    htf_d_last_choch_direction: str = "NONE"
    htf_d_current_leg_type: str = "CORRECTION"
    htf_d_last_swing_high: float = 0.0
    htf_d_last_swing_low: float = 0.0
    fractal_state: str = "NEUTRAL"
    fractal_score: float = 0.0
    htf_ltf_alignment_score: float = 0.0
    htf_bias: str = "NEUTRAL"


@dataclass(frozen=True)
class LiquidityTelemetry:
    has_equal_highs: bool = False
    has_equal_lows: bool = False
    bsl_zone_price: float = 0.0
    ssl_zone_price: float = 0.0
    nearest_bsl_pool_above: float = 0.0
    nearest_ssl_pool_below: float = 0.0
    has_liquidity_void: bool = False
    void_upper: float = 0.0
    void_lower: float = 0.0
    stop_cluster_above: float = 0.0
    stop_cluster_below: float = 0.0
    last_sweep_direction: str = "NONE"
    swept_bsl: bool = False
    swept_ssl: bool = False


@dataclass(frozen=True)
class OrderflowTelemetry:
    bar_delta: float = 0.0
    cumulative_delta: float = 0.0
    footprint_imbalance: float = 0.0
    has_absorption: bool = False
    absorption_side: str = "NONE"
    has_exhaustion: bool = False
    exhaustion_side: str = "NONE"


@dataclass(frozen=True)
class ICTSMCTelemetry:
    current_bullish_ob_low: float = 0.0
    current_bullish_ob_high: float = 0.0
    current_bearish_ob_low: float = 0.0
    current_bearish_ob_high: float = 0.0
    last_touched_ob_type: str = "NONE"
    has_mitigation: bool = False
    has_flip_zone: bool = False
    mitigation_low: float = 0.0
    mitigation_high: float = 0.0
    flip_low: float = 0.0
    flip_high: float = 0.0
    has_fvg: bool = False
    fvg_upper: float = 0.0
    fvg_lower: float = 0.0
    has_ifvg: bool = False
    ifvg_upper: float = 0.0
    ifvg_lower: float = 0.0
    premium_discount_state: str = "EQ"
    equilibrium_level: float = 0.0
    in_london_killzone: bool = False
    in_ny_killzone: bool = False


@dataclass(frozen=True)
class TrendIndicatorTelemetry:
    ema_9: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ma_stack_state: str = "NEUTRAL"
    distance_from_ema_20: float = 0.0
    distance_from_ema_50: float = 0.0
    trend_strength: float = 0.0
    trend_strength_state: str = "WEAK"


@dataclass(frozen=True)
class MomentumIndicatorsTelemetryV2:
    rsi_value: float = 0.0
    rsi_state: str = "NEUTRAL"
    macd_value: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_state: str = "NEUTRAL"
    stoch_k: float = 0.0
    stoch_d: float = 0.0
    stoch_state: str = "NEUTRAL"
    momentum_regime: str = "CHOP"
    momentum_confidence: float = 0.0
    rsi_bullish_divergence: bool = False
    rsi_bearish_divergence: bool = False
    macd_bullish_divergence: bool = False
    macd_bearish_divergence: bool = False


@dataclass(frozen=True)
class NewsMacroTelemetry:
    next_event_type: str = "NONE"
    next_event_time_delta: float = 0.0
    next_event_impact: str = "NONE"
    event_risk_window: str = "NONE"
    expected_volatility_state: str = "LOW"
    expected_volatility_score: float = 0.0
    macro_pressure_score: float = 0.0
    future_events_count: int = 0
    parsed_events_count: int = 0
    liquidity_withdrawal_flag: bool = False
    macro_regime: str = "NEUTRAL"
    macro_regime_score: float = 0.0
    macro_entry_allowed: bool = True
    macro_position_size_multiplier: float = 1.0
    macro_max_leverage_multiplier: float = 1.0
    macro_search_depth_multiplier: float = 1.0
    macro_aggressiveness_bias: float = 0.0
    ollama_unreachable: bool = False


@dataclass(frozen=True)
class NewsOllamaTelemetry:
    sentiment_score: float = 0.0
    sentiment_volatility: float = 0.0
    macro_impact: float = 0.0
    impact: float = 0.0
    directional_bias: float = 0.0
    confidence: float = 0.0
    record_count: int = 0
    future_count: int = 0
    ollama_unreachable: bool = False


@dataclass(frozen=True)
class FutureEvent:
    origin_id: str = ""
    event_name: str = ""
    timestamp: str = ""
    time_delta_minutes: float = 0.0
    impact_level: str = "MEDIUM"
    event_type: str = "OTHER"
    asset_scope: List[str] = field(default_factory=list)
    risk_window: bool = False
    macro_pressure_score: float = 0.0


@dataclass(frozen=True)
class ParsedEvent:
    source: str = "forex"
    origin_id: str = ""
    timestamp: str = ""
    asset_scope: List[str] = field(default_factory=list)
    event_type: str = "OTHER"
    impact_level: str = "MEDIUM"
    directional_bias: str = "NEUTRAL"
    confidence: float = 0.0
    sentiment_score: float = 0.0
    sentiment_volatility: float = 0.0
    summary: str = ""
    keywords: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MacroPressureTelemetry:
    macro_pressure_score: float = 0.0
    future_events_count: int = 0
    parsed_events_count: int = 0


@dataclass(frozen=True)
class TwitterNewsTelemetry:
    sentiment_score: float = 0.0
    sentiment_volatility: float = 0.0
    record_count: int = 0
    dominant_topic: str = "other"


@dataclass(frozen=True)
class BayesianTelemetry:
    bayes_trend_continuation: float = 0.0
    bayes_trend_continuation_confidence: float = 0.0
    bayes_trend_reversal: float = 0.0
    bayes_trend_reversal_confidence: float = 0.0
    bayes_sweep_reversal: float = 0.0
    bayes_sweep_reversal_confidence: float = 0.0
    bayes_sweep_continuation: float = 0.0
    bayes_sweep_continuation_confidence: float = 0.0
    bayes_ob_respect: float = 0.0
    bayes_ob_respect_confidence: float = 0.0
    bayes_ob_violation: float = 0.0
    bayes_ob_violation_confidence: float = 0.0
    bayes_fvg_fill: float = 0.0
    bayes_fvg_fill_confidence: float = 0.0
    bayes_fvg_reject: float = 0.0
    bayes_fvg_reject_confidence: float = 0.0
    bayesian_update_strength: float = 0.0


@dataclass(frozen=True)
class CandlePatternTelemetry:
    body_size: float = 0.0
    upper_wick_size: float = 0.0
    lower_wick_size: float = 0.0
    total_range: float = 0.0
    wick_to_body_upper: float = 0.0
    wick_to_body_lower: float = 0.0
    wick_to_body_total: float = 0.0
    bullish_engulfing: bool = False
    bearish_engulfing: bool = False
    inside_bar: bool = False
    outside_bar: bool = False
    pin_bar_upper: bool = False
    pin_bar_lower: bool = False
    momentum_bar: bool = False
    exhaustion_bar: bool = False
    high_volume_candle: bool = False
    low_volume_candle: bool = False
    pattern_at_liquidity: bool = False
    pattern_at_structure: bool = False
    pattern_context_importance: str = "LOW"


@dataclass(frozen=True)
class VolumeProfileTelemetry:
    poc_price: float = 0.0
    hvn_levels: List[float] = field(default_factory=list)
    lvn_levels: List[float] = field(default_factory=list)
    value_area_low: float = 0.0
    value_area_high: float = 0.0
    value_area_coverage: float = 0.0
    price_vs_value_area_state: str = "UNKNOWN"
    near_hvn: bool = False
    near_lvn: bool = False


@dataclass(frozen=True)
class OrderbookTelemetry:
    l2_bids: List[Dict[str, float]] = field(default_factory=list)
    l2_asks: List[Dict[str, float]] = field(default_factory=list)
    top_level_imbalance: float = 0.0
    multi_level_imbalance: float = 0.0
    spread_ticks: int = 0
    microstructure_shift: str = "NORMAL"
    spread_widening: bool = False
    spread_tightening: bool = False
    hidden_bid_liquidity: bool = False
    hidden_ask_liquidity: bool = False
    queue_position_estimate: float = 0.0


@dataclass(frozen=True)
class RiskTelemetry:
    current_equity: float = 0.0
    open_risk: float = 0.0
    realized_pnl_today: float = 0.0
    risk_used_today: float = 0.0
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    last_risk_veto_reason: str = ""


@dataclass(frozen=True)
class LineSearchSummary:
    best_line_total_pnl: float = 0.0
    best_line_total_eval: float = 0.0
    best_line_depth: int = 0
    best_line_actions: str = ""


@dataclass(frozen=True)
class CockpitSnapshot:
    evaluator: EvaluatorTelemetry
    scenario: ScenarioTelemetry
    policy: PolicyTelemetry
    governance: GovernanceTelemetry
    execution: ExecutionTelemetry
    market_structure: MarketStructureTelemetry
    mtf_structure: MTFStructureTelemetry
    candle_patterns: CandlePatternTelemetry
    volume_profile: VolumeProfileTelemetry
    orderbook: OrderbookTelemetry
    liquidity: LiquidityTelemetry
    ict_smc: ICTSMCTelemetry
    orderflow: OrderflowTelemetry
    quant: QuantTelemetry
    trend: TrendIndicatorTelemetry
    momentum_v2: MomentumIndicatorsTelemetryV2
    news_macro: NewsMacroTelemetry
    news_ollama: NewsOllamaTelemetry
    twitter_news: TwitterNewsTelemetry
    bayesian: BayesianTelemetry
    risk: RiskTelemetry
    line_search: LineSearchSummary
    timestamp: float = 0.0
