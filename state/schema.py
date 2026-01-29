from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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
    liquidity: LiquidityTelemetry
    ict_smc: ICTSMCTelemetry
    orderflow: OrderflowTelemetry
    quant: QuantTelemetry
    trend: TrendIndicatorTelemetry
    risk: RiskTelemetry
    line_search: LineSearchSummary
    timestamp: float = 0.0
