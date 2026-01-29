from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MarketState:
    """Snapshot of the market used by the evaluator and policy engine."""

    # Price data
    current_price: float = 0.0
    recent_returns: List[float] = field(default_factory=list)
    volatility: float = 0.0
    liquidity: float = 0.0
    volume: float = 0.0

    # Regime features
    trend_regime: str = "chop"  # e.g., "up", "down", "chop"
    volatility_regime: str = "normal"  # e.g., "high", "low", "normal"
    liquidity_regime: str = "normal"  # e.g., "high", "low", "normal"
    macro_regime: str = "neutral"  # e.g., "risk_on", "risk_off"
    amd_regime: str = "NEUTRAL"  # ACCUMULATION, DISTRIBUTION, MANIPULATION, NEUTRAL
    amd_confidence: float = 0.0
    session: str = "UNKNOWN"
    momentum_5: float = 0.0
    momentum_10: float = 0.0
    momentum_20: float = 0.0
    roc_5: float = 0.0
    roc_10: float = 0.0
    roc_20: float = 0.0
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
    raw: Dict[str, Any] = field(default_factory=dict)

    # Indicators
    ma_short: float = 0.0
    ma_long: float = 0.0
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
    rsi: float = 50.0
    momentum: float = 0.0
    atr: float = 0.0
    spread: float = 0.0
    volatility_shock: bool = False
    volatility_shock_strength: float = 0.0
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
    risk_current_equity: float = 0.0
    risk_open_risk: float = 0.0
    risk_realized_pnl_today: float = 0.0
    risk_used_today: float = 0.0
    risk_peak_equity: float = 0.0
    risk_current_drawdown: float = 0.0
    risk_last_veto_reason: str = ""

    # Position state
    position_side: str = "flat"  # "long", "short", "flat"
    position_size: float = 0.0
    entry_price: Optional[float] = None
    unrealized_pnl: float = 0.0


@dataclass
class EvaluationOutput:
    """Evaluator output placeholder."""

    score: float
    confidence: float
    trend_regime: str
    volatility_regime: str
    liquidity_regime: str
    macro_regime: str
    risk_flags: List[str] = field(default_factory=list)
    veto_flags: List[str] = field(default_factory=list)


@dataclass
class Action:
    """Policy decision placeholder."""

    action_type: str  # "BUY", "SELL", "HOLD", "CLOSE"
    size: float
    confidence: float
    veto_reason: Optional[str] = None


@dataclass
class TradeResult:
    """Result of executing an Action."""

    realized_pnl: float
    fill_price: float
    slippage: float
    updated_state: MarketState
