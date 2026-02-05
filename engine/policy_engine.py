#!/usr/bin/env python3
"""
Policy Engine - Deterministic, Risk-Aware Trading Decision System

Implements a Stockfish-style policy engine that:
  1. Consumes CausalEvaluator output (eval_score, confidence, reasoning)
  2. Analyzes current position state (side, size, entry price, P&L)
  3. Applies deterministic risk-aware decision rules
  4. Outputs discrete trading actions with target position sizes
  5. Provides full reasoning for every decision

# Phase 12 note: strategy_id/entry_model_id/exit_model_id are placeholders only;
# no strategy selection or runtime behavior changes are introduced in this layer.

Philosophy: Like Stockfish evaluates chess positions and generates candidate moves,
this engine evaluates market conditions and generates candidate trading actions.

Decision Actions:
  - ENTER_SMALL: Open small position (low conviction or high risk)
  - ENTER_FULL: Open full position (high conviction, acceptable risk)
  - ADD: Increase existing position (eval strengthens)
  - HOLD: Maintain current position (no actionable change)
  - REDUCE: Decrease position size (eval weakens but not reversed)
  - EXIT: Close position (eval reversal or max loss triggered)
  - REVERSE: Close and flip to opposite side (strong reversal signal)
  - DO_NOTHING: No action (insufficient signal or max risk reached)

Author: Trading-Stockfish
Version: 1.0.0
License: MIT
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engine.canonical_stack_validator import validate_official_mode_startup
from engine.canonical_validator import enforce_official_env
from engine.causal_evaluator import evaluate_state
from state.regime_engine import RegimeSignal

# Canonical state and regime-aware evaluation
from state.schema import MarketState as SchemaMarketState

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================


class PositionSide(Enum):
    """Trading position side"""

    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class TradingAction(Enum):
    """Discrete trading actions"""

    ENTER_SMALL = "ENTER_SMALL"
    ENTER_FULL = "ENTER_FULL"
    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    REVERSE = "REVERSE"
    DO_NOTHING = "DO_NOTHING"


class EvaluationZone(Enum):
    """Market evaluation conviction zones"""

    NO_TRADE = "NO_TRADE"  # |eval| < 0.2
    LOW_CONVICTION = "LOW_CONVICTION"  # 0.2 ≤ |eval| < 0.5
    MEDIUM_CONVICTION = "MEDIUM_CONVICTION"  # 0.5 ≤ |eval| < 0.8
    HIGH_CONVICTION = "HIGH_CONVICTION"  # |eval| ≥ 0.8


class VolatilityRegime(Enum):
    """Volatility regime classification"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class LiquidityRegime(Enum):
    """Liquidity regime classification"""

    ABUNDANT = "ABUNDANT"  # Easy execution
    NORMAL = "NORMAL"  # Standard conditions
    TIGHT = "TIGHT"  # Limited depth
    EXHAUSTING = "EXHAUSTING"  # Drying up


# Regime-conditioned thresholds (deterministic constants)
THRESHOLDS = {
    "VOL": {
        "LOW": {"enter": 0.15, "add": 0.35, "exit": 0.10},
        "NORMAL": {"enter": 0.20, "add": 0.45, "exit": 0.12},
        "HIGH": {"enter": 0.25, "add": 0.55, "exit": 0.15},
        "EXTREME": {"enter": 0.30, "add": 0.60, "exit": 0.20},
    },
    "LIQ": {
        "DEEP": {"enter": 0.18, "add": 0.40, "exit": 0.10},
        "NORMAL": {"enter": 0.22, "add": 0.48, "exit": 0.12},
        "THIN": {"enter": 0.26, "add": 0.55, "exit": 0.15},
        "FRAGILE": {"enter": 0.30, "add": 0.60, "exit": 0.20},
    },
    "MACRO": {
        "RISK_ON": {"enter": 0.18, "add": 0.42, "exit": 0.10},
        "RISK_OFF": {"enter": 0.22, "add": 0.48, "exit": 0.12},
        "EVENT": {"enter": 0.25, "add": 0.52, "exit": 0.14},
    },
}


from engine.causal_evaluator import evaluate_state

# Minimal core types (placeholder scaffolding)
from .types import Action as CoreAction
from .types import EvaluationOutput as CoreEvaluationOutput
from .types import MarketState as CoreMarketState

# REGIME-CONDITIONED POLICY (v5.0-B)

# ============================================================================


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_size(x: float) -> float:
    return clamp(x, 0.0, 1.0)


def combine_thresholds(
    vol_regime: str, liq_regime: str, macro_regime: str
) -> Dict[str, float]:
    vol_th = THRESHOLDS["VOL"].get(vol_regime, THRESHOLDS["VOL"]["NORMAL"])
    liq_th = THRESHOLDS["LIQ"].get(liq_regime, THRESHOLDS["LIQ"]["NORMAL"])
    mac_th = THRESHOLDS["MACRO"].get(macro_regime, THRESHOLDS["MACRO"]["RISK_OFF"])
    combined = {}
    for key in ["enter", "add", "exit"]:
        combined[key] = (vol_th[key] + liq_th[key] + mac_th[key]) / 3.0
    return combined


def apply_veto_rules(score: float, confidence: float) -> Optional[str]:
    """Deterministic veto: low confidence => FLAT."""
    if confidence < 0.15:
        return "FLAT"
    return None


def compute_target_size(
    score: float, confidence: float, state: SchemaMarketState, regime: RegimeSignal
) -> float:
    """Volatility-adjusted, liquidity-capped deterministic sizing."""
    base = abs(score) * confidence
    vol = max(float(state.volatility.realized_vol), 1e-6)
    vol_factor = 1.0 / (1.0 + vol)
    vol_shock_strength = float(
        getattr(state.volatility, "volatility_shock_strength", 0.0)
    )
    if getattr(state.volatility, "volatility_shock", False):
        vol_factor *= max(0.2, 1.0 - 0.6 * vol_shock_strength)
    depth = float(
        state.liquidity.cumulative_depth_bid + state.liquidity.cumulative_depth_ask
    )
    liq_cap = depth / 1_000_000.0
    size = base * vol_factor
    size = min(size, liq_cap)
    return normalize_size(size)


def _get_regime_from_state(state: SchemaMarketState) -> RegimeSignal:
    reg_raw = state.raw.get("regime") if isinstance(state.raw, dict) else None
    if reg_raw is None:
        raise ValueError("state.raw['regime'] missing for policy evaluation")
    if isinstance(reg_raw, RegimeSignal):
        return reg_raw
    if isinstance(reg_raw, dict):
        return RegimeSignal(**reg_raw)
    raise TypeError("state.raw['regime'] must be RegimeSignal or dict")


def select_action_regime(state: SchemaMarketState) -> Dict[str, Any]:
    """Regime-conditioned policy decision (v5.0-B)."""
    if not isinstance(state, SchemaMarketState):
        raise TypeError("state must be state.schema.MarketState")

    regime = _get_regime_from_state(state)
    thresholds = combine_thresholds(regime.vol, regime.liq, regime.macro)

    amd_tag = "NEUTRAL"
    amd_confidence = 0.0
    if hasattr(state, "amd") and state.amd is not None:
        amd_tag = getattr(state.amd, "amd_tag", "NEUTRAL")
        amd_confidence = float(getattr(state.amd, "amd_confidence", 0.0))
    elif isinstance(state.raw, dict):
        amd_tag = state.raw.get("amd_regime", state.raw.get("amd_tag", "NEUTRAL"))
        amd_confidence = float(state.raw.get("amd_confidence", 0.0) or 0.0)

    eval_result = evaluate_state(state)
    score = float(eval_result["score"])
    confidence = float(eval_result["confidence"])

    veto = apply_veto_rules(score, confidence)
    if veto:
        return {
            "action": veto,
            "target_size": 0.0,
            "score": score,
            "confidence": confidence,
            "regime": regime.to_dict(),
            "version": "v5.0-B",
        }

    target_size = compute_target_size(score, confidence, state, regime)

    s_abs = abs(score)
    if s_abs < thresholds["exit"]:
        action = "EXIT"
        target_size = 0.0
    elif s_abs < thresholds["enter"]:
        action = "REDUCE"
        target_size = min(target_size, 0.25)
    elif s_abs < thresholds["add"]:
        action = "ENTER_LONG" if score > 0 else "ENTER_SHORT"
    else:
        action = "ADD"

    if amd_tag == "MANIPULATION":
        action = "FLAT"
        target_size = 0.0
        confidence *= 0.8
    elif amd_tag == "ACCUMULATION" and action == "ENTER_SHORT":
        action = "REDUCE"
        target_size = min(target_size, 0.1)
    elif amd_tag == "DISTRIBUTION" and action == "ENTER_LONG":
        action = "REDUCE"
        target_size = min(target_size, 0.1)

    vol_shock = bool(getattr(state.volatility, "volatility_shock", False))
    vol_shock_strength = float(
        getattr(state.volatility, "volatility_shock_strength", 0.0)
    )
    if vol_shock:
        target_size *= max(0.2, 1.0 - 0.7 * vol_shock_strength)
        confidence *= max(0.6, 1.0 - 0.4 * vol_shock_strength)
        if vol_shock_strength > 0.8 and action not in ["EXIT", "FLAT", "REDUCE"]:
            action = "FLAT"
            target_size = 0.0

    return {
        "action": action,
        "target_size": target_size,
        "score": score,
        "confidence": confidence,
        "regime": regime.to_dict(),
        "version": "v5.0-B",
        "thresholds": thresholds,
        "amd_regime": amd_tag,
        "amd_confidence": amd_confidence,
    }


# ---------------------------------------------------------------------------
# Minimal placeholder policy (core scaffolding)
# ---------------------------------------------------------------------------


def select_action(state: CoreMarketState, eval_out: CoreEvaluationOutput) -> CoreAction:
    """Policy engine v1.1 (deterministic scaffolding).

    Rules:
    - Flat: score>0.6 → BUY, score<-0.6 → SELL
    - In position: if |score|<0.15 → CLOSE
    - Otherwise HOLD
    - Confidence veto: confidence<0.2 → HOLD with veto_reason
    - size fixed at 0.1
    """

    if not isinstance(state, CoreMarketState):
        raise TypeError("state must be engine.types.MarketState")
    if not isinstance(eval_out, CoreEvaluationOutput):
        raise TypeError("eval_out must be engine.types.EvaluationOutput")

    trend_reg = getattr(eval_out, "trend_regime", "")
    vol_reg = getattr(eval_out, "volatility_regime", "")
    liq_reg = getattr(eval_out, "liquidity_regime", "")
    macro_reg = getattr(eval_out, "macro_regime", "")

    logger.info(
        "[select_action] regimes trend=%s vol=%s liq=%s macro=%s",
        trend_reg,
        vol_reg,
        liq_reg,
        macro_reg,
    )

    score = eval_out.score
    action_type = "HOLD"
    size_fixed = 0.1
    size = state.position_size
    veto_reason = None

    if state.position_side == "flat":
        if score > 0.6:
            action_type = "BUY"
            size = size_fixed
        elif score < -0.6:
            action_type = "SELL"
            size = size_fixed
    else:
        if abs(score) < 0.15:
            action_type = "CLOSE"
            size = 0.0
        else:
            action_type = "HOLD"
            size = state.position_size

    # Soft regime constraints
    if trend_reg == "chop" and action_type in {"BUY", "SELL"}:
        action_type = "HOLD"
        veto_reason = "choppy_trend"
        size = state.position_size

    if vol_reg == "high" and action_type != "HOLD":
        size = 0.05

    if macro_reg == "risk_off" and action_type == "BUY":
        action_type = "HOLD"
        veto_reason = "macro_risk_off"
        size = state.position_size

    if eval_out.confidence < 0.2:
        action_type = "HOLD"
        veto_reason = "low_confidence"
        size = state.position_size

    return CoreAction(
        action_type=action_type,
        size=size,
        confidence=eval_out.confidence,
        veto_reason=veto_reason,
    )


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class PositionState:
    """Current position state"""

    side: PositionSide
    size: float  # Current position size (normalized 0-1)
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0  # In pips or normalized units
    unrealized_pnl_pct: float = 0.0  # Unrealized P&L as percentage
    max_adverse_excursion: float = 0.0  # Peak drawdown since entry
    max_favorable_excursion: float = 0.0  # Peak gain since entry
    bars_since_entry: int = 0
    bars_since_exit: int = 0  # For cooldown tracking


@dataclass
class RiskConfig:
    """Risk management configuration"""

    max_risk_per_trade: float = 0.01  # Max risk per trade (1% of equity)
    max_daily_loss: float = 0.03  # Max daily loss (3% of equity)
    max_daily_loss_triggered: bool = False  # Has daily loss limit been hit?
    max_position_size: float = 1.0  # Max normalized position size

    # Decision thresholds
    add_threshold: float = 0.6  # Min eval to allow adding to position
    reduce_threshold: float = 0.3  # Eval deterioration threshold
    exit_threshold: float = -0.2  # Eval reversal threshold to exit
    reverse_threshold: float = -0.5  # Strong reversal threshold

    # Confidence requirements
    min_confidence: float = 0.50  # Minimum confidence to take action
    min_confidence_add: float = 0.65  # Higher confidence for adding
    min_confidence_reduce: float = 0.40  # Lower confidence to reduce

    # Cooldown
    cooldown_bars: int = 2  # Bars to wait after EXIT or REVERSE

    # Regime-aware adjustments
    high_vol_size_reduction: float = 0.7  # Reduce size to 70% in high vol
    tight_liquidity_size_reduction: float = 0.6  # Reduce to 60% in tight liquidity

    # Risk control flags
    enable_reverse: bool = True  # Allow REVERSE action
    enable_add: bool = True  # Allow ADD action
    skip_on_max_daily_loss: bool = True  # Force DO_NOTHING if max daily loss hit


@dataclass
class ReasoningFactor:
    """Single reasoning factor for decision"""

    factor: str  # "Eval", "Confidence", "Volatility", etc.
    detail: str  # Explanation
    weight: float = 1.0  # Importance (0-1)


@dataclass
class PolicyDecision:
    """Output of policy engine decision"""

    action: TradingAction
    target_size: float  # Normalized target position size (0-1)
    confidence: float  # Confidence in this decision (0-1)
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: List[ReasoningFactor] = field(default_factory=list)

    # Session and Flow Context (v1.1.1)
    session_name: str = ""
    session_modifiers: Dict[str, float] = field(default_factory=dict)
    flow_signals: Dict[str, Any] = field(default_factory=dict)
    stop_run_detected: bool = False
    initiative_move_detected: bool = False
    level_reaction_score: float = 0.0

    # Regime Context (v2.1)
    regime_label: str = ""  # TREND, RANGE, REVERSAL
    regime_confidence: float = 0.0  # [0, 1] confidence in regime
    regime_adjustments: Dict[str, Any] = field(
        default_factory=dict
    )  # What was adjusted

    # Scenario Context (v2.3)
    scenario_alignment: float = 0.0  # [0, 1] how well scenarios align with decision
    scenario_bias: str = (
        ""  # bullish, bearish, symmetric, bullish_reversal, bearish_reversal
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "action": self.action.value,
            "target_size": self.target_size,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "session": self.session_name,
            "session_modifiers": self.session_modifiers or {},
            "flow_signals": self.flow_signals,
            "stop_run_detected": self.stop_run_detected,
            "initiative_move_detected": self.initiative_move_detected,
            "level_reaction_score": self.level_reaction_score,
            "regime_label": self.regime_label,
            "regime_confidence": self.regime_confidence,
            "regime_adjustments": self.regime_adjustments or {},
            "scenario_alignment": self.scenario_alignment,
            "scenario_bias": self.scenario_bias,
            "reasoning": [
                {"factor": r.factor, "detail": r.detail, "weight": r.weight}
                for r in self.reasoning
            ],
        }


# ============================================================================
# POLICY ENGINE
# ============================================================================


class PolicyEngine:
    """Deterministic, risk-aware trading decision engine"""

    def __init__(
        self,
        default_risk_config: Optional[RiskConfig] = None,
        verbose: bool = False,
        official_mode: bool = False,
    ):
        """Initialize PolicyEngine.

        Args:
            default_risk_config: Default RiskConfig (uses built-in if None)
            verbose: Print verbose logging
            official_mode: Strict deterministic mode (no randomness)
        """
        self.default_risk_config = default_risk_config or RiskConfig()
        self.verbose = verbose
        self.official_mode = official_mode

        if self.official_mode:
            enforce_official_env(True)
            validate_official_mode_startup(context="policy_engine", use_causal=True)

        logger.info("PolicyEngine initialized")
        if official_mode:
            logger.info("[OFFICIAL] Deterministic, rule-based policy mode")

    def _apply_session_adjustments(
        self, risk_config: RiskConfig, session_name: str, flow_signals: Dict[str, Any]
    ) -> RiskConfig:
        """
        Apply session-aware adjustments to risk configuration.

        Session-specific strategies:
        - RTH_OPEN: Increase entry thresholds, require higher confidence, reduce reversals
        - MIDDAY: Lower aggression, increase mean-reversion, widen cooldowns
        - POWER_HOUR: Allow stronger continuation, reduce hesitation on initiative moves
        - GLOBEX: Reduce size, avoid aggressive entries, require strong conviction
        - CLOSE: Allow flow persistence, tighten risk on new entries

        Args:
            risk_config: Base RiskConfig
            session_name: Current trading session
            flow_signals: Dict with flow context

        Returns:
            Adjusted RiskConfig for the session
        """
        # Create a copy to avoid modifying original
        adj = RiskConfig(
            **{
                k: getattr(risk_config, k)
                for k in [
                    "max_risk_per_trade",
                    "max_daily_loss",
                    "max_position_size",
                    "add_threshold",
                    "reduce_threshold",
                    "exit_threshold",
                    "reverse_threshold",
                    "min_confidence",
                    "min_confidence_add",
                    "min_confidence_reduce",
                    "cooldown_bars",
                    "high_vol_size_reduction",
                    "tight_liquidity_size_reduction",
                    "enable_reverse",
                    "enable_add",
                    "skip_on_max_daily_loss",
                ]
            }
        )
        adj.max_daily_loss_triggered = risk_config.max_daily_loss_triggered

        if session_name == "GLOBEX":
            # Overnight: reduce size, avoid aggressive entries
            adj.max_position_size *= 0.7
            adj.min_confidence = 0.70
            adj.enable_reverse = False  # No reversals overnight
            adj.enable_add = False  # No adds overnight
            adj.add_threshold = 0.8
            adj.reduce_threshold = 0.2

        elif session_name == "PREMARKET":
            # Pre-open: high uncertainty, increase thresholds
            adj.min_confidence = 0.65
            adj.add_threshold = 0.75
            adj.reduce_threshold = 0.25
            adj.cooldown_bars = 3  # Longer cooldown

        elif session_name == "RTH_OPEN":
            # Open: volume ramp-up, increase entry threshold
            adj.min_confidence = 0.60
            adj.add_threshold = 0.70
            adj.reduce_threshold = 0.30
            adj.enable_reverse = False  # No reversals at open
            adj.cooldown_bars = 2

        elif session_name == "MIDDAY":
            # Midday: lower aggression, mean-reversion, widen cooldowns
            adj.min_confidence = 0.55
            adj.add_threshold = 0.50  # Lower threshold for adds
            adj.reduce_threshold = 0.35
            adj.cooldown_bars = 3  # Wider cooldown
            adj.enable_reverse = True
            adj.enable_add = True

        elif session_name == "POWER_HOUR":
            # 3-4 PM: flow-driven, allow stronger continuation
            adj.min_confidence = 0.50
            adj.add_threshold = 0.45  # Lower for initiative moves
            adj.reduce_threshold = 0.25
            adj.reverse_threshold = -0.40  # Easier reversal
            adj.cooldown_bars = 1  # Faster turnaround
            adj.enable_reverse = True
            adj.enable_add = True

        elif session_name == "CLOSE":
            # Close: flow persistence, tighten risk on new entries
            adj.min_confidence = 0.58
            adj.add_threshold = 0.50
            adj.reduce_threshold = 0.30
            adj.max_position_size *= 0.9  # Slightly tighter
            adj.cooldown_bars = 1

        return adj

    def _create_decision(
        self,
        action: TradingAction,
        target_size: float,
        confidence: float,
        reasoning: List[ReasoningFactor],
        session_name: str = "",
        session_modifiers: Dict[str, float] = None,
        flow_signals: Dict[str, Any] = None,
        stop_run_detected: bool = False,
        initiative_move_detected: bool = False,
        level_reaction_score: float = 0.0,
        regime_label: str = "",
        regime_confidence: float = 0.0,
        regime_adjustments: Dict[str, Any] = None,
    ) -> PolicyDecision:
        """Create PolicyDecision with session/flow context and regime context (v2.1)."""
        return PolicyDecision(
            action=action,
            target_size=target_size,
            confidence=confidence,
            reasoning=reasoning,
            session_name=session_name,
            session_modifiers=session_modifiers or {},
            flow_signals=flow_signals or {},
            stop_run_detected=stop_run_detected,
            initiative_move_detected=initiative_move_detected,
            level_reaction_score=level_reaction_score,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            regime_adjustments=regime_adjustments or {},
        )

    def _apply_regime_conditioning(
        self,
        action: TradingAction,
        target_size: float,
        eval_score: float,
        regime_label: str,
        regime_confidence: float,
        position_state: PositionState,
        risk_config: RiskConfig,
    ) -> Tuple[TradingAction, float]:
        """
        Apply regime-conditioned modifications to trading action and sizing.

        TREND REGIME (v2.1):
        - Allow continuation entries
        - Reduce reversal attempts
        - Allow scaling in direction of trend
        - Favor position growth in trend direction

        RANGE REGIME (v2.1):
        - Avoid chasing breakouts
        - Prefer fade/mean-reversion entries
        - Reduce size on trend attempts
        - Favor small mean-reversion trades

        REVERSAL REGIME (v2.1):
        - Allow counter-trend entries after exhaustion
        - Require strong level reactions
        - Tighten stops (enforce via risk config reduction)
        - Reduce size until reversal confirmed

        Args:
            action: Base TradingAction
            target_size: Base target size [0, 1]
            eval_score: Current evaluation score [-1, 1]
            regime_label: TREND, RANGE, REVERSAL, or empty
            regime_confidence: [0, 1] confidence in regime classification
            position_state: Current position state
            risk_config: Current risk configuration

        Returns:
            Tuple of (adjusted_action, adjusted_target_size)
        """
        if not regime_label or regime_confidence < 0.3:
            return action, target_size

        adjusted_action = action
        adjusted_size = target_size
        regime_boost = regime_confidence  # Confidence acts as adjustment magnitude

        if regime_label == "TREND":
            # In TREND: Allow scaling in trend direction
            if (
                action == TradingAction.ENTER_SMALL
                or action == TradingAction.ENTER_FULL
            ):
                # Increase size allocation in TREND if eval is strong
                if abs(eval_score) > 0.5:
                    adjusted_size = min(target_size * (1.0 + 0.15 * regime_boost), 1.0)
                    self._log(
                        f"  [TREND] Scaling up entry size: {target_size:.2f} → {adjusted_size:.2f}"
                    )

            elif action == TradingAction.ADD:
                # More aggressive adding in TREND
                adjusted_size = min(target_size * (1.0 + 0.10 * regime_boost), 1.0)
                self._log(
                    f"  [TREND] Scaling up add size: {target_size:.2f} → {adjusted_size:.2f}"
                )

            elif action == TradingAction.REVERSE:
                # Less likely to reverse in TREND
                adjusted_action = TradingAction.REDUCE
                self._log(f"  [TREND] Converted REVERSE → REDUCE (trend active)")

        elif regime_label == "RANGE":
            # In RANGE: Reduce size, favor fades
            if (
                action == TradingAction.ENTER_SMALL
                or action == TradingAction.ENTER_FULL
            ):
                # Reduce size for breakout entries in RANGE
                if abs(eval_score) > 0.6:
                    adjusted_size = max(target_size * (1.0 - 0.20 * regime_boost), 0.1)
                    adjusted_action = TradingAction.ENTER_SMALL  # Force small entries
                    self._log(
                        f"  [RANGE] Reducing breakout entry size: {target_size:.2f} → {adjusted_size:.2f}"
                    )

            elif action == TradingAction.ADD:
                # Don't add to breakout positions in RANGE
                if abs(eval_score) > 0.5:
                    adjusted_action = TradingAction.HOLD
                    self._log(f"  [RANGE] Blocked ADD on breakout signal")

            elif action == TradingAction.REVERSE:
                # Favor reversals in RANGE
                adjusted_size = min(target_size * 1.1, 1.0)
                self._log(
                    f"  [RANGE] Favoring REVERSE: {target_size:.2f} → {adjusted_size:.2f}"
                )

        elif regime_label == "REVERSAL":
            # In REVERSAL: Reduce size, tighten controls
            size_reduction = 1.0 - (0.25 * regime_boost)  # Reduce up to 25%

            if action in [TradingAction.ENTER_SMALL, TradingAction.ENTER_FULL]:
                adjusted_size = target_size * size_reduction
                adjusted_action = TradingAction.ENTER_SMALL  # Prefer small entries
                self._log(
                    f"  [REVERSAL] Reduced entry size: {target_size:.2f} → {adjusted_size:.2f}"
                )

            elif action == TradingAction.ADD:
                # Avoid adding in REVERSAL until confirmed
                adjusted_action = TradingAction.HOLD
                self._log(f"  [REVERSAL] Blocked ADD (reversal in progress)")

            elif action == TradingAction.EXIT:
                # Exit faster in REVERSAL
                adjusted_size = target_size  # Already exiting
                self._log(f"  [REVERSAL] Maintaining EXIT (tighter reversal control)")

        return adjusted_action, adjusted_size

    def _apply_scenario_conditioning(
        self,
        action: TradingAction,
        target_size: float,
        eval_score: float,
        scenario_result: Any,  # ScenarioResult from v2.2
        position_state: PositionState,
    ) -> Tuple[TradingAction, float, float, str]:
        """
        Apply scenario-conditioned modifications to trading action and sizing (v2.3).

        Scenario conditioning uses probability distributions to shape decisions:

        TREND REGIME scenarios:
        - If UP_MOVE prob > 55% (bullish): Allow continuation entries, allow scaling (ADD)
        - If DOWN_MOVE prob > 55% (bearish): Reduce long entries, allow reversals only with strong exhaustion

        RANGE REGIME scenarios:
        - If CHOP prob > 50%: Prefer ENTER_SMALL, block ADD, avoid chasing breakouts
        - If UP/DOWN imbalance < 10%: Treat as mean-reversion environment

        REVERSAL REGIME scenarios:
        - If DOWN_MOVE prob > 60% after strong UP eval: Allow reversal entries
        - If UP_MOVE prob > 60% after strong DOWN eval: Allow reversal entries
        - Reduce size until reversal confirmed

        Args:
            action: Original trading action
            target_size: Original target position size
            eval_score: Evaluation score [-1, +1]
            scenario_result: ScenarioResult with probability distributions
            position_state: Current position state

        Returns:
            (adjusted_action, adjusted_size, scenario_alignment, scenario_bias)
        """
        if not scenario_result:
            return action, target_size, 0.0, ""

        adjusted_action = action
        adjusted_size = target_size
        scenario_alignment = (
            scenario_result.regime_alignment
            if hasattr(scenario_result, "regime_alignment")
            else 0.0
        )
        scenario_bias = (
            scenario_result.scenario_bias
            if hasattr(scenario_result, "scenario_bias")
            else ""
        )

        regime_label = (
            scenario_result.regime_label
            if hasattr(scenario_result, "regime_label")
            else ""
        )
        prob_up = scenario_result.probability_up
        prob_down = scenario_result.probability_down
        prob_chop = scenario_result.probability_chop

        # TREND regime: Continuation-biased scenarios
        if regime_label == "TREND":
            if prob_up > 0.55 and eval_score > 0.3:
                # Strong UP probability + bullish eval: Allow continuation and scaling
                if action == TradingAction.ENTER_SMALL:
                    adjusted_action = TradingAction.ENTER_FULL  # Upgrade entry
                    adjusted_size = min(target_size * 1.2, 1.0)
                    self._log(
                        f"  [TREND-SCENARIO] Strong UP prob {prob_up:.1%}: upgraded ENTER_SMALL → ENTER_FULL"
                    )
                elif (
                    action == TradingAction.HOLD
                    and position_state.side == PositionSide.LONG
                ):
                    adjusted_action = TradingAction.ADD  # Allow scaling
                    adjusted_size = min(position_state.size * 1.15, 1.0)
                    self._log(
                        f"  [TREND-SCENARIO] Strong UP prob {prob_up:.1%}: converted HOLD → ADD"
                    )

            elif prob_down > 0.55 and eval_score < -0.3:
                # Strong DOWN probability + bearish eval: Reduce long entries, allow reversals
                if (
                    action == TradingAction.ENTER_SMALL
                    or action == TradingAction.ENTER_FULL
                ):
                    adjusted_action = TradingAction.DO_NOTHING
                    adjusted_size = 0.0
                    self._log(
                        f"  [TREND-SCENARIO] Strong DOWN prob {prob_down:.1%}: blocked entry in bearish trend"
                    )
                elif action == TradingAction.ADD:
                    adjusted_action = TradingAction.REDUCE
                    adjusted_size = position_state.size * 0.5
                    self._log(
                        f"  [TREND-SCENARIO] Strong DOWN prob {prob_down:.1%}: converted ADD → REDUCE"
                    )

        # RANGE regime: Symmetric with elevated chop
        elif regime_label == "RANGE":
            if prob_chop > 0.50:
                # Chop dominates: Prefer small entries, block adds, avoid breakouts
                if action in [TradingAction.ENTER_SMALL, TradingAction.ENTER_FULL]:
                    adjusted_action = TradingAction.ENTER_SMALL
                    adjusted_size = target_size * 0.6  # Significant size reduction
                    self._log(
                        f"  [RANGE-SCENARIO] High chop prob {prob_chop:.1%}: reduced entry size"
                    )

                elif action == TradingAction.ADD:
                    adjusted_action = TradingAction.HOLD
                    self._log(
                        f"  [RANGE-SCENARIO] High chop prob {prob_chop:.1%}: blocked ADD in chop"
                    )

            # Check for symmetric up/down (mean-reversion signal)
            imbalance = abs(prob_up - prob_down)
            if imbalance < 0.10 and action == TradingAction.ENTER_SMALL:
                # Nearly symmetric: Treat as mean-reversion trade
                self._log(
                    f"  [RANGE-SCENARIO] Symmetric probabilities (imbalance {imbalance:.1%}): mean-reversion setup"
                )
                adjusted_size = (
                    target_size * 0.9
                )  # Slight size increase for mean-reversion confidence
                scenario_alignment = min(scenario_alignment + 0.1, 1.0)

        # REVERSAL regime: Reversal-biased scenarios
        elif regime_label == "REVERSAL":
            # Strong DOWN probability after UP eval: Allow reversal entries
            if prob_down > 0.60 and eval_score > 0.5:
                if action == TradingAction.HOLD or action == TradingAction.REDUCE:
                    adjusted_action = TradingAction.REVERSE
                    adjusted_size = target_size * 0.8  # Reduced size for reversal entry
                    self._log(
                        f"  [REVERSAL-SCENARIO] Strong DOWN prob {prob_down:.1%} after UP eval: REVERSE allowed"
                    )

            # Strong UP probability after DOWN eval: Allow reversal entries
            elif prob_up > 0.60 and eval_score < -0.5:
                if action == TradingAction.HOLD or action == TradingAction.REDUCE:
                    adjusted_action = TradingAction.REVERSE
                    adjusted_size = target_size * 0.8  # Reduced size for reversal entry
                    self._log(
                        f"  [REVERSAL-SCENARIO] Strong UP prob {prob_up:.1%} after DOWN eval: REVERSE allowed"
                    )

            # All actions in REVERSAL: Reduce size until confirmed
            else:
                adjusted_size = target_size * 0.7  # 30% size reduction
                self._log(
                    f"  [REVERSAL-SCENARIO] Reduced size 30% pending reversal confirmation"
                )

        return adjusted_action, adjusted_size, scenario_alignment, scenario_bias

    def _log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            logger.info(message)

    def decide_action(
        self,
        market_state: Any,  # MarketState with all 8 components
        eval_result: Dict[str, Any],  # CausalEvaluator output
        position_state: PositionState,
        risk_config: Optional[RiskConfig] = None,
        daily_loss_pct: float = 0.0,  # Current daily loss as percentage
    ) -> PolicyDecision:
        """Make a trading decision based on market state and position.

        Args:
            market_state: MarketState with all 8 causal components
            eval_result: Dict from CausalEvaluator.evaluate() with:
                - eval_score: float [-1, +1]
                - confidence: float [0, 1]
                - scoring_factors: List of factor breakdowns
                - session_name: Trading session (v1.1.1)
                - session_modifiers: Dict with vol/liq/risk scales (v1.1.1)
                - flow_signals: Dict with flow context (v1.1.1)
            position_state: Current PositionState
            risk_config: RiskConfig (uses default if None)
            daily_loss_pct: Current daily loss percentage

        Returns:
            PolicyDecision with action, target_size, and reasoning
        """
        risk_config = risk_config or self.default_risk_config
        risk_config.max_daily_loss_triggered = (
            daily_loss_pct >= risk_config.max_daily_loss
        )

        reasoning = []

        # Extract evaluation metrics
        eval_score = eval_result.get("eval_score", 0.0)
        confidence = eval_result.get("confidence", 0.0)

        # Extract session/flow context (v1.1.1)
        session_name = eval_result.get("session", "")
        session_modifiers = eval_result.get("session_modifiers", {})
        flow_signals = eval_result.get("flow_signals", {})
        stop_run_detected = eval_result.get("stop_run_detected", False)
        initiative_move_detected = eval_result.get("initiative_move_detected", False)
        level_reaction_score = eval_result.get("level_reaction_score", 0.0)

        # Extract regime context (v2.1)
        regime_label = eval_result.get("regime_label", "")
        regime_confidence = eval_result.get("regime_confidence", 0.0)
        regime_adjustments = eval_result.get("regime_adjustments", {})

        # Trend/swing context (from canonical regime signal if present)
        trend_direction = ""
        swing_structure = ""
        trend_strength = 0.0
        swing_high = 0.0
        swing_low = 0.0
        if market_state is not None:
            try:
                reg_raw = getattr(market_state, "raw", {}).get("regime")
            except Exception:
                reg_raw = None
            if reg_raw is not None:
                if isinstance(reg_raw, RegimeSignal):
                    reg_sig = reg_raw
                elif isinstance(reg_raw, dict):
                    reg_sig = RegimeSignal(**reg_raw)
                else:
                    reg_sig = None
                if reg_sig:
                    swing_structure = getattr(reg_sig, "swing_structure", "") or ""
                    trend_direction = getattr(reg_sig, "trend_direction", "") or ""
                    trend_strength = float(
                        getattr(reg_sig, "trend_strength", 0.0) or 0.0
                    )
                    swing_high = float(getattr(reg_sig, "swing_high", 0.0) or 0.0)
                    swing_low = float(getattr(reg_sig, "swing_low", 0.0) or 0.0)
            # Fallback to state-level fields if present
            swing_structure = (
                getattr(market_state, "swing_structure", swing_structure)
                or swing_structure
            )
            trend_direction = (
                getattr(market_state, "trend_direction", trend_direction)
                or trend_direction
            )
            trend_strength = float(
                getattr(market_state, "trend_strength", trend_strength)
                or trend_strength
            )
            swing_high = float(
                getattr(market_state, "swing_high", swing_high) or swing_high
            )
            swing_low = float(
                getattr(market_state, "swing_low", swing_low) or swing_low
            )

        self._log(
            f"[{session_name}] Deciding action for eval={eval_score:.3f}, conf={confidence:.2f}, position={position_state.side.value}"
        )

        # Apply session-aware adjustments to risk config
        adjusted_risk_config = self._apply_session_adjustments(
            risk_config, session_name, flow_signals
        )

        # 1. Check hard risk constraints
        if (
            adjusted_risk_config.skip_on_max_daily_loss
            and adjusted_risk_config.max_daily_loss_triggered
        ):
            reasoning.append(
                ReasoningFactor(
                    "RiskControl",
                    f"Daily loss {daily_loss_pct:.1%} exceeds max {adjusted_risk_config.max_daily_loss:.1%}",
                    weight=1.0,
                )
            )
            return self._create_decision(
                action=TradingAction.DO_NOTHING,
                target_size=0.0,
                confidence=confidence,
                reasoning=reasoning,
                session_name=session_name,
                session_modifiers=session_modifiers,
                flow_signals=flow_signals,
                stop_run_detected=stop_run_detected,
                initiative_move_detected=initiative_move_detected,
                level_reaction_score=level_reaction_score,
            )

        # 2. Check stop-run avoidance (v1.1.1)
        if stop_run_detected:
            reasoning.append(
                ReasoningFactor(
                    "FlowContext",
                    "Stop-run detected - avoiding chasing, preferring fades",
                    weight=0.9,
                )
            )
            # In stop-run, only enter on strong reversals or with fade logic
            if eval_score > 0.3 or (eval_score > -0.5 and confidence < 0.60):
                # Don't chase into stop-runs
                self._log(f"  → Avoiding entry due to stop-run detection")
                return self._create_decision(
                    action=TradingAction.DO_NOTHING,
                    target_size=position_state.size,
                    confidence=confidence,
                    reasoning=reasoning,
                    session_name=session_name,
                    session_modifiers=session_modifiers,
                    flow_signals=flow_signals,
                    stop_run_detected=stop_run_detected,
                    initiative_move_detected=initiative_move_detected,
                    level_reaction_score=level_reaction_score,
                )

        # 3. Check minimum confidence requirement (with session adjustment)
        if confidence < adjusted_risk_config.min_confidence:
            reasoning.append(
                ReasoningFactor(
                    "Confidence",
                    f"Confidence {confidence:.2f} below minimum {adjusted_risk_config.min_confidence:.2f} (session-adjusted)",
                    weight=1.0,
                )
            )
            self._log(f"  → DO_NOTHING (low confidence)")
            return self._create_decision(
                action=TradingAction.DO_NOTHING,
                target_size=position_state.size,
                confidence=confidence,
                reasoning=reasoning,
                session_name=session_name,
                session_modifiers=session_modifiers,
                flow_signals=flow_signals,
                stop_run_detected=stop_run_detected,
                initiative_move_detected=initiative_move_detected,
                level_reaction_score=level_reaction_score,
            )

        # 4. Determine evaluation zone
        eval_zone = self._get_evaluation_zone(eval_score)
        reasoning.append(
            ReasoningFactor(
                "EvalZone",
                f"Evaluation in {eval_zone.value} zone (score={eval_score:.3f})",
                weight=0.8,
            )
        )

        # 5. Get volatility and liquidity regimes
        vol_regime = self._get_volatility_regime(market_state)
        liq_regime = self._get_liquidity_regime(market_state)

        reasoning.append(
            ReasoningFactor("Volatility", f"Regime: {vol_regime.value}", weight=0.6)
        )
        reasoning.append(
            ReasoningFactor("Liquidity", f"Regime: {liq_regime.value}", weight=0.6)
        )

        # 6. Main decision logic based on current position
        if position_state.side == PositionSide.FLAT:
            decision = self._decide_entry(
                eval_score,
                confidence,
                eval_zone,
                vol_regime,
                liq_regime,
                adjusted_risk_config,
                reasoning,
                initiative_move_detected,
                level_reaction_score,
                session_name,
            )
        elif position_state.side == PositionSide.LONG:
            decision = self._decide_long(
                eval_score,
                confidence,
                eval_zone,
                vol_regime,
                liq_regime,
                position_state,
                adjusted_risk_config,
                reasoning,
                stop_run_detected,
                initiative_move_detected,
                session_name,
            )
        else:  # SHORT
            decision = self._decide_short(
                eval_score,
                confidence,
                eval_zone,
                vol_regime,
                liq_regime,
                position_state,
                adjusted_risk_config,
                reasoning,
                stop_run_detected,
                initiative_move_detected,
                session_name,
            )

        # 7. Apply VWAP distance caution (v1.1.1)
        vwap_distance = (
            flow_signals.get("vwap_distance_pct", 0.0) if flow_signals else 0.0
        )
        if abs(vwap_distance) > 2.0 and decision.action in [
            TradingAction.ENTER_SMALL,
            TradingAction.ENTER_FULL,
        ]:
            reasoning.append(
                ReasoningFactor(
                    "FlowContext",
                    f"VWAP distance {vwap_distance:.2f}% extreme - reducing size",
                    weight=0.7,
                )
            )
            decision.target_size *= 0.8  # Reduce size

        # 8. Apply round-level caution (v1.1.1)
        round_proximity = (
            flow_signals.get("round_level_proximity", None) if flow_signals else None
        )
        if (
            round_proximity is not None
            and round_proximity > 0.8
            and decision.action in [TradingAction.ENTER_SMALL, TradingAction.ENTER_FULL]
        ):
            reasoning.append(
                ReasoningFactor(
                    "FlowContext",
                    f"Near round level (proximity {round_proximity:.2f}) - increasing caution",
                    weight=0.7,
                )
            )
            # Increase stop width implicitly by reducing conviction
            decision.confidence *= 0.9

        # 9. Enforce cooldown
        if (
            position_state.bars_since_exit < adjusted_risk_config.cooldown_bars
            and position_state.bars_since_exit > 0
        ):
            if decision.action in [
                TradingAction.ENTER_SMALL,
                TradingAction.ENTER_FULL,
                TradingAction.REVERSE,
            ]:
                reasoning.append(
                    ReasoningFactor(
                        "Cooldown",
                        f"In cooldown: {position_state.bars_since_exit}/{adjusted_risk_config.cooldown_bars} bars",
                        weight=1.0,
                    )
                )
                decision.action = TradingAction.DO_NOTHING
                decision.target_size = 0.0

        # 10. Trend-structure guard: reduce size on counter-trend attempts
        if trend_direction and trend_direction != "RANGE" and trend_strength > 0.1:
            counter_trend = (trend_direction == "UP" and eval_score < 0) or (
                trend_direction == "DOWN" and eval_score > 0
            )
            if counter_trend:
                reasoning.append(
                    ReasoningFactor(
                        "TrendGuard",
                        f"Counter-trend vs swing structure {swing_structure or trend_direction}",
                        weight=0.9,
                    )
                )
                decision.target_size *= 0.5
                if decision.action == TradingAction.ENTER_FULL:
                    decision.action = TradingAction.ENTER_SMALL
                elif decision.action == TradingAction.ADD:
                    decision.action = TradingAction.REDUCE

        # Add session/flow context to decision
        decision.session_name = session_name
        decision.session_modifiers = session_modifiers
        decision.flow_signals = flow_signals
        decision.stop_run_detected = stop_run_detected
        decision.initiative_move_detected = initiative_move_detected
        decision.level_reaction_score = level_reaction_score
        decision.reasoning = reasoning

        # Apply regime conditioning (v2.1)
        decision.regime_label = regime_label
        decision.regime_confidence = regime_confidence
        if regime_label and regime_confidence >= 0.3:
            base_action = decision.action
            base_size = decision.target_size
            adjusted_action, adjusted_size = self._apply_regime_conditioning(
                decision.action,
                decision.target_size,
                eval_score,
                regime_label,
                regime_confidence,
                position_state,
                adjusted_risk_config,
            )
            decision.action = adjusted_action
            decision.target_size = adjusted_size
            decision.regime_adjustments = {
                "base_action": base_action.value,
                "adjusted_action": adjusted_action.value,
                "size_adjustment": adjusted_size - base_size,
            }

        # Apply scenario conditioning (v2.3)
        scenario_result = eval_result.get("scenario_result")
        if scenario_result:
            (
                adjusted_action,
                adjusted_size,
                scenario_alignment,
                scenario_bias,
            ) = self._apply_scenario_conditioning(
                decision.action,
                decision.target_size,
                eval_score,
                scenario_result,
                position_state,
            )
            decision.action = adjusted_action
            decision.target_size = adjusted_size
            decision.scenario_alignment = scenario_alignment
            decision.scenario_bias = scenario_bias

        self._log(
            f"  → {decision.action.value} (target_size={decision.target_size:.2f}, session={session_name}, regime={regime_label})"
        )

        # --- GovernanceEngine integration (v4.0‑E) ---
        try:
            from engine.governance_engine import GovernanceEngine
        except ImportError:
            GovernanceEngine = None
        if GovernanceEngine:
            governance_engine = GovernanceEngine()
            governance_decision = governance_engine.decide(
                market_state,
                eval_result,
                {"action": decision.action.value, "size": decision.target_size},
                {"unrealized_pnl": getattr(position_state, "unrealized_pnl", 0.0)},
            )
            if not governance_decision.approved:
                decision.action = (
                    TradingAction.FLAT
                    if governance_decision.adjusted_action is None
                    else TradingAction[governance_decision.adjusted_action]
                )
                decision.target_size = (
                    0.0
                    if governance_decision.adjusted_size is None
                    else governance_decision.adjusted_size
                )
                reasoning.append(
                    ReasoningFactor(
                        "Governance",
                        f"VETO: {governance_decision.reason}",
                        weight=1.0,
                    )
                )
            elif governance_decision.adjusted_size is not None:
                decision.target_size = governance_decision.adjusted_size
                reasoning.append(
                    ReasoningFactor(
                        "Governance",
                        f"SIZE_ADJUSTED: {governance_decision.reason}",
                        weight=0.8,
                    )
                )
            if governance_decision.adjusted_action is not None:
                decision.action = TradingAction[governance_decision.adjusted_action]
        return decision

    def _decide_entry(
        self,
        eval_score: float,
        confidence: float,
        eval_zone: EvaluationZone,
        vol_regime: VolatilityRegime,
        liq_regime: LiquidityRegime,
        risk_config: RiskConfig,
        reasoning: List[ReasoningFactor],
        initiative_move_detected: bool = False,
        level_reaction_score: float = 0.0,
        session_name: str = "",
    ) -> PolicyDecision:
        """Decide entry action when FLAT"""

        # Check for initiative moves in POWER_HOUR (reduce hesitation)
        if (
            initiative_move_detected
            and session_name == "POWER_HOUR"
            and eval_score > 0.2
        ):
            reasoning.append(
                ReasoningFactor(
                    "FlowContext",
                    "Initiative move detected in POWER_HOUR - allowing entry",
                    weight=0.9,
                )
            )
            # Allow lower threshold entry on initiative
            size = (
                self._compute_position_size(
                    eval_score, confidence, vol_regime, liq_regime, risk_config
                )
                * 0.8
            )
            return PolicyDecision(
                action=TradingAction.ENTER_SMALL,
                target_size=size,
                confidence=confidence,
                reasoning=reasoning,
            )

        if eval_zone == EvaluationZone.NO_TRADE:
            reasoning.append(
                ReasoningFactor(
                    "Decision", "No trade zone - insufficient signal", weight=1.0
                )
            )
            return PolicyDecision(
                action=TradingAction.DO_NOTHING,
                target_size=0.0,
                confidence=confidence,
                reasoning=reasoning,
            )

        if eval_zone == EvaluationZone.LOW_CONVICTION:
            # Small entry only in excellent liquidity
            if liq_regime == LiquidityRegime.ABUNDANT:
                size = self._compute_position_size(
                    eval_score, confidence, vol_regime, liq_regime, risk_config
                )
                reasoning.append(
                    ReasoningFactor(
                        "Decision",
                        "Low conviction entry with excellent liquidity",
                        weight=1.0,
                    )
                )
                return PolicyDecision(
                    action=TradingAction.ENTER_SMALL,
                    target_size=size * 0.5,  # Half normal size
                    confidence=confidence,
                    reasoning=reasoning,
                )
            else:
                reasoning.append(
                    ReasoningFactor(
                        "Decision",
                        "Low conviction + poor liquidity - no entry",
                        weight=1.0,
                    )
                )
                return PolicyDecision(
                    action=TradingAction.DO_NOTHING,
                    target_size=0.0,
                    confidence=confidence,
                    reasoning=reasoning,
                )

        # MEDIUM or HIGH conviction entry
        size = self._compute_position_size(
            eval_score, confidence, vol_regime, liq_regime, risk_config
        )

        # Choose between ENTER_SMALL and ENTER_FULL
        if eval_zone == EvaluationZone.MEDIUM_CONVICTION:
            action = TradingAction.ENTER_SMALL
            size *= 0.7
        else:  # HIGH_CONVICTION
            action = TradingAction.ENTER_FULL

        reasoning.append(
            ReasoningFactor(
                "Decision", f"{eval_zone.value} entry - opening position", weight=1.0
            )
        )

        return PolicyDecision(
            action=action, target_size=size, confidence=confidence, reasoning=reasoning
        )

    def _decide_long(
        self,
        eval_score: float,
        confidence: float,
        eval_zone: EvaluationZone,
        vol_regime: VolatilityRegime,
        liq_regime: LiquidityRegime,
        position_state: PositionState,
        risk_config: RiskConfig,
        reasoning: List[ReasoningFactor],
        stop_run_detected: bool = False,
        initiative_move_detected: bool = False,
        session_name: str = "",
    ) -> PolicyDecision:
        """Decide action when in LONG position"""

        # Check for reversal signals (eval goes negative)
        if eval_score < risk_config.reverse_threshold and risk_config.enable_reverse:
            reasoning.append(
                ReasoningFactor(
                    "Decision",
                    f"Strong reversal signal (eval={eval_score:.3f} < {risk_config.reverse_threshold:.3f})",
                    weight=1.0,
                )
            )
            # Only reverse if liquidity allows
            if liq_regime != LiquidityRegime.EXHAUSTING:
                return PolicyDecision(
                    action=TradingAction.REVERSE,
                    target_size=0.0,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            else:
                # Prefer EXIT in poor liquidity
                return PolicyDecision(
                    action=TradingAction.EXIT,
                    target_size=0.0,
                    confidence=confidence,
                    reasoning=reasoning,
                )

        # Check for exit signals (eval goes slightly negative)
        if (
            eval_score < risk_config.exit_threshold
            and confidence > risk_config.min_confidence_reduce
        ):
            reasoning.append(
                ReasoningFactor(
                    "Decision",
                    f"Exit signal (eval={eval_score:.3f} < {risk_config.exit_threshold:.3f})",
                    weight=1.0,
                )
            )
            return PolicyDecision(
                action=TradingAction.EXIT,
                target_size=0.0,
                confidence=confidence,
                reasoning=reasoning,
            )

        # Check for add signals (eval strengthens and confidence high)
        if (
            eval_score > risk_config.add_threshold
            and confidence > risk_config.min_confidence_add
            and risk_config.enable_add
        ):
            size = self._compute_position_size(
                eval_score, confidence, vol_regime, liq_regime, risk_config
            )
            reasoning.append(
                ReasoningFactor(
                    "Decision",
                    f"Adding to long position (eval={eval_score:.3f} > {risk_config.add_threshold:.3f})",
                    weight=1.0,
                )
            )
            return PolicyDecision(
                action=TradingAction.ADD,
                target_size=min(
                    position_state.size + size * 0.5, risk_config.max_position_size
                ),
                confidence=confidence,
                reasoning=reasoning,
            )

        # Check for reduce signals (eval weakens)
        if (
            eval_score < risk_config.reduce_threshold
            and confidence > risk_config.min_confidence_reduce
        ):
            reasoning.append(
                ReasoningFactor(
                    "Decision",
                    f"Reducing long position (eval={eval_score:.3f} deteriorating)",
                    weight=1.0,
                )
            )
            return PolicyDecision(
                action=TradingAction.REDUCE,
                target_size=max(position_state.size * 0.7, 0.0),
                confidence=confidence,
                reasoning=reasoning,
            )

        # HOLD current position
        reasoning.append(
            ReasoningFactor(
                "Decision",
                f"Holding long position (eval={eval_score:.3f} stable)",
                weight=1.0,
            )
        )
        return PolicyDecision(
            action=TradingAction.HOLD,
            target_size=position_state.size,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _decide_short(
        self,
        eval_score: float,
        confidence: float,
        eval_zone: EvaluationZone,
        vol_regime: VolatilityRegime,
        liq_regime: LiquidityRegime,
        position_state: PositionState,
        risk_config: RiskConfig,
        reasoning: List[ReasoningFactor],
        stop_run_detected: bool = False,
        initiative_move_detected: bool = False,
        session_name: str = "",
    ) -> PolicyDecision:
        """Decide action when in SHORT position (mirror logic of LONG)"""

        # Check for reversal signals (eval goes positive)
        if eval_score > -risk_config.reverse_threshold and risk_config.enable_reverse:
            reasoning.append(
                ReasoningFactor(
                    "Decision",
                    f"Strong reversal signal (eval={eval_score:.3f} > {-risk_config.reverse_threshold:.3f})",
                    weight=1.0,
                )
            )
            if liq_regime != LiquidityRegime.EXHAUSTING:
                return PolicyDecision(
                    action=TradingAction.REVERSE,
                    target_size=0.0,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            else:
                return PolicyDecision(
                    action=TradingAction.EXIT,
                    target_size=0.0,
                    confidence=confidence,
                    reasoning=reasoning,
                )

        # Check for exit signals (eval goes slightly positive)
        if (
            eval_score > -risk_config.exit_threshold
            and confidence > risk_config.min_confidence_reduce
        ):
            reasoning.append(
                ReasoningFactor(
                    "Decision",
                    f"Exit signal (eval={eval_score:.3f} > {-risk_config.exit_threshold:.3f})",
                    weight=1.0,
                )
            )
            return PolicyDecision(
                action=TradingAction.EXIT,
                target_size=0.0,
                confidence=confidence,
                reasoning=reasoning,
            )

        # Check for add signals (eval weakens and confidence high)
        if (
            eval_score < -risk_config.add_threshold
            and confidence > risk_config.min_confidence_add
            and risk_config.enable_add
        ):
            size = self._compute_position_size(
                eval_score, confidence, vol_regime, liq_regime, risk_config
            )
            reasoning.append(
                ReasoningFactor(
                    "Decision",
                    f"Adding to short position (eval={eval_score:.3f} < {-risk_config.add_threshold:.3f})",
                    weight=1.0,
                )
            )
            return PolicyDecision(
                action=TradingAction.ADD,
                target_size=min(
                    position_state.size + size * 0.5, risk_config.max_position_size
                ),
                confidence=confidence,
                reasoning=reasoning,
            )

        # Check for reduce signals (eval strengthens)
        if (
            eval_score > -risk_config.reduce_threshold
            and confidence > risk_config.min_confidence_reduce
        ):
            reasoning.append(
                ReasoningFactor(
                    "Decision",
                    f"Reducing short position (eval={eval_score:.3f} improving)",
                    weight=1.0,
                )
            )
            return PolicyDecision(
                action=TradingAction.REDUCE,
                target_size=max(position_state.size * 0.7, 0.0),
                confidence=confidence,
                reasoning=reasoning,
            )

        # HOLD current position
        reasoning.append(
            ReasoningFactor(
                "Decision",
                f"Holding short position (eval={eval_score:.3f} stable)",
                weight=1.0,
            )
        )
        return PolicyDecision(
            action=TradingAction.HOLD,
            target_size=position_state.size,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _compute_position_size(
        self,
        eval_score: float,
        confidence: float,
        vol_regime: VolatilityRegime,
        liq_regime: LiquidityRegime,
        risk_config: RiskConfig,
    ) -> float:
        """Compute risk-aware position size.

        Returns:
            Normalized position size (0 to max_position_size)
        """
        # Base size from conviction
        base_size = min(abs(eval_score), 1.0)  # Use magnitude

        # Confidence adjustment
        size = base_size * confidence

        # Volatility adjustment
        if vol_regime == VolatilityRegime.LOW:
            vol_multiplier = 1.0
        elif vol_regime == VolatilityRegime.MEDIUM:
            vol_multiplier = 1.0
        elif vol_regime == VolatilityRegime.HIGH:
            vol_multiplier = risk_config.high_vol_size_reduction
        else:  # EXTREME
            vol_multiplier = 0.5

        size *= vol_multiplier

        # Liquidity adjustment
        if liq_regime == LiquidityRegime.ABUNDANT:
            liq_multiplier = 1.0
        elif liq_regime == LiquidityRegime.NORMAL:
            liq_multiplier = 1.0
        elif liq_regime == LiquidityRegime.TIGHT:
            liq_multiplier = 0.8
        else:  # EXHAUSTING
            liq_multiplier = risk_config.tight_liquidity_size_reduction

        size *= liq_multiplier

        # Apply max position size constraint
        return min(size, risk_config.max_position_size)

    def _get_evaluation_zone(self, eval_score: float) -> EvaluationZone:
        """Classify evaluation score into conviction zone"""
        abs_eval = abs(eval_score)

        if abs_eval < 0.2:
            return EvaluationZone.NO_TRADE
        elif abs_eval < 0.5:
            return EvaluationZone.LOW_CONVICTION
        elif abs_eval < 0.8:
            return EvaluationZone.MEDIUM_CONVICTION
        else:
            return EvaluationZone.HIGH_CONVICTION

    def _get_volatility_regime(self, market_state: Any) -> VolatilityRegime:
        """Extract volatility regime from market state"""
        try:
            if hasattr(market_state, "volatility_state"):
                vol_state = market_state.volatility_state
                if hasattr(vol_state, "regime"):
                    # Map VolatilityRegimeType to our VolatilityRegime
                    regime_str = str(vol_state.regime).lower()
                    if "expanding" in regime_str or "high" in regime_str:
                        return VolatilityRegime.HIGH
                    elif "compressing" in regime_str or "low" in regime_str:
                        return VolatilityRegime.LOW
                    else:
                        return VolatilityRegime.MEDIUM
        except Exception as e:
            logger.warning(f"Error extracting volatility regime: {e}")

        return VolatilityRegime.MEDIUM

    def _get_liquidity_regime(self, market_state: Any) -> LiquidityRegime:
        """Extract liquidity regime from market state"""
        try:
            if hasattr(market_state, "liquidity_state"):
                liq_state = market_state.liquidity_state
                if hasattr(liq_state, "regime"):
                    # Map LiquidityRegimeType to our LiquidityRegime
                    regime_str = str(liq_state.regime).lower()
                    if "absorbing" in regime_str:
                        return LiquidityRegime.ABUNDANT
                    elif "exhausting" in regime_str:
                        return LiquidityRegime.EXHAUSTING
                    else:
                        return LiquidityRegime.NORMAL
        except Exception as e:
            logger.warning(f"Error extracting liquidity regime: {e}")

        return LiquidityRegime.NORMAL

    def _log(self, msg: str):
        """Log if verbose enabled"""
        if self.verbose:
            logger.info(msg)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_policy_engine(
    risk_config: Optional[RiskConfig] = None,
    verbose: bool = False,
    official_mode: bool = False,
) -> PolicyEngine:
    """Factory function to create PolicyEngine instance"""
    return PolicyEngine(
        default_risk_config=risk_config, verbose=verbose, official_mode=official_mode
    )


def get_default_risk_config() -> RiskConfig:
    """Return default risk configuration"""
    return RiskConfig()


def get_aggressive_risk_config() -> RiskConfig:
    """Return aggressive risk configuration"""
    return RiskConfig(
        max_risk_per_trade=0.02,  # 2%
        max_position_size=1.0,
        add_threshold=0.5,
        reduce_threshold=0.2,
        min_confidence=0.45,
        enable_add=True,
        enable_reverse=True,
    )


def get_conservative_risk_config() -> RiskConfig:
    """Return conservative risk configuration"""
    return RiskConfig(
        max_risk_per_trade=0.005,  # 0.5%
        max_position_size=0.5,
        add_threshold=0.7,
        reduce_threshold=0.4,
        min_confidence=0.65,
        enable_add=False,
        enable_reverse=False,
    )


if __name__ == "__main__":
    """Example usage and testing"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "=" * 70)
    print("POLICY ENGINE - EXAMPLE USAGE")
    print("=" * 70 + "\n")

    # Create engine
    engine = PolicyEngine(verbose=True)

    # Example 1: Strong bullish, flat position
    print("[EXAMPLE 1] Strong bullish eval, flat position")
    position = PositionState(side=PositionSide.FLAT, size=0.0)
    eval_result = {"eval_score": 0.75, "confidence": 0.85}

    # Note: In real usage, market_state would be a full MarketState object
    decision = engine.decide_action(
        market_state=None,  # Simplified for example
        eval_result=eval_result,
        position_state=position,
    )
    print(f"  Action: {decision.action.value}")
    print(f"  Target Size: {decision.target_size:.2f}")
    print()

    # Example 2: Weakening eval, long position
    print("[EXAMPLE 2] Weakening eval, in long position")
    position = PositionState(side=PositionSide.LONG, size=0.5)
    eval_result = {"eval_score": 0.25, "confidence": 0.60}

    decision = engine.decide_action(
        market_state=None, eval_result=eval_result, position_state=position
    )
    print(f"  Action: {decision.action.value}")
    print(f"  Target Size: {decision.target_size:.2f}")
    print()

    print("=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70 + "\n")
