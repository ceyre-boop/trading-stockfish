#!/usr/bin/env python3
"""
Engine Evaluator Module - Trading Stockfish

Evaluates market state and generates trading decisions.
Implements multi-layered decision logic with risk filters and safety checks.

Decision Output: "buy", "sell", "hold", "close"

Logic Flow:
1. Safety checks (stale data, missing indicators, extreme conditions)
2. Trend regime detection (uptrend, downtrend, sideways)
3. Multi-timeframe confirmation (M1, M5, M15, H1)
4. Volatility and spread filters
5. Sentiment weighting
6. Final decision with confidence score

CausalEvaluator Integration:
- When use_causal_evaluator=True, uses Stockfish-style evaluation combining 8 market factors
- Deterministic, rule-based, fully explainable
- Requires all 8 market state components
- Produces eval_score [-1, +1] + confidence [0, 1]
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import test_flags
from .abs_balance_controller import BALANCE_CONTROLLER
from .canonical_validator import (
    assert_causal_required,
    assert_ml_advisory_only,
    canonical_enforced,
)
from .evaluator_probabilities import compute_probability_tilts
from .ml_aux_signals import compute_ml_hints

# Regime engine helpers
from .regime_engine import compute_regime_bundle

# Minimal core types (placeholder scaffolding)
from .types import EvaluationOutput, MarketState

# Microstructure imports
try:
    from engine.liquidity_metrics import compute_liquidity_metrics
except ImportError:
    compute_liquidity_metrics = None

from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Integration imports (for causal + policy pipeline)
try:
    from engine.integration import (
        create_integrated_evaluator_factory,
        evaluate_and_decide,
    )

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    logger.warning("Integration module not available - causal+policy pipeline disabled")


# ---------------------------------------------------------------------------
# Evaluator v1.3 (deterministic scaffolding with regime engine)
# ---------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def evaluate_state(state: MarketState) -> EvaluationOutput:
    """Deterministic evaluator v1.2 with regime classification.

    Steps:
    1) Base trend/momentum scoring (v1.1 logic)
    2) Regime classification via helpers (overrides state fields)
    3) Regime-aware adjustments to score/confidence
    4) Clamp score and derive confidence
    """

    if not isinstance(state, MarketState):
        raise TypeError("state must be engine.types.MarketState")

    risk_flags: List[str] = []

    # Base scoring
    trend_score = 0.0
    if state.current_price > state.ma_short:
        trend_score += 0.3
    elif state.current_price < state.ma_short:
        trend_score -= 0.3

    if state.current_price > state.ma_long:
        trend_score += 0.2
    elif state.current_price < state.ma_long:
        trend_score -= 0.2

    momentum_score = 0.0
    if state.momentum > 0:
        momentum_score += 0.2
    elif state.momentum < 0:
        momentum_score -= 0.2

    total_score = trend_score + momentum_score
    score = clamp(total_score, -1.0, 1.0)

    # Regime bundle (overrides state-provided fields)
    regime = compute_regime_bundle(state)

    # Volatility sanity check
    avg_abs_returns = 0.0
    if state.recent_returns:
        avg_abs_returns = sum(abs(r) for r in state.recent_returns) / len(
            state.recent_returns
        )
    if avg_abs_returns > 0 and state.volatility > 2.0 * avg_abs_returns:
        risk_flags.append("high_volatility")

    # Regime-aware adjustments
    score += regime["trend_strength"] * 0.4
    score -= regime["liquidity_penalty"]
    score += regime["macro_bias"]

    amd_regime = regime.get("amd_regime", getattr(state, "amd_regime", "NEUTRAL"))
    amd_conf = getattr(state, "amd_confidence", 0.0)
    amd_adj = 0.0
    if amd_regime == "ACCUMULATION":
        amd_adj = min(0.12, 0.08 + 0.05 * amd_conf)
    elif amd_regime == "DISTRIBUTION":
        amd_adj = -min(0.12, 0.08 + 0.05 * amd_conf)
    elif amd_regime == "MANIPULATION":
        amd_adj = -min(0.18, 0.12 + 0.06 * (1.0 + amd_conf))
        risk_flags.append("amd_manipulation")
    score += amd_adj

    vol_shock = regime.get(
        "volatility_shock", getattr(state, "volatility_shock", False)
    )
    vol_shock_strength = float(getattr(state, "volatility_shock_strength", 0.0))
    if vol_shock:
        risk_flags.append("volatility_shock")
        score *= max(0.4, 1.0 - 0.5 * max(0.0, min(1.0, vol_shock_strength)))
        score -= 0.05 * vol_shock_strength

    prob_tilts = compute_probability_tilts(state)
    score += prob_tilts["total_probability_tilt"]

    score = clamp(score, -1.0, 1.0)

    confidence = min(1.0, abs(score))
    confidence *= 1.0 - regime["volatility_intensity"] * 0.5
    session_regime = regime.get("session_regime", getattr(state, "session", "UNKNOWN"))
    if session_regime == "ASIA":
        confidence *= 0.97
    elif session_regime == "NEW_YORK":
        confidence *= 1.02

    # Momentum / ROC tilts (bounded and deterministic)
    m20 = float(getattr(state, "momentum_20", 0.0))
    roc20 = float(getattr(state, "roc_20", 0.0))
    trend_dir = regime.get(
        "trend_direction", getattr(state, "trend_direction", "RANGE")
    )
    if trend_dir == "UP" and m20 > 0.01 and roc20 > 0.01:
        confidence *= 1.02
    elif trend_dir == "DOWN" and m20 < -0.01 and roc20 < -0.01:
        confidence *= 1.02

    depth_imbalance = float(getattr(state, "depth_imbalance", 0.0))
    if score > 0 and depth_imbalance < -0.3:
        confidence *= 0.97
    elif score < 0 and depth_imbalance > 0.3:
        confidence *= 0.97

    if vol_shock:
        confidence *= max(0.5, 1.0 - 0.4 * vol_shock_strength)
    if amd_regime == "MANIPULATION":
        confidence *= 0.8

    confidence = clamp(confidence, 0.0, 1.0)

    # Regime-based risk flags
    if regime["volatility_intensity"] > 0.8:
        risk_flags.append("extreme_volatility")
    if regime["liquidity_penalty"] > 0.2:
        risk_flags.append("thin_liquidity")
    if regime["macro_bias"] < -0.2:
        risk_flags.append("macro_headwind")

    confidence = clamp(confidence, 0.0, 1.0)

    return EvaluationOutput(
        score=score,
        confidence=confidence,
        trend_regime=regime["trend_regime"],
        volatility_regime=regime["volatility_regime"],
        liquidity_regime=regime["liquidity_regime"],
        macro_regime=regime["macro_regime"],
        risk_flags=risk_flags,
        veto_flags=[],
    )


# Decision types
class Decision(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


# Configuration thresholds
class EvaluatorConfig:
    # Microstructure config
    ENABLE_MICROSTRUCTURE = False  # Set True to enable microstructure logic
    MICROSTRUCTURE_SPREAD_KEY = "liquidity_metrics"  # Where to look for micro spread
    """Configuration for evaluator decision thresholds"""
    # Spread and liquidity
    MAX_SPREAD_PIPS = 3.0  # Don't trade if spread > 3 pips (EURUSD)
    MIN_SPREAD_PIPS = 0.5  # Don't trade if spread < 0.5 (likely stale)

    # Volatility
    MAX_VOLATILITY_PCT = 2.0  # Don't trade if volatility > 2% unless strong trend
    MIN_VOLATILITY_PCT = 0.01  # Don't trade if volatility < 0.01% (too quiet)
    HIGH_VOLATILITY_THRESHOLD = 1.5  # Requires stronger signal

    # Trend strength
    MIN_TREND_STRENGTH = 0.3  # Minimum confidence to trade a trend
    STRONG_TREND_STRENGTH = 0.7  # Strong enough to override other filters

    # RSI zones
    RSI_OVERSOLD = 30  # Buy signal zone
    RSI_OVERBOUGHT = 70  # Sell signal zone
    RSI_NEUTRAL_LOW = 40  # Lower neutral bound
    RSI_NEUTRAL_HIGH = 60  # Upper neutral bound

    # Sentiment
    SENTIMENT_WEIGHT = 0.15  # 15% of decision
    SENTIMENT_THRESHOLD = 0.3  # Minimum confidence to use sentiment

    # Data quality
    STALE_STATE_AGE_SEC = 5  # State > 5s old is stale
    MIN_CANDLES_REQUIRED = 20  # Need at least 20 candles for analysis

    # Multi-timeframe
    REQUIRE_HIGHER_TF_CONFIRMATION = True  # Require H1 confirmation for trades

    # Position management
    CLOSE_DECISION_RSI_THRESHOLD = 0.5  # Close if opposite RSI extreme reached
    CLOSE_DECISION_PROFIT_RATIO = 0.7  # Partial close at profit targets


class EvaluatorError(Exception):
    """Base exception for evaluator errors"""

    pass


class SafetyCheckError(EvaluatorError):
    """Raised when safety checks fail"""

    pass


def check_state_safety(state: Dict) -> Tuple[bool, List[str]]:
    """
    Perform safety checks on state before evaluation.

    Args:
        state: Market state dictionary from state_builder

    Returns:
        Tuple of (is_safe: bool, errors: list of warning messages)
    """
    errors = []

    # Check state exists
    if state is None:
        errors.append("State is None")
        return False, errors

    # Check state is not stale
    if state.get("health", {}).get("is_stale", False):
        errors.append("State data is stale")

    # Check for data health errors
    health_errors = state.get("health", {}).get("errors", [])
    if health_errors:
        errors.extend(health_errors)

    # Check timestamp exists and is recent
    timestamp = state.get("timestamp")
    if timestamp is None:
        errors.append("Missing state timestamp")

    # Check tick data
    tick = state.get("tick")
    if not tick or "bid" not in tick or "ask" not in tick:
        errors.append("Missing or invalid tick data")
        return False, errors

    # Check indicators
    indicators = state.get("indicators", {})
    required_indicators = ["rsi_14", "sma_50", "sma_200", "atr_14", "volatility"]
    missing_indicators = [
        ind for ind in required_indicators if indicators.get(ind) is None
    ]
    if missing_indicators:
        errors.append(f"Missing indicators: {missing_indicators}")

    # Check candles
    candles = state.get("candles", {})
    if not candles or "H1" not in candles or candles["H1"] is None:
        errors.append("Missing H1 candle data")

    # Check trend data
    trend = state.get("trend", {})
    if "regime" not in trend or "strength" not in trend:
        errors.append("Missing trend data")

    is_safe = len(errors) == 0
    return is_safe, errors


def check_spread_filter(
    state: Dict, use_microstructure: bool = False
) -> Tuple[bool, str]:
    """
    Check if spread is within acceptable range for trading.

    Args:
        state: Market state dictionary

    Returns:
        Tuple of (pass_filter: bool, reason: str)
    """
    if (
        use_microstructure
        and "liquidity_metrics" in state
        and state["liquidity_metrics"]
    ):
        spread = state["liquidity_metrics"].get("spread", float("inf"))
        liquidity_score = state["liquidity_metrics"].get("liquidity_score", 0)
        stress_flags = state["liquidity_metrics"].get("stress_flags", [])
        if spread > EvaluatorConfig.MAX_SPREAD_PIPS:
            return (
                False,
                f"Microstructure: Spread too wide: {spread:.2f} pips (max: {EvaluatorConfig.MAX_SPREAD_PIPS})",
            )
        if spread < EvaluatorConfig.MIN_SPREAD_PIPS:
            return (
                False,
                f"Microstructure: Spread too tight: {spread:.2f} pips (likely stale data)",
            )
        if "low_liquidity" in stress_flags:
            return (
                False,
                f"Microstructure: Low liquidity detected (score={liquidity_score})",
            )
        return (
            True,
            f"Microstructure: Spread OK: {spread:.2f} pips, liquidity_score={liquidity_score}",
        )
    else:
        spread = state.get("tick", {}).get("spread", float("inf"))
        if spread > EvaluatorConfig.MAX_SPREAD_PIPS:
            return (
                False,
                f"Spread too wide: {spread:.2f} pips (max: {EvaluatorConfig.MAX_SPREAD_PIPS})",
            )
        if spread < EvaluatorConfig.MIN_SPREAD_PIPS:
            return False, f"Spread too tight: {spread:.2f} pips (likely stale data)"
        return True, f"Spread OK: {spread:.2f} pips"


def check_volatility_filter(state: Dict, trend_strength: float) -> Tuple[bool, str]:
    """
    Check if volatility is within acceptable range.
    Allows higher volatility during strong trends.

    Args:
        state: Market state dictionary
        trend_strength: Trend confidence (0-1)

    Returns:
        Tuple of (pass_filter: bool, reason: str)
    """
    volatility = state.get("indicators", {}).get("volatility", 0)

    if volatility < EvaluatorConfig.MIN_VOLATILITY_PCT:
        return False, f"Volatility too low: {volatility:.3f}% (market too quiet)"

    if volatility > EvaluatorConfig.MAX_VOLATILITY_PCT:
        # Allow high volatility if trend is strong
        if trend_strength < EvaluatorConfig.STRONG_TREND_STRENGTH:
            return False, f"Volatility too high: {volatility:.3f}% and trend weak"
        else:
            logger.info(
                f"High volatility {volatility:.3f}% but trend strong ({trend_strength:.2f}), allowing"
            )
            return True, f"High volatility OK due to strong trend"

    return True, f"Volatility normal: {volatility:.3f}%"


def check_multitimeframe_alignment(state: Dict) -> Tuple[str, float]:
    """
    Check alignment across multiple timeframes (M1, M5, M15, H1).

    Returns signal direction ('buy', 'sell', 'hold') with confidence.

    Args:
        state: Market state dictionary

    Returns:
        Tuple of (signal: str, confidence: float)
    """
    candles = state.get("candles", {})
    signals = {}
    confidences = {}

    # Analyze each timeframe
    for tf in ["M1", "M5", "M15", "H1"]:
        if candles.get(tf) is None:
            logger.warning(f"Missing {tf} candles for multi-timeframe analysis")
            signals[tf] = "hold"
            confidences[tf] = 0.0
            continue

        tf_candle = candles[tf]
        tf_indicators = tf_candle.get("indicators", {})

        # Get RSI for this timeframe
        rsi = tf_indicators.get("rsi_14")
        if rsi is None:
            signals[tf] = "hold"
            confidences[tf] = 0.0
            continue

        # Generate signal from RSI
        if rsi < EvaluatorConfig.RSI_OVERSOLD:
            signals[tf] = "buy"
            confidences[tf] = (EvaluatorConfig.RSI_OVERSOLD - rsi) / 30.0  # 0-1
        elif rsi > EvaluatorConfig.RSI_OVERBOUGHT:
            signals[tf] = "sell"
            confidences[tf] = (rsi - EvaluatorConfig.RSI_OVERBOUGHT) / 30.0
        else:
            signals[tf] = "hold"
            confidences[tf] = 0.0

        logger.debug(
            f"{tf} RSI: {rsi:.1f} → {signals[tf]} (confidence: {confidences[tf]:.2f})"
        )

    # Aggregate signals (require H1 agreement if configured)
    buy_votes = sum(1 for s in signals.values() if s == "buy")
    sell_votes = sum(1 for s in signals.values() if s == "sell")

    logger.debug(
        f"Multi-TF votes: BUY={buy_votes}, SELL={sell_votes}, HOLD={4-buy_votes-sell_votes}"
    )

    # Determine consensus signal
    if EvaluatorConfig.REQUIRE_HIGHER_TF_CONFIRMATION:
        # Require H1 to align
        if signals.get("H1") == "buy" and buy_votes >= 2:
            return "buy", min(0.9, (buy_votes / 4.0) * 0.9)
        elif signals.get("H1") == "sell" and sell_votes >= 2:
            return "sell", min(0.9, (sell_votes / 4.0) * 0.9)
        else:
            return "hold", 0.0
    else:
        # Any alignment
        if buy_votes > sell_votes:
            return "buy", (buy_votes / 4.0)
        elif sell_votes > buy_votes:
            return "sell", (sell_votes / 4.0)
        else:
            return "hold", 0.0


def calculate_signal_confidence(
    trend_signal: str,
    trend_strength: float,
    multitf_signal: str,
    multitf_confidence: float,
    sentiment_score: float,
    sentiment_confidence: float,
) -> float:
    """
    Calculate overall confidence score for the trading signal.

    Combines trend, multi-timeframe, and sentiment signals.

    Args:
        trend_signal: 'buy', 'sell', or 'hold'
        trend_strength: Trend confidence (0-1)
        multitf_signal: Multi-timeframe signal
        multitf_confidence: Multi-timeframe confidence (0-1)
        sentiment_score: News sentiment (-1 to 1)
        sentiment_confidence: Sentiment confidence (0-1)

    Returns:
        Overall confidence (0-1)
    """
    confidence = 0.0

    # Trend contribution (40%)
    if trend_signal in ["buy", "sell"]:
        confidence += trend_strength * 0.4

    # Multi-timeframe contribution (45%)
    if multitf_signal == trend_signal:
        confidence += multitf_confidence * 0.45
    elif multitf_signal == "hold":
        confidence += multitf_confidence * 0.20  # Weak support

    # Sentiment contribution (15%)
    if sentiment_confidence > EvaluatorConfig.SENTIMENT_THRESHOLD:
        if trend_signal == "buy" and sentiment_score > 0:
            confidence += min(sentiment_score, 1.0) * EvaluatorConfig.SENTIMENT_WEIGHT
        elif trend_signal == "sell" and sentiment_score < 0:
            confidence += (
                abs(min(sentiment_score, -1.0)) * EvaluatorConfig.SENTIMENT_WEIGHT
            )

    return min(confidence, 1.0)


def evaluate_close_signal(
    state: Dict, open_position: Optional[Dict] = None
) -> Tuple[str, float]:
    """
    Determine if an open position should be closed.

    Args:
        state: Market state dictionary
        open_position: Optional dict with position info:
            {'direction': 'buy' or 'sell', 'entry_price': float, 'current_pnl_pct': float}

    Returns:
        Tuple of (decision: 'close' or 'hold', confidence: float)
    """
    if open_position is None:
        return "hold", 0.0

    rsi = state.get("indicators", {}).get("rsi_14")
    if rsi is None:
        return "hold", 0.0

    direction = open_position.get("direction")

    # Close if opposite extreme is reached
    if direction == "buy" and rsi > EvaluatorConfig.RSI_OVERBOUGHT:
        logger.info(f"Close signal: Buy position, RSI overbought at {rsi:.1f}")
        return "close", min(0.8, (rsi - EvaluatorConfig.RSI_OVERBOUGHT) / 30.0)

    if direction == "sell" and rsi < EvaluatorConfig.RSI_OVERSOLD:
        logger.info(f"Close signal: Sell position, RSI oversold at {rsi:.1f}")
        return "close", min(0.8, (EvaluatorConfig.RSI_OVERSOLD - rsi) / 30.0)

    return "hold", 0.0


def evaluate(
    state: Dict,
    open_position: Optional[Dict] = None,
    require_high_confidence: bool = False,
    enable_microstructure: Optional[bool] = None,
) -> Dict:
    """
    Main evaluation function - returns trading decision.

    Implements multi-layered decision logic:
    1. Safety checks
    2. Trend detection
    3. Multi-timeframe confirmation
    4. Volatility and spread filters
    5. Sentiment weighting
    6. Position management (close vs new entry)

    Args:
        state: Market state dictionary from state_builder
        open_position: Optional dict with position info for close evaluation
        require_high_confidence: If True, only return buy/sell for confidence > 0.6

    Returns:
        Dict with:
        {
            'decision': 'buy' | 'sell' | 'hold' | 'close',
            'confidence': float (0-1),
            'reason': str (explanation),
            'details': dict (internal analysis details),
        }
    """
    if canonical_enforced() and not getattr(test_flags, "CANONICAL_TEST_BYPASS", False):
        # Legacy evaluator must not be reachable in canonical/official modes
        raise ValueError(
            "Legacy evaluator is forbidden in canonical/official modes; use CausalEvaluator."
        )
    logger.info("=" * 60)
    logger.info("EVALUATION STARTED")
    logger.info("=" * 60)

    decision = Decision.HOLD
    confidence = 0.0
    reason = ""
    details = {}

    # LAYER 1: Safety Checks
    # ============================================================
    logger.info("\n[LAYER 1] Safety Checks")
    is_safe, safety_errors = check_state_safety(state)

    if not is_safe:
        reason = f"Safety check failed: {'; '.join(safety_errors)}"
        logger.warning(f"❌ {reason}")
        details["safety_errors"] = safety_errors
        return {
            "decision": Decision.HOLD.value,
            "confidence": 0.0,
            "reason": reason,
            "details": details,
        }

    logger.info("✓ Safety checks passed")

    # LAYER 2: Spread and Liquidity Filter
    # ============================================================
    logger.info("\n[LAYER 2] Spread Filter")
    # Determine if microstructure is enabled (config or override)
    use_micro = (
        enable_microstructure
        if enable_microstructure is not None
        else getattr(EvaluatorConfig, "ENABLE_MICROSTRUCTURE", False)
    )
    spread_pass, spread_reason = check_spread_filter(
        state, use_microstructure=use_micro
    )
    details["spread_check"] = spread_reason
    if use_micro:
        details["spread"] = state.get("spread")
        details["liquidity_score"] = state.get("liquidity_score")
        details["liquidity_stress_flags"] = state.get("liquidity_stress_flags")
        details["order_flow_features"] = state.get("order_flow_features")
    if not spread_pass:
        logger.warning(f"❌ {spread_reason}")
        reason = f"Liquidity insufficient: {spread_reason}"
        return {
            "decision": Decision.HOLD.value,
            "confidence": 0.0,
            "reason": reason,
            "details": details,
        }
    logger.info(f"✓ {spread_reason}")
    # LAYER 3.5: Microstructure EV/Risk/Cost Adjustments
    if use_micro:
        spread = state.get("spread", 0)
        liquidity_score = state.get("liquidity_score", 0)
        order_flow = state.get("order_flow_features", {})
        ev_penalty = 0.0
        risk_penalty = 0.0
        if spread > 2.0:
            ev_penalty += 0.1 * (spread - 2.0)
            risk_penalty += 0.1 * (spread - 2.0)
        if liquidity_score < 20:
            ev_penalty += 0.1
            risk_penalty += 0.1
        if order_flow.get("spoofing_score", 0) > 0:
            ev_penalty += 0.05 * order_flow["spoofing_score"]
            risk_penalty += 0.05 * order_flow["spoofing_score"]
        if order_flow.get("quote_pulling_score", 0) > 0:
            risk_penalty += 0.05 * order_flow["quote_pulling_score"]
        if order_flow.get("net_imbalance", 0) > 2:
            ev_penalty -= 0.05
        ev_penalty = min(max(ev_penalty, -0.2), 0.5)
        risk_penalty = min(max(risk_penalty, 0), 0.5)
        details["micro_ev_penalty"] = ev_penalty
        details["micro_risk_penalty"] = risk_penalty

    # LAYER 3: Extract core indicators
    # ============================================================
    logger.info("\n[LAYER 3] Extract Indicators")
    indicators = state.get("indicators", {})
    rsi = indicators.get("rsi_14")
    sma_50 = indicators.get("sma_50")
    sma_200 = indicators.get("sma_200")
    atr = indicators.get("atr_14")
    volatility = indicators.get("volatility", 0)

    logger.debug(
        f"RSI: {rsi:.1f}, SMA50: {sma_50:.4f}, SMA200: {sma_200:.4f}, ATR: {atr:.4f}, Vol: {volatility:.3f}%"
    )

    # LAYER 3.5: Adaptive Factor Weights (v4.0‑E)
    volatility_state = state.get("volatility_state", {})
    regime_state = state.get("regime_state", {})
    vol_regime = volatility_state.get("vol_regime", "NORMAL")
    liq_regime = regime_state.get("liq_regime", "NORMAL")
    macro_regime = regime_state.get("macro_regime", "RISK_ON")

    # Default weights
    trend_weight = 1.0
    order_flow_weight = 1.0
    liquidity_weight = 1.0
    volatility_weight = 1.0
    macro_weight = 1.0
    long_bias_weight = 1.0

    # Regime-conditioned adjustments
    if vol_regime in ["HIGH", "EXTREME"]:
        trend_weight *= 0.5
        liquidity_weight *= 1.5
    if liq_regime in ["THIN", "FRAGILE"]:
        order_flow_weight *= 0.7
        liquidity_weight *= 1.3
    if macro_regime == "RISK_OFF":
        long_bias_weight *= 0.5

    details["adaptive_weights"] = {
        "trend": trend_weight,
        "order_flow": order_flow_weight,
        "liquidity": liquidity_weight,
        "volatility": volatility_weight,
        "macro": macro_weight,
        "long_bias": long_bias_weight,
    }

    # Apply ABS break-level balancing so no factor can dominate.
    balanced_weights = BALANCE_CONTROLLER.balance_weights(
        details["adaptive_weights"],
        regime_state=regime_state,
        volatility_state=volatility_state,
    )
    details["balanced_weights"] = balanced_weights

    # Optional ML auxiliary hints (advisory only, passed through balancing).
    ml_hints_raw = compute_ml_hints(state)
    ml_hints_balanced = BALANCE_CONTROLLER.balance_weights(
        {
            "volatility": ml_hints_raw.get("vol_cluster_hint", 0.0),
            "macro": ml_hints_raw.get("macro_vol_hint", 0.0),
        },
        regime_state=regime_state,
        volatility_state=volatility_state,
    )
    details["ml_hints"] = {
        "raw": ml_hints_raw,
        "balanced": ml_hints_balanced,
    }

    # LAYER 4: Volatility Filter
    # ============================================================
    logger.info("\n[LAYER 4] Volatility Filter")
    trend = state.get("trend", {})
    trend_strength = trend.get("strength", 0)
    volatility_pass, volatility_reason = check_volatility_filter(state, trend_strength)
    details["volatility_check"] = volatility_reason

    if not volatility_pass:
        logger.warning(f"❌ {volatility_reason}")
        reason = f"Volatility condition failed: {volatility_reason}"
        return {
            "decision": Decision.HOLD.value,
            "confidence": 0.0,
            "reason": reason,
            "details": details,
        }

    logger.info(f"✓ {volatility_reason}")

    # LAYER 5: Trend Detection
    # ============================================================
    logger.info("\n[LAYER 5] Trend Detection")
    trend_regime = trend.get("regime", "sideways")
    logger.info(
        f"Trend Regime: {trend_regime.upper()} (strength: {trend_strength:.2f})"
    )

    # Generate base trend signal
    trend_signal = "hold"
    if (
        trend_regime == "uptrend"
        and trend_strength >= EvaluatorConfig.MIN_TREND_STRENGTH
    ):
        trend_signal = "buy"
        logger.info(f"→ Bullish trend signal (strength: {trend_strength:.2f})")
    elif (
        trend_regime == "downtrend"
        and trend_strength >= EvaluatorConfig.MIN_TREND_STRENGTH
    ):
        trend_signal = "sell"
        logger.info(f"→ Bearish trend signal (strength: {trend_strength:.2f})")
    else:
        logger.info(f"→ Insufficient trend strength ({trend_strength:.2f})")

    details["trend"] = {
        "regime": trend_regime,
        "strength": trend_strength,
        "signal": trend_signal,
    }

    # LAYER 6: Multi-Timeframe Confirmation
    # ============================================================
    logger.info("\n[LAYER 6] Multi-Timeframe Confirmation")
    multitf_signal, multitf_confidence = check_multitimeframe_alignment(state)
    logger.info(
        f"Multi-TF Signal: {multitf_signal.upper()} (confidence: {multitf_confidence:.2f})"
    )

    details["multitf"] = {
        "signal": multitf_signal,
        "confidence": multitf_confidence,
    }

    # LAYER 7: Sentiment Analysis
    # ============================================================
    logger.info("\n[LAYER 7] Sentiment Analysis")
    sentiment = state.get("sentiment", {})
    sentiment_score = sentiment.get("score", 0)
    sentiment_confidence = sentiment.get("confidence", 0)
    logger.info(
        f"Sentiment: {sentiment_score:.2f} (confidence: {sentiment_confidence:.2f})"
    )

    details["sentiment"] = {
        "score": sentiment_score,
        "confidence": sentiment_confidence,
        "source": sentiment.get("source", "unknown"),
    }

    # LAYER 8: Position Management (Close Check)
    # ============================================================
    logger.info("\n[LAYER 8] Position Management")
    if open_position:
        close_decision, close_confidence = evaluate_close_signal(state, open_position)
        if close_decision == "close":
            logger.info(f"✓ CLOSE DECISION (confidence: {close_confidence:.2f})")
            return {
                "decision": Decision.CLOSE.value,
                "confidence": close_confidence,
                "reason": f"Close position signal triggered",
                "details": {**details, "close_confidence": close_confidence},
            }

    # LAYER 9: Final Decision Logic
    # ============================================================
    logger.info("\n[LAYER 9] Final Decision")

    # Combine signals
    if trend_signal in ["buy", "sell"]:
        if trend_signal == multitf_signal or multitf_signal == "hold":
            # Aligned or multi-tf doesn't contradict
            decision_value = trend_signal
            confidence = calculate_signal_confidence(
                trend_signal,
                trend_strength,
                multitf_signal,
                multitf_confidence,
                sentiment_score,
                sentiment_confidence,
            )

            # Apply ABS balance confidence guard so capped weights cannot over-inflate confidence.
            confidence, balance_scale = BALANCE_CONTROLLER.apply_confidence_guard(
                confidence, balanced_weights
            )
            details["balance_scale"] = balance_scale

            # Advisory ML hints are recorded but cannot alter decisions/confidence.
            ml_conf_adj = 0.0
            details["ml_conf_adj"] = ml_conf_adj
            assert_ml_advisory_only(advisory_adjustment_applied=False)

            if confidence < EvaluatorConfig.MIN_TREND_STRENGTH:
                logger.info(
                    f"Signal too weak: {confidence:.2f} < {EvaluatorConfig.MIN_TREND_STRENGTH}"
                )
                decision_value = "hold"
        else:
            # Conflicting signals
            logger.warning(
                f"Conflicting signals: trend={trend_signal}, multitf={multitf_signal}"
            )
            decision_value = "hold"
            confidence = 0.0
    else:
        decision_value = "hold"

    # Apply high confidence requirement if needed
    if (
        require_high_confidence
        and decision_value in ["buy", "sell"]
        and confidence < 0.6
    ):
        logger.info(
            f"Confidence {confidence:.2f} below high-confidence threshold (0.6)"
        )
        decision_value = "hold"
        confidence = 0.0

    decision = Decision(decision_value)

    # Generate reason
    if decision == Decision.BUY:
        reason = f"BUY signal: {trend_regime} trend (strength: {trend_strength:.2f}), confirmed by multi-timeframe"
    elif decision == Decision.SELL:
        reason = f"SELL signal: {trend_regime} trend (strength: {trend_strength:.2f}), confirmed by multi-timeframe"
    else:
        reason = "HOLD: Insufficient signal strength or conflicting indicators"

    logger.info(f"\n{'='*60}")
    logger.info(f"DECISION: {decision.value.upper()} (confidence: {confidence:.2f})")
    logger.info(f"REASON: {reason}")
    logger.info(f"{'='*60}\n")

    details["final_confidence"] = confidence

    return {
        "decision": decision.value,
        "confidence": confidence,
        "reason": reason,
        "details": details,
    }


def evaluate_bulk(
    states: Dict[str, Dict],
    open_positions: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """
    Evaluate multiple symbols in bulk (for multiple trading pairs).

    Args:
        states: Dict of {symbol: state_dict}
        open_positions: Optional dict of {symbol: position_dict}

    Returns:
        Dict of {symbol: evaluation_result}
    """
    results = {}
    open_positions = open_positions or {}
    for symbol, state in states.items():
        # Extract new volatility and regime states
        volatility_state = state.get("volatility_state", {})
        regime_state = state.get("regime_state", {})

        # Example: Use vol_regime and regime_transition in decision
        vol_regime = volatility_state.get("vol_regime", "NORMAL")
        regime_transition = regime_state.get("regime_transition", False)
        regime_confidence = regime_state.get("regime_confidence", 0.0)

        # Existing evaluation logic (simplified for illustration)
        # You may want to call your main evaluation function here
        result = {
            "decision": "hold",
            "confidence": regime_confidence,
            "reason": f"Volatility regime: {vol_regime}, Regime transition: {regime_transition}",
            "details": {
                "volatility_state": volatility_state,
                "regime_state": regime_state,
            },
        }
        results[symbol] = result
    return results

    for symbol, state in states.items():
        position = open_positions.get(symbol)
        results[symbol] = evaluate(state, open_position=position)

    return results


# ============================================================================
# CAUSAL EVALUATOR INTEGRATION
# ============================================================================


def evaluate_with_causal(
    state: Dict,
    causal_evaluator: Optional[Any] = None,
    market_state: Optional[Any] = None,
    policy: Optional[Any] = None,
) -> Dict:
    """
    Evaluate market state using CausalEvaluator (Stockfish-style).

    This integrates the deterministic, rule-based CausalEvaluator which combines
    8 market factors into a single evaluation score [-1, +1].

    Args:
        state: Traditional market state dictionary (legacy format)
        causal_evaluator: Initialized CausalEvaluator instance
        market_state: CausalEvaluator MarketState dataclass (preferred)
        policy: Optional policy config (PolicyConfig) for feature weights/trust

    Returns:
        Dict with decision, confidence, reason, and full causal reasoning

    Raises:
        ValueError: If causal_evaluator is None or market_state not properly configured
    """
    if causal_evaluator is None:
        raise ValueError("causal_evaluator cannot be None")

    if market_state is None:
        raise ValueError(
            "market_state (CausalEvaluator.MarketState) required for causal evaluation"
        )

    # Evaluate using CausalEvaluator
    try:
        result = causal_evaluator.evaluate(market_state)
    except Exception as e:
        logger.error(f"CausalEvaluator failed: {e}")
        raise

    def _derive_regimes(ms: Any) -> List[str]:
        regimes: List[str] = []
        try:
            session_label = getattr(ms, "session", None) or getattr(
                ms, "session_regime", None
            )
            if session_label:
                regimes.append(str(session_label))

            vol_state = getattr(ms, "volatility_state", None)
            if vol_state is not None:
                vol_reg = getattr(vol_state, "regime", None)
                if vol_reg is not None:
                    regimes.append(str(getattr(vol_reg, "name", vol_reg)))

            macro_news = getattr(ms, "macro_news_state", None)
            macro_label = (
                getattr(macro_news, "macro_news_state", None) if macro_news else None
            )
            if macro_label:
                upper_label = str(macro_label).upper()
                if "RISK_ON" in upper_label:
                    regimes.append("MACRO_ON")
                elif "RISK_OFF" in upper_label:
                    regimes.append("MACRO_OFF")

            macro_state = getattr(ms, "macro_state", None)
            sentiment = (
                getattr(macro_state, "sentiment_score", None) if macro_state else None
            )
            if sentiment is not None:
                if sentiment > 0.2:
                    regimes.append("MACRO_ON")
                elif sentiment < -0.2:
                    regimes.append("MACRO_OFF")
        except Exception:
            pass

        dedup: List[str] = []
        for r in regimes:
            if r and r not in dedup:
                dedup.append(str(r))
        return dedup

    def _extract_session_context(
        ms: Any, legacy_state: Optional[Dict]
    ) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {"session": "UNKNOWN", "modifiers": {}}
        try:
            session_label = (
                getattr(ms, "session", None)
                or getattr(ms, "session_regime", None)
                or (legacy_state or {}).get("session_regime")
                or (legacy_state or {}).get("session")
            )
            ctx["session"] = str(session_label or "UNKNOWN")

            # Prefer explicit session_context->modifiers shape, then flat session_modifiers, then legacy
            sc = getattr(ms, "session_context", None)
            if isinstance(sc, dict) and sc.get("modifiers"):
                ctx["modifiers"] = sc.get("modifiers", {})
            else:
                mods = getattr(ms, "session_modifiers", None)
                if isinstance(mods, dict):
                    ctx["modifiers"] = mods
                else:
                    ctx["modifiers"] = (
                        (legacy_state or {})
                        .get("session_context", {})
                        .get("modifiers", {})
                    )
        except Exception:
            ctx = {"session": "UNKNOWN", "modifiers": {}}
        return ctx

    def _print_session_trace(
        session_ctx: Dict[str, Any], regimes: List[str], factors: List[Dict[str, Any]]
    ) -> None:
        try:
            mods = session_ctx.get("modifiers") or {}
            print("\n[SESSION CONTEXT]")
            print(
                f"  session={session_ctx.get('session', 'UNKNOWN')} | "
                f"vol_scale={mods.get('volatility_scale', 1.0):.2f} "
                f"liq_scale={mods.get('liquidity_scale', 1.0):.2f} "
                f"trade_scale={mods.get('trade_freq_scale', 1.0):.2f} "
                f"risk_scale={mods.get('risk_scale', 1.0):.2f}"
            )
            print(f"  regimes={', '.join(regimes) if regimes else '<none>'}")
            if factors:
                print("[SESSION WEIGHTS]")
                for f in factors:
                    print(
                        "  "
                        f"{f.get('factor', ''):18s} base={f.get('policy_base_weight', 1.0):.3f} "
                        f"trust={f.get('trust_score', 1.0):.3f} "
                        f"regime_mult={f.get('regime_multiplier', 1.0):.3f} "
                        f"session_mult={f.get('session_multiplier', 1.0):.3f} "
                        f"eff={f.get('policy_weight', 1.0):.3f} "
                        f"raw={f.get('raw_score', 0.0):.3f} "
                        f"weighted={f.get('weighted_score', 0.0):.3f}"
                    )
        except Exception:
            # Terminal trace is best-effort only
            pass

    # Convert causal evaluation to decision
    eval_score = result.eval_score
    confidence = result.confidence

    # Interpret evaluation as decision
    if eval_score > 0.2:
        decision = "buy"
        reason = f"CausalEvaluator BULLISH (score: {eval_score:.3f})"
    elif eval_score < -0.2:
        decision = "sell"
        reason = f"CausalEvaluator BEARISH (score: {eval_score:.3f})"
    else:
        decision = "hold"
        reason = f"CausalEvaluator NEUTRAL (score: {eval_score:.3f})"

    # Build reasoning from causal factors
    factor_explanations = []

    # Apply policy weights/trust if provided (factor names are used as keys)
    session_context = _extract_session_context(market_state, state)
    policy_applied = policy is not None
    policy_factors = []
    regimes: List[str] = _derive_regimes(market_state) if market_state else []
    if policy_applied:
        # deterministic ordering by factor name
        sorted_factors = sorted(result.scoring_factors, key=lambda f: f.factor_name)
        eval_score = 0.0
        session_mods = session_context.get("modifiers") or {}
        for factor in sorted_factors:
            trust = policy.get_trust(factor.factor_name) if policy else 1.0
            base_weight = policy.get_base_weight(factor.factor_name) if policy else 1.0
            regime_multiplier = (
                policy.get_regime_multiplier(factor.factor_name, regimes)
                if policy
                else 1.0
            )
            session_multiplier = float(session_mods.get("risk_scale", 1.0))
            policy_effective_weight = (
                base_weight * trust * regime_multiplier * session_multiplier
                if policy
                else 1.0
            )
            combined_weight = factor.weight * policy_effective_weight
            raw_score = factor.score
            weighted_score = 0.0 if trust == 0 else raw_score * combined_weight
            eval_score += weighted_score
            policy_factors.append(
                {
                    "factor": factor.factor_name,
                    "raw_score": raw_score,
                    "weight": factor.weight,
                    "policy_base_weight": base_weight,
                    "policy_weight": policy_effective_weight,
                    "regime_multiplier": regime_multiplier,
                    "session_multiplier": session_multiplier,
                    "trust_score": trust,
                    "weighted_score": weighted_score,
                    "explanation": factor.explanation,
                }
            )
        eval_score = float(np.clip(eval_score, -1.0, 1.0))
        _print_session_trace(session_context, regimes, policy_factors)
    else:
        sorted_factors = result.scoring_factors
        eval_score = result.eval_score
        for factor in sorted_factors:
            policy_factors.append(
                {
                    "factor": factor.factor_name,
                    "raw_score": factor.score,
                    "weight": factor.weight,
                    "policy_base_weight": 1.0,
                    "policy_weight": 1.0,
                    "regime_multiplier": 1.0,
                    "trust_score": 1.0,
                    "weighted_score": factor.score * factor.weight,
                    "explanation": factor.explanation,
                }
            )

    for pf in policy_factors:
        factor_explanations.append(
            {
                "factor": pf["factor"],
                "score": pf["raw_score"],
                "weight": pf["weight"],
                "policy_base_weight": pf.get("policy_base_weight", 1.0),
                "policy_weight": pf["policy_weight"],
                "regime_multiplier": pf.get("regime_multiplier", 1.0),
                "trust_score": pf["trust_score"],
                "weighted_score": pf["weighted_score"],
                "explanation": pf.get("explanation"),
            }
        )

    return {
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "eval_score": eval_score,
        "causal_reasoning": factor_explanations,
        "timestamp": result.timestamp,
        "evaluator_mode": "causal",
        "policy_applied": policy_applied,
        "details": {
            "causal_eval": eval_score,
            "factors": factor_explanations,
        },
    }


def create_evaluator_factory(
    use_causal: bool = False, use_policy_engine: bool = False, **causal_kwargs
) -> Callable:
    """
    Factory function to create evaluator with selected backend.

    Args:
        use_causal: If True, use CausalEvaluator; if False, use traditional evaluator
        use_policy_engine: If True AND use_causal=True, add PolicyEngine decision layer
        **causal_kwargs: Keyword arguments for CausalEvaluator (weights, verbose, official_mode)

    Returns:
        Evaluator function that accepts (state, market_state, open_position)

    Raises:
        ImportError: If mode not supported or modules unavailable
    """
    if canonical_enforced():
        assert_causal_required(use_causal)

    # ========================================================================
    # MODE 1: Integrated CausalEvaluator + PolicyEngine (NEW)
    # ========================================================================
    if use_causal and use_policy_engine:
        if not INTEGRATION_AVAILABLE:
            raise ImportError("Integration module required for causal+policy mode")

        try:
            from engine.causal_evaluator import CausalEvaluator
            from engine.policy_engine import PolicyEngine, RiskConfig

            # Initialize evaluators
            causal_evaluator = CausalEvaluator(**causal_kwargs)
            policy_engine = PolicyEngine(verbose=causal_kwargs.get("verbose", False))

            def integrated_wrapper(
                state: Dict,
                market_state: Optional[Any] = None,
                open_position: Optional[Dict] = None,
                position_state: Optional[Any] = None,
                risk_config: Optional[Any] = None,
                daily_loss_pct: float = 0.0,
            ) -> Dict:
                """Integrated causal+policy evaluation"""

                if market_state is None:
                    if canonical_enforced():
                        raise ValueError(
                            "MarketState required in canonical/official modes; legacy evaluator fallback is blocked."
                        )
                    logger.warning(
                        "market_state not provided; falling back to traditional evaluator"
                    )
                    return evaluate(state, open_position=open_position)

                # Use defaults if not provided
                if position_state is None:
                    from engine.policy_engine import PositionSide, PositionState

                    position_state = PositionState(side=PositionSide.FLAT, size=0.0)

                if risk_config is None:
                    risk_config = RiskConfig()

                # Run integrated pipeline
                try:
                    result = evaluate_and_decide(
                        market_state=market_state,
                        position_state=position_state,
                        risk_config=risk_config,
                        causal_evaluator=causal_evaluator,
                        policy_engine=policy_engine,
                        daily_loss_pct=daily_loss_pct,
                        verbose=causal_kwargs.get("verbose", False),
                    )

                    # Convert to legacy format for compatibility
                    return {
                        "decision": (
                            result["action"].lower()
                            if isinstance(result["action"], str)
                            else result["action"].value.lower()
                        ),
                        "confidence": result["confidence"],
                        "reason": f"CausalEval+Policy: {result['decision_zone']}",
                        "eval_score": result["eval_score"],
                        "details": {
                            "causal_eval": result["eval_score"],
                            "policy_action": result["action"],
                            "target_size": result["target_size"],
                            "evaluation_zone": result["decision_zone"],
                            "causal_reasoning": result["reasoning"]["eval"],
                            "policy_reasoning": result["reasoning"]["policy"],
                        },
                        "integrated_mode": True,
                        "causal_evaluator": True,
                        "policy_engine": True,
                        "deterministic": True,
                        "lookahead_safe": True,
                    }
                except Exception as e:
                    logger.error(
                        f"Integrated pipeline failed: {e}; falling back to traditional"
                    )
                    return evaluate(state, open_position=open_position)

            return integrated_wrapper

        except ImportError as e:
            logger.error(f"Cannot import causal+policy modules: {e}")
            raise

    # ========================================================================
    # MODE 2: CausalEvaluator only
    # ========================================================================
    elif use_causal:
        try:
            from engine.causal_evaluator import CausalEvaluator

            # Initialize CausalEvaluator with provided kwargs
            causal_evaluator = CausalEvaluator(**causal_kwargs)

            def causal_eval_wrapper(
                state: Dict,
                market_state: Optional[Any] = None,
                open_position: Optional[Dict] = None,
            ) -> Dict:
                """Wrapper for causal evaluation"""
                if market_state is None:
                    # If no causal market_state provided, use traditional evaluator
                    logger.warning(
                        "market_state not provided; falling back to traditional evaluator"
                    )
                    return evaluate(state, open_position=open_position)

                return evaluate_with_causal(
                    state=state,
                    causal_evaluator=causal_evaluator,
                    market_state=market_state,
                )

            return causal_eval_wrapper

        except ImportError:
            logger.error(
                "CausalEvaluator not available; falling back to traditional evaluator"
            )
            return lambda state, market_state=None, open_position=None: evaluate(
                state, open_position=open_position
            )

    # ========================================================================
    # MODE 3: Traditional evaluator (default)
    # ========================================================================
    else:
        # Return traditional evaluator
        return lambda state, market_state=None, open_position=None: evaluate(
            state, open_position=open_position
        )


if __name__ == "__main__":
    """Example usage and testing"""

    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "=" * 70)
    print("EVALUATOR ENGINE - TEST RUN")
    print("=" * 70)

    # Create a mock state (would come from state_builder.py)
    mock_state = {
        "timestamp": 1705441218.528,
        "symbol": "EURUSD",
        "tick": {
            "bid": 1.0850,
            "ask": 1.0852,
            "spread": 2.0,
            "last_tick_time": 1705441213,
        },
        "indicators": {
            "rsi_14": 35.0,  # Oversold = buy signal
            "sma_50": 1.0835,
            "sma_200": 1.0800,
            "atr_14": 0.0012,
            "volatility": 0.5,
        },
        "trend": {
            "regime": "uptrend",
            "strength": 0.75,
        },
        "sentiment": {
            "score": 0.2,
            "confidence": 0.3,
            "source": "placeholder",
        },
        "candles": {
            "M1": {
                "indicators": {"rsi_14": 25.0},
                "latest": {"time": 1705441210},
            },
            "M5": {
                "indicators": {"rsi_14": 32.0},
                "latest": {"time": 1705441210},
            },
            "M15": {
                "indicators": {"rsi_14": 35.0},
                "latest": {"time": 1705441210},
            },
            "H1": {
                "indicators": {"rsi_14": 38.0},
                "latest": {"time": 1705441210},
            },
        },
        "health": {
            "is_stale": False,
            "last_update": 1705441218.528,
            "errors": [],
        },
    }

    # Test 1: Basic evaluation
    print("\n[TEST 1] Basic Evaluation (Uptrend + Oversold RSI)")
    result = evaluate(mock_state)
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reason: {result['reason']}")

    # Test 2: Evaluation with position
    print("\n[TEST 2] Evaluation with Open Position (Close Check)")
    mock_state_overbought = mock_state.copy()
    mock_state_overbought["indicators"]["rsi_14"] = 75.0

    open_pos = {"direction": "buy", "entry_price": 1.0840}
    result = evaluate(mock_state_overbought, open_position=open_pos)
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reason: {result['reason']}")

    # Test 3: Sell signal
    print("\n[TEST 3] Sell Signal (Downtrend + Overbought)")
    mock_state_sell = mock_state.copy()
    mock_state_sell["trend"]["regime"] = "downtrend"
    mock_state_sell["indicators"]["rsi_14"] = 75.0
    mock_state_sell["sentiment"]["score"] = -0.3

    result = evaluate(mock_state_sell)
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reason: {result['reason']}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70 + "\n")
