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
from typing import Dict, Optional, Tuple, List, Any, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# Integration imports (for causal + policy pipeline)
try:
    from engine.integration import evaluate_and_decide, create_integrated_evaluator_factory
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    logger.warning("Integration module not available - causal+policy pipeline disabled")

# Decision types
class Decision(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


# Configuration thresholds
class EvaluatorConfig:
    """Configuration for evaluator decision thresholds"""
    
    # Spread and liquidity
    MAX_SPREAD_PIPS = 3.0               # Don't trade if spread > 3 pips (EURUSD)
    MIN_SPREAD_PIPS = 0.5               # Don't trade if spread < 0.5 (likely stale)
    
    # Volatility
    MAX_VOLATILITY_PCT = 2.0             # Don't trade if volatility > 2% unless strong trend
    MIN_VOLATILITY_PCT = 0.01            # Don't trade if volatility < 0.01% (too quiet)
    HIGH_VOLATILITY_THRESHOLD = 1.5      # Requires stronger signal
    
    # Trend strength
    MIN_TREND_STRENGTH = 0.3             # Minimum confidence to trade a trend
    STRONG_TREND_STRENGTH = 0.7          # Strong enough to override other filters
    
    # RSI zones
    RSI_OVERSOLD = 30                    # Buy signal zone
    RSI_OVERBOUGHT = 70                  # Sell signal zone
    RSI_NEUTRAL_LOW = 40                 # Lower neutral bound
    RSI_NEUTRAL_HIGH = 60                # Upper neutral bound
    
    # Sentiment
    SENTIMENT_WEIGHT = 0.15              # 15% of decision
    SENTIMENT_THRESHOLD = 0.3            # Minimum confidence to use sentiment
    
    # Data quality
    STALE_STATE_AGE_SEC = 5              # State > 5s old is stale
    MIN_CANDLES_REQUIRED = 20            # Need at least 20 candles for analysis
    
    # Multi-timeframe
    REQUIRE_HIGHER_TF_CONFIRMATION = True  # Require H1 confirmation for trades
    
    # Position management
    CLOSE_DECISION_RSI_THRESHOLD = 0.5   # Close if opposite RSI extreme reached
    CLOSE_DECISION_PROFIT_RATIO = 0.7    # Partial close at profit targets


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
    if state.get('health', {}).get('is_stale', False):
        errors.append("State data is stale")
    
    # Check for data health errors
    health_errors = state.get('health', {}).get('errors', [])
    if health_errors:
        errors.extend(health_errors)
    
    # Check timestamp exists and is recent
    timestamp = state.get('timestamp')
    if timestamp is None:
        errors.append("Missing state timestamp")
    
    # Check tick data
    tick = state.get('tick')
    if not tick or 'bid' not in tick or 'ask' not in tick:
        errors.append("Missing or invalid tick data")
        return False, errors
    
    # Check indicators
    indicators = state.get('indicators', {})
    required_indicators = ['rsi_14', 'sma_50', 'sma_200', 'atr_14', 'volatility']
    missing_indicators = [ind for ind in required_indicators if indicators.get(ind) is None]
    if missing_indicators:
        errors.append(f"Missing indicators: {missing_indicators}")
    
    # Check candles
    candles = state.get('candles', {})
    if not candles or 'H1' not in candles or candles['H1'] is None:
        errors.append("Missing H1 candle data")
    
    # Check trend data
    trend = state.get('trend', {})
    if 'regime' not in trend or 'strength' not in trend:
        errors.append("Missing trend data")
    
    is_safe = len(errors) == 0
    return is_safe, errors


def check_spread_filter(state: Dict) -> Tuple[bool, str]:
    """
    Check if spread is within acceptable range for trading.
    
    Args:
        state: Market state dictionary
        
    Returns:
        Tuple of (pass_filter: bool, reason: str)
    """
    spread = state.get('tick', {}).get('spread', float('inf'))
    
    if spread > EvaluatorConfig.MAX_SPREAD_PIPS:
        return False, f"Spread too wide: {spread:.2f} pips (max: {EvaluatorConfig.MAX_SPREAD_PIPS})"
    
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
    volatility = state.get('indicators', {}).get('volatility', 0)
    
    if volatility < EvaluatorConfig.MIN_VOLATILITY_PCT:
        return False, f"Volatility too low: {volatility:.3f}% (market too quiet)"
    
    if volatility > EvaluatorConfig.MAX_VOLATILITY_PCT:
        # Allow high volatility if trend is strong
        if trend_strength < EvaluatorConfig.STRONG_TREND_STRENGTH:
            return False, f"Volatility too high: {volatility:.3f}% and trend weak"
        else:
            logger.info(f"High volatility {volatility:.3f}% but trend strong ({trend_strength:.2f}), allowing")
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
    candles = state.get('candles', {})
    signals = {}
    confidences = {}
    
    # Analyze each timeframe
    for tf in ['M1', 'M5', 'M15', 'H1']:
        if candles.get(tf) is None:
            logger.warning(f"Missing {tf} candles for multi-timeframe analysis")
            signals[tf] = 'hold'
            confidences[tf] = 0.0
            continue
        
        tf_candle = candles[tf]
        tf_indicators = tf_candle.get('indicators', {})
        
        # Get RSI for this timeframe
        rsi = tf_indicators.get('rsi_14')
        if rsi is None:
            signals[tf] = 'hold'
            confidences[tf] = 0.0
            continue
        
        # Generate signal from RSI
        if rsi < EvaluatorConfig.RSI_OVERSOLD:
            signals[tf] = 'buy'
            confidences[tf] = (EvaluatorConfig.RSI_OVERSOLD - rsi) / 30.0  # 0-1
        elif rsi > EvaluatorConfig.RSI_OVERBOUGHT:
            signals[tf] = 'sell'
            confidences[tf] = (rsi - EvaluatorConfig.RSI_OVERBOUGHT) / 30.0
        else:
            signals[tf] = 'hold'
            confidences[tf] = 0.0
        
        logger.debug(f"{tf} RSI: {rsi:.1f} → {signals[tf]} (confidence: {confidences[tf]:.2f})")
    
    # Aggregate signals (require H1 agreement if configured)
    buy_votes = sum(1 for s in signals.values() if s == 'buy')
    sell_votes = sum(1 for s in signals.values() if s == 'sell')
    
    logger.debug(f"Multi-TF votes: BUY={buy_votes}, SELL={sell_votes}, HOLD={4-buy_votes-sell_votes}")
    
    # Determine consensus signal
    if EvaluatorConfig.REQUIRE_HIGHER_TF_CONFIRMATION:
        # Require H1 to align
        if signals.get('H1') == 'buy' and buy_votes >= 2:
            return 'buy', min(0.9, (buy_votes / 4.0) * 0.9)
        elif signals.get('H1') == 'sell' and sell_votes >= 2:
            return 'sell', min(0.9, (sell_votes / 4.0) * 0.9)
        else:
            return 'hold', 0.0
    else:
        # Any alignment
        if buy_votes > sell_votes:
            return 'buy', (buy_votes / 4.0)
        elif sell_votes > buy_votes:
            return 'sell', (sell_votes / 4.0)
        else:
            return 'hold', 0.0


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
    if trend_signal in ['buy', 'sell']:
        confidence += trend_strength * 0.4
    
    # Multi-timeframe contribution (45%)
    if multitf_signal == trend_signal:
        confidence += multitf_confidence * 0.45
    elif multitf_signal == 'hold':
        confidence += multitf_confidence * 0.20  # Weak support
    
    # Sentiment contribution (15%)
    if sentiment_confidence > EvaluatorConfig.SENTIMENT_THRESHOLD:
        if trend_signal == 'buy' and sentiment_score > 0:
            confidence += min(sentiment_score, 1.0) * EvaluatorConfig.SENTIMENT_WEIGHT
        elif trend_signal == 'sell' and sentiment_score < 0:
            confidence += abs(min(sentiment_score, -1.0)) * EvaluatorConfig.SENTIMENT_WEIGHT
    
    return min(confidence, 1.0)


def evaluate_close_signal(state: Dict, open_position: Optional[Dict] = None) -> Tuple[str, float]:
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
        return 'hold', 0.0
    
    rsi = state.get('indicators', {}).get('rsi_14')
    if rsi is None:
        return 'hold', 0.0
    
    direction = open_position.get('direction')
    
    # Close if opposite extreme is reached
    if direction == 'buy' and rsi > EvaluatorConfig.RSI_OVERBOUGHT:
        logger.info(f"Close signal: Buy position, RSI overbought at {rsi:.1f}")
        return 'close', min(0.8, (rsi - EvaluatorConfig.RSI_OVERBOUGHT) / 30.0)
    
    if direction == 'sell' and rsi < EvaluatorConfig.RSI_OVERSOLD:
        logger.info(f"Close signal: Sell position, RSI oversold at {rsi:.1f}")
        return 'close', min(0.8, (EvaluatorConfig.RSI_OVERSOLD - rsi) / 30.0)
    
    return 'hold', 0.0


def evaluate(
    state: Dict,
    open_position: Optional[Dict] = None,
    require_high_confidence: bool = False,
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
        details['safety_errors'] = safety_errors
        return {
            'decision': Decision.HOLD.value,
            'confidence': 0.0,
            'reason': reason,
            'details': details,
        }
    
    logger.info("✓ Safety checks passed")
    
    # LAYER 2: Spread and Liquidity Filter
    # ============================================================
    logger.info("\n[LAYER 2] Spread Filter")
    spread_pass, spread_reason = check_spread_filter(state)
    details['spread_check'] = spread_reason
    
    if not spread_pass:
        logger.warning(f"❌ {spread_reason}")
        reason = f"Liquidity insufficient: {spread_reason}"
        return {
            'decision': Decision.HOLD.value,
            'confidence': 0.0,
            'reason': reason,
            'details': details,
        }
    
    logger.info(f"✓ {spread_reason}")
    
    # LAYER 3: Extract core indicators
    # ============================================================
    logger.info("\n[LAYER 3] Extract Indicators")
    indicators = state.get('indicators', {})
    rsi = indicators.get('rsi_14')
    sma_50 = indicators.get('sma_50')
    sma_200 = indicators.get('sma_200')
    atr = indicators.get('atr_14')
    volatility = indicators.get('volatility', 0)
    
    logger.debug(f"RSI: {rsi:.1f}, SMA50: {sma_50:.4f}, SMA200: {sma_200:.4f}, ATR: {atr:.4f}, Vol: {volatility:.3f}%")
    
    # LAYER 4: Volatility Filter
    # ============================================================
    logger.info("\n[LAYER 4] Volatility Filter")
    trend = state.get('trend', {})
    trend_strength = trend.get('strength', 0)
    volatility_pass, volatility_reason = check_volatility_filter(state, trend_strength)
    details['volatility_check'] = volatility_reason
    
    if not volatility_pass:
        logger.warning(f"❌ {volatility_reason}")
        reason = f"Volatility condition failed: {volatility_reason}"
        return {
            'decision': Decision.HOLD.value,
            'confidence': 0.0,
            'reason': reason,
            'details': details,
        }
    
    logger.info(f"✓ {volatility_reason}")
    
    # LAYER 5: Trend Detection
    # ============================================================
    logger.info("\n[LAYER 5] Trend Detection")
    trend_regime = trend.get('regime', 'sideways')
    logger.info(f"Trend Regime: {trend_regime.upper()} (strength: {trend_strength:.2f})")
    
    # Generate base trend signal
    trend_signal = 'hold'
    if trend_regime == 'uptrend' and trend_strength >= EvaluatorConfig.MIN_TREND_STRENGTH:
        trend_signal = 'buy'
        logger.info(f"→ Bullish trend signal (strength: {trend_strength:.2f})")
    elif trend_regime == 'downtrend' and trend_strength >= EvaluatorConfig.MIN_TREND_STRENGTH:
        trend_signal = 'sell'
        logger.info(f"→ Bearish trend signal (strength: {trend_strength:.2f})")
    else:
        logger.info(f"→ Insufficient trend strength ({trend_strength:.2f})")
    
    details['trend'] = {
        'regime': trend_regime,
        'strength': trend_strength,
        'signal': trend_signal,
    }
    
    # LAYER 6: Multi-Timeframe Confirmation
    # ============================================================
    logger.info("\n[LAYER 6] Multi-Timeframe Confirmation")
    multitf_signal, multitf_confidence = check_multitimeframe_alignment(state)
    logger.info(f"Multi-TF Signal: {multitf_signal.upper()} (confidence: {multitf_confidence:.2f})")
    
    details['multitf'] = {
        'signal': multitf_signal,
        'confidence': multitf_confidence,
    }
    
    # LAYER 7: Sentiment Analysis
    # ============================================================
    logger.info("\n[LAYER 7] Sentiment Analysis")
    sentiment = state.get('sentiment', {})
    sentiment_score = sentiment.get('score', 0)
    sentiment_confidence = sentiment.get('confidence', 0)
    logger.info(f"Sentiment: {sentiment_score:.2f} (confidence: {sentiment_confidence:.2f})")
    
    details['sentiment'] = {
        'score': sentiment_score,
        'confidence': sentiment_confidence,
        'source': sentiment.get('source', 'unknown'),
    }
    
    # LAYER 8: Position Management (Close Check)
    # ============================================================
    logger.info("\n[LAYER 8] Position Management")
    if open_position:
        close_decision, close_confidence = evaluate_close_signal(state, open_position)
        if close_decision == 'close':
            logger.info(f"✓ CLOSE DECISION (confidence: {close_confidence:.2f})")
            return {
                'decision': Decision.CLOSE.value,
                'confidence': close_confidence,
                'reason': f"Close position signal triggered",
                'details': {**details, 'close_confidence': close_confidence},
            }
    
    # LAYER 9: Final Decision Logic
    # ============================================================
    logger.info("\n[LAYER 9] Final Decision")
    
    # Combine signals
    if trend_signal in ['buy', 'sell']:
        if trend_signal == multitf_signal or multitf_signal == 'hold':
            # Aligned or multi-tf doesn't contradict
            decision_value = trend_signal
            confidence = calculate_signal_confidence(
                trend_signal, trend_strength,
                multitf_signal, multitf_confidence,
                sentiment_score, sentiment_confidence
            )
            
            if confidence < EvaluatorConfig.MIN_TREND_STRENGTH:
                logger.info(f"Signal too weak: {confidence:.2f} < {EvaluatorConfig.MIN_TREND_STRENGTH}")
                decision_value = 'hold'
        else:
            # Conflicting signals
            logger.warning(f"Conflicting signals: trend={trend_signal}, multitf={multitf_signal}")
            decision_value = 'hold'
            confidence = 0.0
    else:
        decision_value = 'hold'
    
    # Apply high confidence requirement if needed
    if require_high_confidence and decision_value in ['buy', 'sell'] and confidence < 0.6:
        logger.info(f"Confidence {confidence:.2f} below high-confidence threshold (0.6)")
        decision_value = 'hold'
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
    
    details['final_confidence'] = confidence
    
    return {
        'decision': decision.value,
        'confidence': confidence,
        'reason': reason,
        'details': details,
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
) -> Dict:
    """
    Evaluate market state using CausalEvaluator (Stockfish-style).
    
    This integrates the deterministic, rule-based CausalEvaluator which combines
    8 market factors into a single evaluation score [-1, +1].
    
    Args:
        state: Traditional market state dictionary (legacy format)
        causal_evaluator: Initialized CausalEvaluator instance
        market_state: CausalEvaluator MarketState dataclass (preferred)
        
    Returns:
        Dict with decision, confidence, reason, and full causal reasoning
        
    Raises:
        ValueError: If causal_evaluator is None or market_state not properly configured
    """
    if causal_evaluator is None:
        raise ValueError("causal_evaluator cannot be None")
    
    if market_state is None:
        raise ValueError("market_state (CausalEvaluator.MarketState) required for causal evaluation")
    
    # Evaluate using CausalEvaluator
    try:
        result = causal_evaluator.evaluate(market_state)
    except Exception as e:
        logger.error(f"CausalEvaluator failed: {e}")
        raise
    
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
    for factor in result.scoring_factors:
        factor_explanations.append({
            'factor': factor.factor_name,
            'score': factor.score,
            'weight': factor.weight,
            'explanation': factor.explanation,
        })
    
    return {
        'decision': decision,
        'confidence': confidence,
        'reason': reason,
        'eval_score': eval_score,
        'causal_reasoning': factor_explanations,
        'timestamp': result.timestamp,
        'evaluator_mode': 'causal',
        'details': {
            'causal_eval': eval_score,
            'factors': factor_explanations,
        },
    }


def create_evaluator_factory(
    use_causal: bool = False,
    use_policy_engine: bool = False,
    **causal_kwargs
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
            policy_engine = PolicyEngine(verbose=causal_kwargs.get('verbose', False))
            
            def integrated_wrapper(
                state: Dict,
                market_state: Optional[Any] = None,
                open_position: Optional[Dict] = None,
                position_state: Optional[Any] = None,
                risk_config: Optional[Any] = None,
                daily_loss_pct: float = 0.0
            ) -> Dict:
                """Integrated causal+policy evaluation"""
                
                if market_state is None:
                    logger.warning("market_state not provided; falling back to traditional evaluator")
                    return evaluate(state, open_position=open_position)
                
                # Use defaults if not provided
                if position_state is None:
                    from engine.policy_engine import PositionState, PositionSide
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
                        verbose=causal_kwargs.get('verbose', False)
                    )
                    
                    # Convert to legacy format for compatibility
                    return {
                        'decision': result['action'].lower() if isinstance(result['action'], str) else result['action'].value.lower(),
                        'confidence': result['confidence'],
                        'reason': f"CausalEval+Policy: {result['decision_zone']}",
                        'eval_score': result['eval_score'],
                        'details': {
                            'causal_eval': result['eval_score'],
                            'policy_action': result['action'],
                            'target_size': result['target_size'],
                            'evaluation_zone': result['decision_zone'],
                            'causal_reasoning': result['reasoning']['eval'],
                            'policy_reasoning': result['reasoning']['policy'],
                        },
                        'integrated_mode': True,
                        'causal_evaluator': True,
                        'policy_engine': True,
                        'deterministic': True,
                        'lookahead_safe': True,
                    }
                except Exception as e:
                    logger.error(f"Integrated pipeline failed: {e}; falling back to traditional")
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
                open_position: Optional[Dict] = None
            ) -> Dict:
                """Wrapper for causal evaluation"""
                if market_state is None:
                    # If no causal market_state provided, use traditional evaluator
                    logger.warning("market_state not provided; falling back to traditional evaluator")
                    return evaluate(state, open_position=open_position)

                return evaluate_with_causal(
                    state=state,
                    causal_evaluator=causal_evaluator,
                    market_state=market_state
                )

            return causal_eval_wrapper

        except ImportError:
            logger.error("CausalEvaluator not available; falling back to traditional evaluator")
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


if __name__ == '__main__':
    """Example usage and testing"""
    
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 70)
    print("EVALUATOR ENGINE - TEST RUN")
    print("=" * 70)
    
    # Create a mock state (would come from state_builder.py)
    mock_state = {
        'timestamp': 1705441218.528,
        'symbol': 'EURUSD',
        'tick': {
            'bid': 1.0850,
            'ask': 1.0852,
            'spread': 2.0,
            'last_tick_time': 1705441213,
        },
        'indicators': {
            'rsi_14': 35.0,  # Oversold = buy signal
            'sma_50': 1.0835,
            'sma_200': 1.0800,
            'atr_14': 0.0012,
            'volatility': 0.5,
        },
        'trend': {
            'regime': 'uptrend',
            'strength': 0.75,
        },
        'sentiment': {
            'score': 0.2,
            'confidence': 0.3,
            'source': 'placeholder',
        },
        'candles': {
            'M1': {
                'indicators': {'rsi_14': 25.0},
                'latest': {'time': 1705441210},
            },
            'M5': {
                'indicators': {'rsi_14': 32.0},
                'latest': {'time': 1705441210},
            },
            'M15': {
                'indicators': {'rsi_14': 35.0},
                'latest': {'time': 1705441210},
            },
            'H1': {
                'indicators': {'rsi_14': 38.0},
                'latest': {'time': 1705441210},
            },
        },
        'health': {
            'is_stale': False,
            'last_update': 1705441218.528,
            'errors': [],
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
    mock_state_overbought['indicators']['rsi_14'] = 75.0
    
    open_pos = {'direction': 'buy', 'entry_price': 1.0840}
    result = evaluate(mock_state_overbought, open_position=open_pos)
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reason: {result['reason']}")
    
    # Test 3: Sell signal
    print("\n[TEST 3] Sell Signal (Downtrend + Overbought)")
    mock_state_sell = mock_state.copy()
    mock_state_sell['trend']['regime'] = 'downtrend'
    mock_state_sell['indicators']['rsi_14'] = 75.0
    mock_state_sell['sentiment']['score'] = -0.3
    
    result = evaluate(mock_state_sell)
    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reason: {result['reason']}")
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70 + "\n")
