#!/usr/bin/env python3
"""
Unified Trading Pipeline Integration: CausalEvaluator + PolicyEngine

This module provides the integration layer between:
1. CausalEvaluator: Stockfish-style market evaluation
2. PolicyEngine: Deterministic, risk-aware trading decisions
3. State Builder: Market state construction
4. Tournament: Official trading evaluation

Philosophy:
- DETERMINISTIC: Same inputs → same outputs (no randomness)
- TIME-CAUSAL: No lookahead bias, respects temporal ordering
- RULE-BASED: All logic exposed, fully explainable
- PRODUCTION-READY: Full error handling, safety checks

Usage:
    from engine.integration import evaluate_and_decide
    
    result = evaluate_and_decide(
        market_state=market_state,
        position_state=position_state,
        risk_config=risk_config,
        causal_evaluator=evaluator,
        policy_engine=engine,
        daily_loss_pct=0.005
    )
    
    print(f"Action: {result['action']}")
    print(f"Target Size: {result['target_size']}")
    print(f"Confidence: {result['confidence']}")
"""

import logging
from typing import Dict, Optional, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class IntegrationError(Exception):
    """Base exception for integration errors"""
    pass


def evaluate_and_decide(
    market_state: Any,
    position_state: Any,
    risk_config: Any,
    causal_evaluator: Any,
    policy_engine: Any,
    daily_loss_pct: float = 0.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Unified evaluation + decision pipeline combining CausalEvaluator and PolicyEngine.
    
    This is the core integration function that:
    1. Gets market evaluation from CausalEvaluator
    2. Applies risk-aware decision logic from PolicyEngine
    3. Returns complete decision with full reasoning
    
    Args:
        market_state: MarketState from state builder (with all 8 causal components)
        position_state: PositionState dataclass (side, size, entry_price, etc.)
        risk_config: RiskConfig dataclass (max risk, thresholds, etc.)
        causal_evaluator: Initialized CausalEvaluator instance
        policy_engine: Initialized PolicyEngine instance
        daily_loss_pct: Current daily loss as % (0.0 to 1.0)
        verbose: Enable detailed logging
        
    Returns:
        Dict with:
        {
            'action': TradingAction enum value (ENTER_FULL, HOLD, EXIT, etc.)
            'target_size': Normalized position size (0 to max_position_size)
            'confidence': Decision confidence (0.0 to 1.0)
            'eval_score': Causal evaluation score (-1.0 to +1.0)
            'eval_confidence': Evaluation confidence (0.0 to 1.0)
            'decision_zone': Evaluation zone (NO_TRADE, LOW, MEDIUM, HIGH conviction)
            'reasoning': {
                'eval': List of evaluation factors
                'policy': List of policy decision factors
            },
            'timestamp': Decision timestamp
            'deterministic': True (no randomness)
            'lookahead_safe': True (no future data)
            'causal_mode': 'deterministic'
        }
        
    Raises:
        IntegrationError: If evaluators not properly initialized or data invalid
        ValueError: If market_state or position_state invalid
    """
    
    try:
        # ====================================================================
        # STEP 1: Validate inputs
        # ====================================================================
        
        if causal_evaluator is None:
            raise IntegrationError("causal_evaluator cannot be None")
        if policy_engine is None:
            raise IntegrationError("policy_engine cannot be None")
        if market_state is None:
            raise ValueError("market_state cannot be None")
        if position_state is None:
            raise ValueError("position_state cannot be None")
        if risk_config is None:
            raise ValueError("risk_config cannot be None")
            
        if verbose:
            logger.info(f"[Integration] Starting evaluation + decision pipeline")
            logger.info(f"  Position: {position_state.side.value if hasattr(position_state.side, 'value') else position_state.side}")
            logger.info(f"  Daily Loss: {daily_loss_pct:.2%}")
        
        # ====================================================================
        # STEP 2: Get market evaluation from CausalEvaluator
        # ====================================================================
        
        if verbose:
            logger.info(f"[Integration] Running CausalEvaluator...")
        
        eval_result = causal_evaluator.evaluate(market_state)
        
        if verbose:
            logger.info(f"  Eval Score: {eval_result.eval_score:.3f}")
            logger.info(f"  Confidence: {eval_result.confidence:.3f}")
        
        # Convert CausalEvaluator result to dict format for PolicyEngine
        eval_dict = {
            'eval_score': eval_result.eval_score,
            'confidence': eval_result.confidence,
            'factors': [
                {
                    'factor': f.factor_name,
                    'score': f.score,
                    'weight': f.weight,
                    'explanation': f.explanation,
                }
                for f in eval_result.scoring_factors
            ]
        }
        
        # ====================================================================
        # STEP 3: Get decision from PolicyEngine
        # ====================================================================
        
        if verbose:
            logger.info(f"[Integration] Running PolicyEngine...")
        
        policy_decision = policy_engine.decide_action(
            market_state=market_state,
            eval_result=eval_dict,
            position_state=position_state,
            risk_config=risk_config,
            daily_loss_pct=daily_loss_pct
        )
        
        if verbose:
            logger.info(f"  Action: {policy_decision.action.value}")
            logger.info(f"  Target Size: {policy_decision.target_size:.4f}")
            logger.info(f"  Policy Confidence: {policy_decision.confidence:.3f}")
        
        # ====================================================================
        # STEP 4: Determine evaluation zone
        # ====================================================================
        
        eval_score = eval_result.eval_score
        abs_eval = abs(eval_score)
        
        if abs_eval < 0.2:
            zone = "NO_TRADE"
        elif abs_eval < 0.5:
            zone = "LOW_CONVICTION"
        elif abs_eval < 0.8:
            zone = "MEDIUM_CONVICTION"
        else:
            zone = "HIGH_CONVICTION"
        
        # ====================================================================
        # STEP 5: Combine reasoning from both evaluators
        # ====================================================================
        
        combined_reasoning = {
            'eval': eval_dict.get('factors', []),
            'policy': [
                {
                    'factor': factor.factor,
                    'detail': factor.detail,
                    'weight': factor.weight,
                }
                for factor in policy_decision.reasoning
            ]
        }
        
        # ====================================================================
        # STEP 6: Build comprehensive result
        # ====================================================================
        
        result = {
            # Decision outputs
            'action': policy_decision.action.value,
            'target_size': policy_decision.target_size,
            'confidence': policy_decision.confidence,
            
            # Evaluation outputs
            'eval_score': eval_result.eval_score,
            'eval_confidence': eval_result.confidence,
            'decision_zone': zone,
            
            # Reasoning chains
            'reasoning': combined_reasoning,
            
            # Metadata
            'timestamp': datetime.utcnow().isoformat(),
            'deterministic': True,
            'lookahead_safe': True,
            'causal_mode': 'deterministic',
            
            # Source info
            'causal_evaluator': 'enabled',
            'policy_engine': 'enabled',
        }
        
        if verbose:
            logger.info(f"[Integration] Decision complete: {result['action']}")
            logger.info(f"[Integration] Reasoning factors: {len(combined_reasoning['policy'])} policy, {len(combined_reasoning['eval'])} eval")
        
        return result
        
    except (IntegrationError, ValueError) as e:
        logger.error(f"[Integration] Error in pipeline: {e}")
        raise
    except Exception as e:
        logger.error(f"[Integration] Unexpected error: {e}")
        raise IntegrationError(f"Integration pipeline failed: {e}") from e


def create_integrated_evaluator_factory(
    use_causal_policy: bool = False,
    **kwargs
) -> callable:
    """
    Factory function to create integrated causal + policy evaluator.
    
    Args:
        use_causal_policy: If True, use CausalEvaluator + PolicyEngine
        **kwargs: Arguments for CausalEvaluator and PolicyEngine
            - causal_kwargs: Dict of args for CausalEvaluator
            - policy_kwargs: Dict of args for PolicyEngine
            
    Returns:
        Evaluator function that accepts (market_state, position_state, risk_config)
        
    Raises:
        ImportError: If CausalEvaluator or PolicyEngine not available
    """
    
    if not use_causal_policy:
        # Return legacy evaluator
        from engine.evaluator import evaluate
        return lambda state, **kw: evaluate(state)
    
    try:
        from engine.causal_evaluator import CausalEvaluator
        from engine.policy_engine import PolicyEngine, RiskConfig
        
        # Extract kwargs
        causal_kwargs = kwargs.get('causal_kwargs', {})
        policy_kwargs = kwargs.get('policy_kwargs', {})
        
        # Initialize evaluators
        causal_evaluator = CausalEvaluator(**causal_kwargs)
        policy_engine = PolicyEngine(**policy_kwargs)
        
        # Return wrapper function
        def integrated_evaluator(
            market_state,
            position_state=None,
            risk_config: Optional[RiskConfig] = None,
            daily_loss_pct: float = 0.0,
            **extra_kwargs
        ) -> Dict[str, Any]:
            """Integrated evaluation wrapper"""
            
            # Use default risk config if not provided
            if risk_config is None:
                risk_config = RiskConfig()
            
            # Use default position state if not provided
            if position_state is None:
                from engine.policy_engine import PositionState, PositionSide
                position_state = PositionState(side=PositionSide.FLAT, size=0.0)
            
            # Run integrated pipeline
            return evaluate_and_decide(
                market_state=market_state,
                position_state=position_state,
                risk_config=risk_config,
                causal_evaluator=causal_evaluator,
                policy_engine=policy_engine,
                daily_loss_pct=daily_loss_pct,
                verbose=kwargs.get('verbose', False)
            )
        
        return integrated_evaluator
        
    except ImportError as e:
        logger.error(f"Cannot import required evaluators: {e}")
        raise


if __name__ == '__main__':
    """Test the integrated pipeline"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from engine.causal_evaluator import CausalEvaluator, MarketState, TrendState
    from engine.policy_engine import PolicyEngine, PositionState, PositionSide, RiskConfig
    from analytics.data_loader import MarketState as LegacyMarketState
    
    print("\n" + "="*70)
    print("INTEGRATED CAUSAL EVAL + POLICY ENGINE PIPELINE TEST")
    print("="*70 + "\n")
    
    # Initialize evaluators
    causal_eval = CausalEvaluator(verbose=False, official_mode=True)
    policy_engine = PolicyEngine(verbose=False, official_mode=True)
    
    # Create mock market state (would come from real data)
    print("[1] Creating mock market state...")
    
    # This is simplified - real usage would have full market state from data loader
    mock_market_state = type('MockMarketState', (), {
        'volatility_state': type('VolState', (), {'regime': 'MEDIUM'})(),
        'liquidity_state': type('LiqState', (), {'regime': 'NORMAL'})(),
        'trend_state': TrendState(direction='UP', strength=0.65),
        'momentum_state': type('MomState', (), {'score': 0.55})(),
    })()
    
    # Create position state
    position_state = PositionState(
        side=PositionSide.FLAT,
        size=0.0,
        entry_price=None,
        current_price=1.0850
    )
    
    # Create risk config
    risk_config = RiskConfig(max_risk_per_trade=0.01)
    
    # Run integrated pipeline
    print("[2] Running integrated pipeline...")
    result = evaluate_and_decide(
        market_state=mock_market_state,
        position_state=position_state,
        risk_config=risk_config,
        causal_evaluator=causal_eval,
        policy_engine=policy_engine,
        daily_loss_pct=0.005,
        verbose=True
    )
    
    # Display results
    print("\n" + "="*70)
    print("PIPELINE RESULTS")
    print("="*70)
    print(f"Action: {result['action']}")
    print(f"Target Size: {result['target_size']:.4f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Eval Score: {result['eval_score']:.3f}")
    print(f"Decision Zone: {result['decision_zone']}")
    print(f"Deterministic: {result['deterministic']}")
    print(f"Lookahead Safe: {result['lookahead_safe']}")
    print(f"Reasoning factors: {len(result['reasoning']['eval'])} eval + {len(result['reasoning']['policy'])} policy")
    print("="*70 + "\n")
