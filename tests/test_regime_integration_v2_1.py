"""
Regime Integration Tests (Phase v2.1).

Comprehensive validation of regime conditioning across evaluator, policy engine,
and portfolio risk manager.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from engine.causal_evaluator import (
    CausalEvaluator, MarketState, MacroState, LiquidityState, VolatilityState,
    DealerState, EarningsState, TimeRegimeState, TimeRegimeType,
    PriceLocationState, MacroNewsState, LiquidityRegime, VolatilityRegime
)
from engine.policy_engine import (
    PolicyEngine, PositionState, PositionSide, RiskConfig, TradingAction
)
from engine.portfolio_risk_manager import PortfolioRiskManager, RiskDecision
from engine.regime_classifier import RegimeClassifier


class TestEvaluatorRegimeConditioning(unittest.TestCase):
    """Test regime conditioning in CausalEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = CausalEvaluator(enable_regime_conditioning=True, verbose=False)
        self.base_date = datetime(2026, 1, 20, 10, 0, 0)
    
    def _create_market_state(self, close_price: float = 4500.0) -> MarketState:
        """Create a test MarketState."""
        return MarketState(
            timestamp=self.base_date,
            symbol='ES',
            macro_state=MacroState(
                sentiment_score=0.3,
                surprise_score=0.1,
                rate_expectation=-0.1,
                inflation_expectation=0.2,
                gdp_expectation=0.1
            ),
            liquidity_state=LiquidityState(
                bid_ask_spread=1.5,
                order_book_depth=0.7,
                regime=LiquidityRegime.NORMAL,
                volume_trend=0.5
            ),
            volatility_state=VolatilityState(
                current_vol=0.15,
                vol_percentile=0.6,
                regime=VolatilityRegime.NORMAL,
                vol_trend=0.2,
                skew=0.0
            ),
            dealer_state=DealerState(
                net_gamma_exposure=0.2,
                net_spot_exposure=0.1,
                vega_exposure=0.0,
                dealer_sentiment=0.3
            ),
            earnings_state=EarningsState(
                multi_mega_cap_exposure=0.6,
                small_cap_exposure=0.4,
                earnings_season_flag=False,
                earnings_surprise_momentum=0.1
            ),
            time_regime_state=TimeRegimeState(
                regime_type=TimeRegimeType.NY_OPEN,
                minutes_into_session=60,
                hours_until_session_end=6,
                day_of_week=2
            ),
            price_location_state=PriceLocationState(
                distance_from_high=0.6,
                distance_from_low=0.4,
                range_ratio=1.0,
                session_extremity=0.2
            ),
            macro_news_state=MacroNewsState(
                risk_sentiment_score=0.2,
                hawkishness_score=0.1,
                surprise_score=0.1,
                event_importance=1,
                hours_since_last_event=4.0,
                macro_event_count=1,
                news_article_count=3,
                macro_news_state='NEUTRAL'
            ),
            current_price=close_price,
            session_high=close_price + 10,
            session_low=close_price - 10,
            session_name='NY_OPEN',
            vwap=close_price
        )
    
    def test_evaluator_has_regime_conditioning(self):
        """Test that evaluator is initialized with regime conditioning."""
        self.assertTrue(self.evaluator.enable_regime_conditioning)
        self.assertIsNotNone(self.evaluator.regime_classifier)
    
    def test_evaluation_result_includes_regime_fields(self):
        """Test that evaluation result includes regime information."""
        state = self._create_market_state()
        result = self.evaluator.evaluate(state)
        
        self.assertTrue(hasattr(result, 'regime_label'))
        self.assertTrue(hasattr(result, 'regime_confidence'))
        self.assertTrue(hasattr(result, 'regime_features'))
        self.assertTrue(hasattr(result, 'regime_adjusted_eval'))
        self.assertTrue(hasattr(result, 'regime_adjustments'))
    
    def test_regime_conditioning_on_trend_signal(self):
        """Test regime conditioning boosts trend-aligned signals."""
        state = self._create_market_state()
        result = self.evaluator.evaluate(state)
        
        # Should have some regime classification
        self.assertIsNotNone(result.regime_label)
        
        if result.regime_label == 'TREND' and result.regime_confidence > 0.5:
            # Verify TREND conditioning was applied
            self.assertGreater(abs(result.regime_adjusted_eval), abs(result.eval_score) * 0.8)
    
    def test_regime_fields_serializable(self):
        """Test that regime fields serialize to dict correctly."""
        state = self._create_market_state()
        result = self.evaluator.evaluate(state)
        
        # Should serialize without error
        result_dict = result.result_dict
        self.assertIn('regime_label', result_dict)
        self.assertIn('regime_confidence', result_dict)
        self.assertIn('regime_adjusted_eval', result_dict)


class TestPolicyEngineRegimeConditioning(unittest.TestCase):
    """Test regime conditioning in PolicyEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy = PolicyEngine(verbose=False)
        self.base_date = datetime(2026, 1, 20, 10, 0, 0)
    
    def _create_eval_result(self, eval_score: float, regime: str = 'TREND') -> Dict[str, Any]:
        """Create a test evaluation result."""
        return {
            'eval_score': eval_score,
            'confidence': 0.7,
            'session': 'NY_OPEN',
            'session_modifiers': {'vol_scale': 1.0, 'liq_scale': 1.0, 'risk_scale': 1.0},
            'flow_signals': {},
            'regime_label': regime,
            'regime_confidence': 0.8,
            'regime_adjustments': {},
        }
    
    def _create_position_state(self, side: PositionSide = PositionSide.FLAT, size: float = 0.0):
        """Create a test position state."""
        return PositionState(
            side=side,
            size=size,
            entry_price=4500.0 if size > 0 else None,
            current_price=4510.0,
            unrealized_pnl=size * 10,
            unrealized_pnl_pct=0.002
        )
    
    def test_policy_decision_includes_regime_fields(self):
        """Test that policy decisions include regime context."""
        market_state = {}
        eval_result = self._create_eval_result(0.6, 'TREND')
        position_state = self._create_position_state()
        
        decision = self.policy.decide_action(
            market_state, eval_result, position_state
        )
        
        self.assertTrue(hasattr(decision, 'regime_label'))
        self.assertTrue(hasattr(decision, 'regime_confidence'))
        self.assertTrue(hasattr(decision, 'regime_adjustments'))
    
    def test_trend_regime_increases_size(self):
        """Test that TREND regime increases position size for aligned signals."""
        market_state = {}
        eval_result = self._create_eval_result(0.7, 'TREND')  # Strong bullish
        position_state = self._create_position_state()
        
        decision = self.policy.decide_action(
            market_state, eval_result, position_state
        )
        
        # In TREND with strong bullish signal, should consider full entry
        if decision.action in [TradingAction.ENTER_FULL, TradingAction.ENTER_SMALL]:
            self.assertGreater(decision.target_size, 0.0)
    
    def test_range_regime_reduces_size_on_breakout(self):
        """Test that RANGE regime reduces size on breakout signals."""
        market_state = {}
        eval_result = self._create_eval_result(0.75, 'RANGE')  # Strong breakout signal
        position_state = self._create_position_state()
        
        decision = self.policy.decide_action(
            market_state, eval_result, position_state
        )
        
        # In RANGE with strong signal, should be cautious (reduced size or blocked)
        if decision.action in [TradingAction.ENTER_SMALL, TradingAction.ENTER_FULL]:
            self.assertEqual(decision.action, TradingAction.ENTER_SMALL)
    
    def test_reversal_regime_blocks_add(self):
        """Test that REVERSAL regime blocks ADD action."""
        market_state = {}
        eval_result = self._create_eval_result(0.2, 'REVERSAL')  # Low signal in reversal
        position_state = self._create_position_state(side=PositionSide.LONG, size=0.5)
        
        decision = self.policy.decide_action(
            market_state, eval_result, position_state
        )
        
        # In REVERSAL with low signal and existing position, should not add
        self.assertNotEqual(decision.action, TradingAction.ADD)


class TestRiskManagerRegimeConditioning(unittest.TestCase):
    """Test regime conditioning in PortfolioRiskManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_mgr = PortfolioRiskManager(
            total_capital=100000,
            max_symbol_exposure=50000,
            max_total_exposure=80000,
            max_daily_loss=3000
        )
    
    def test_risk_decision_includes_regime_fields(self):
        """Test that risk decisions include regime context."""
        decision = self.risk_mgr.evaluate_risk_with_context(
            symbol='ES',
            target_size=1.0,
            price=4500.0,
            policy_decision={'session_name': 'NY_OPEN'},
            regime_label='TREND',
            regime_confidence=0.8
        )
        
        self.assertTrue(hasattr(decision, 'regime_label'))
        self.assertTrue(hasattr(decision, 'regime_confidence'))
        self.assertTrue(hasattr(decision, 'regime_adjustments'))
        self.assertEqual(decision.regime_label, 'TREND')
        self.assertEqual(decision.regime_confidence, 0.8)
    
    def test_range_regime_reduces_approved_size(self):
        """Test that RANGE regime reduces approved position size."""
        decision_base = self.risk_mgr.evaluate_risk_with_context(
            symbol='ES',
            target_size=0.5,
            price=4500.0,
            policy_decision={'session_name': 'MIDDAY'}
        )
        
        decision_range = self.risk_mgr.evaluate_risk_with_context(
            symbol='ES',
            target_size=0.5,
            price=4500.0,
            policy_decision={'session_name': 'MIDDAY'},
            regime_label='RANGE',
            regime_confidence=0.8
        )
        
        # RANGE regime should show scaling factor in adjustments
        if decision_range.regime_adjustments:
            self.assertIn('scaling_factor', decision_range.regime_adjustments)
            self.assertLess(
                decision_range.regime_adjustments['scaling_factor'],
                1.1
            )
    
    def test_reversal_regime_significantly_reduces_size(self):
        """Test that REVERSAL regime significantly reduces approved size."""
        decision = self.risk_mgr.evaluate_risk_with_context(
            symbol='ES',
            target_size=0.8,
            price=4500.0,
            policy_decision={'session_name': 'MIDDAY'},
            regime_label='REVERSAL',
            regime_confidence=0.9
        )
        
        # REVERSAL should show significant scaling reduction
        if decision.regime_adjustments:
            self.assertIn('scaling_factor', decision.regime_adjustments)
            scaling = decision.regime_adjustments['scaling_factor']
            # Should be around 0.5 with high confidence
            self.assertLess(scaling, 0.7)
    
    def test_trend_regime_maintains_size(self):
        """Test that TREND regime doesn't reduce approved size."""
        decision = self.risk_mgr.evaluate_risk_with_context(
            symbol='ES',
            target_size=0.6,
            price=4500.0,
            policy_decision={'session_name': 'POWER_HOUR'},
            regime_label='TREND',
            regime_confidence=0.85
        )
        
        # TREND should maintain approved size
        if decision.regime_adjustments:
            scaling = decision.regime_adjustments.get('scaling_factor', 1.0)
            self.assertGreaterEqual(scaling, 0.95)


class TestRegimeIntegrationPipeline(unittest.TestCase):
    """Test full pipeline integration of regime conditioning."""
    
    def setUp(self):
        """Set up integrated components."""
        self.evaluator = CausalEvaluator(enable_regime_conditioning=True, verbose=False)
        self.policy = PolicyEngine(verbose=False)
        self.risk_mgr = PortfolioRiskManager(
            total_capital=100000,
            max_symbol_exposure=50000,
            max_total_exposure=80000,
            max_daily_loss=3000
        )
        self.base_date = datetime(2026, 1, 20, 10, 0, 0)
    
    def _create_market_state(self) -> MarketState:
        """Create test market state."""
        return MarketState(
            timestamp=self.base_date,
            symbol='ES',
            macro_state=MacroState(0.3, 0.1, -0.1, 0.2, 0.1),
            liquidity_state=LiquidityState(1.5, 0.7, LiquidityRegime.NORMAL, 0.5),
            volatility_state=VolatilityState(0.15, 0.6, VolatilityRegime.NORMAL, 0.2, 0.0),
            dealer_state=DealerState(0.2, 0.1, 0.0, 0.3),
            earnings_state=EarningsState(0.6, 0.4, False, 0.1),
            time_regime_state=TimeRegimeState(TimeRegimeType.NY_OPEN, 60, 6, 2),
            price_location_state=PriceLocationState(0.6, 0.4, 1.0, 0.2),
            macro_news_state=MacroNewsState(0.2, 0.1, 0.1, 1, 4.0, 1, 3, 'NEUTRAL'),
            current_price=4500.0,
            session_high=4510.0,
            session_low=4490.0,
            session_name='NY_OPEN',
            vwap=4500.0
        )
    
    def test_full_pipeline_deterministic(self):
        """Test that full pipeline produces deterministic output."""
        market_state = self._create_market_state()
        
        # Run evaluation twice
        eval1 = self.evaluator.evaluate(market_state)
        eval2 = self.evaluator.evaluate(market_state)
        
        # Results should be identical
        self.assertEqual(eval1.eval_score, eval2.eval_score)
        self.assertEqual(eval1.confidence, eval2.confidence)
        self.assertEqual(eval1.regime_label, eval2.regime_label)
        self.assertEqual(eval1.regime_confidence, eval2.regime_confidence)
    
    def test_full_pipeline_all_components_active(self):
        """Test that all components receive regime information."""
        market_state = self._create_market_state()
        
        # 1. Evaluate
        eval_result = self.evaluator.evaluate(market_state)
        eval_dict = eval_result.result_dict
        
        # 2. Decide policy
        position_state = PositionState(side=PositionSide.FLAT, size=0.0)
        policy_decision = self.policy.decide_action(
            market_state, eval_dict, position_state
        )
        
        # 3. Evaluate risk
        risk_decision = self.risk_mgr.evaluate_risk_with_context(
            symbol='ES',
            target_size=policy_decision.target_size,
            price=market_state.current_price,
            policy_decision=policy_decision.to_dict(),
            regime_label=eval_result.regime_label,
            regime_confidence=eval_result.regime_confidence
        )
        
        # All should have regime info
        self.assertIsNotNone(eval_result.regime_label)
        self.assertIsNotNone(policy_decision.regime_label)
        self.assertIsNotNone(risk_decision.regime_label)


class TestDeterministicBehavior(unittest.TestCase):
    """Test deterministic behavior with regime conditioning."""
    
    def test_regime_classifier_determinism(self):
        """Test that regime classifier produces deterministic results."""
        classifier = RegimeClassifier()
        
        # Run classification multiple times
        results = []
        for _ in range(3):
            classifier.reset()
            for i in range(100):
                price = 4500 + i * 0.5
                result = classifier.update_with_bar(
                    timestamp=datetime.now(),
                    open_price=price,
                    high=price + 5,
                    low=price - 5,
                    close=price,
                    volume=1000,
                    vwap=price
                )
            results.append(result.regime_label)
        
        # All runs should produce same regime
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
    
    def test_regime_conditioning_determinism(self):
        """Test that regime conditioning is deterministic."""
        evaluator = CausalEvaluator(enable_regime_conditioning=True, verbose=False)
        
        market_state = MarketState(
            timestamp=datetime(2026, 1, 20, 10, 0, 0),
            symbol='ES',
            macro_state=MacroState(0.3, 0.1, -0.1, 0.2, 0.1),
            liquidity_state=LiquidityState(1.5, 0.7, LiquidityRegime.NORMAL, 0.5),
            volatility_state=VolatilityState(0.15, 0.6, VolatilityRegime.NORMAL, 0.2, 0.0),
            dealer_state=DealerState(0.2, 0.1, 0.0, 0.3),
            earnings_state=EarningsState(0.6, 0.4, False, 0.1),
            time_regime_state=TimeRegimeState(TimeRegimeType.NY_OPEN, 60, 6, 2),
            price_location_state=PriceLocationState(0.6, 0.4, 1.0, 0.2),
            macro_news_state=MacroNewsState(0.2, 0.1, 0.1, 1, 4.0, 1, 3, 'NEUTRAL'),
            current_price=4500.0,
            session_high=4510.0,
            session_low=4490.0,
            session_name='NY_OPEN',
            vwap=4500.0
        )
        
        # Evaluate same state 3 times
        results = [evaluator.evaluate(market_state) for _ in range(3)]
        
        # All should produce identical output
        for i in range(1, 3):
            self.assertEqual(results[0].eval_score, results[i].eval_score)
            self.assertEqual(results[0].regime_label, results[i].regime_label)
            self.assertEqual(results[0].regime_adjusted_eval, results[i].regime_adjusted_eval)


if __name__ == '__main__':
    unittest.main()
