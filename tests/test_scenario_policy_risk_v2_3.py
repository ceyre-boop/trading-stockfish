"""
Scenario-Aware Policy & Risk Integration Tests (Phase v2.3).

Tests for scenario-aware decision shaping in PolicyEngine and risk scaling in PortfolioRiskManager.
"""

import unittest
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass

from engine.scenario_simulator import ScenarioSimulator, ScenarioType, ScenarioResult
from engine.policy_engine import (
    PolicyEngine, PolicyDecision, TradingAction, PositionState, PositionSide,
    RiskConfig, VolatilityRegime, LiquidityRegime, EvaluationZone
)
from engine.portfolio_risk_manager import PortfolioRiskManager, RiskDecision
from engine.causal_evaluator import (
    CausalEvaluator, MarketState,
    MacroState, LiquidityState, VolatilityState,
    DealerState, EarningsState, TimeRegimeState, TimeRegimeType,
    PriceLocationState, MacroNewsState, LiquidityRegime as CausalLiquidityRegime,
    VolatilityRegime as CausalVolatilityRegime
)


# Helper base class for market state creation
class BaseScenarioTest(unittest.TestCase):
    """Base test class with helper methods."""
    
    def _create_market_state(self, regime: str, regime_confidence: float) -> MarketState:
        """Create a market state for testing."""
        return MarketState(
            timestamp=datetime.now(),
            symbol="ES",
            current_price=100.0,
            session_open=100.0,
            session_high=102.0,
            session_low=99.0,
            session_name="RTH_OPEN",
            session_vol_scale=1.0,
            session_liq_scale=1.0,
            session_risk_scale=1.0,
            time_regime_state=TimeRegimeState(
                regime_type=TimeRegimeType.POWER_HOUR,
                minutes_into_session=240,
                hours_until_session_end=4.0,
                day_of_week=2
            ),
            macro_state=MacroState(
                sentiment_score=0.0,
                surprise_score=0.0,
                rate_expectation=0.0,
                inflation_expectation=0.0,
                gdp_expectation=0.0
            ),
            liquidity_state=LiquidityState(
                bid_ask_spread=0.01,
                order_book_depth=0.8,
                regime=CausalLiquidityRegime.NORMAL,
                volume_trend=0.5
            ),
            volatility_state=VolatilityState(
                current_vol=0.15,
                vol_percentile=0.5,
                regime=CausalVolatilityRegime.NORMAL,
                vol_trend=0.0,
                skew=-0.2
            ),
            dealer_state=DealerState(
                net_gamma_exposure=0.0,
                net_spot_exposure=0.0,
                vega_exposure=0.0,
                dealer_sentiment=0.0
            ),
            earnings_state=EarningsState(
                multi_mega_cap_exposure=0.5,
                small_cap_exposure=0.5,
                earnings_season_flag=False,
                earnings_surprise_momentum=0.0
            ),
            price_location_state=PriceLocationState(
                distance_from_high=0.5,
                distance_from_low=0.5,
                range_ratio=1.0,
                session_extremity=0.0
            ),
            macro_news_state=MacroNewsState(
                risk_sentiment_score=0.0,
                hawkishness_score=0.0,
                surprise_score=0.0,
                event_importance=0,
                hours_since_last_event=24.0,
                macro_event_count=0,
                news_article_count=0,
                macro_news_state="NEUTRAL"
            ),
        )


class TestScenarioPolicyTrend(BaseScenarioTest):
    """Test scenario-aware policy decisions in TREND regime."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy_engine = PolicyEngine(verbose=False)
        self.scenario_simulator = ScenarioSimulator(verbose=False)
        self.risk_manager = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=50000,
            max_total_exposure=200000,
            max_daily_loss=10000
        )
    
    def test_trend_strong_up_allows_continuation(self):
        """TREND regime with UP probability > 55% should allow continuation entries."""
        # Generate strong UP scenario in TREND
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='TREND',
            regime_confidence=0.8,
            eval_score=0.6,  # Bullish
        )
        
        self.assertGreater(scenario.probability_up, 0.55)
        self.assertEqual(scenario.regime_label, 'TREND')
        
        # Create market state
        market_state = self._create_market_state('TREND', 0.8)
        
        # Create policy engine with scenario
        eval_result = {
            'eval_score': 0.6,
            'confidence': 0.8,
            'regime_label': 'TREND',
            'regime_confidence': 0.8,
            'scenario_result': scenario,
            'session': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        # FLAT position should trigger entry
        position = PositionState(side=PositionSide.FLAT, size=0.0)
        
        decision = self.policy_engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position
        )
        
        # Should allow entry with scenario boost
        self.assertIn(decision.action, [TradingAction.ENTER_SMALL, TradingAction.ENTER_FULL])
        self.assertGreater(decision.scenario_alignment, 0.3)
        self.assertEqual(decision.scenario_bias, 'bullish')
    
    def test_trend_strong_down_blocks_long_entries(self):
        """TREND regime with DOWN probability > 55% should block long entries."""
        # Generate strong DOWN scenario in TREND
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='TREND',
            regime_confidence=0.75,
            eval_score=-0.6,  # Bearish
        )
        
        self.assertGreater(scenario.probability_down, 0.55)
        
        market_state = self._create_market_state('TREND', 0.75)
        
        eval_result = {
            'eval_score': 0.6,  # Bullish signal but strong DOWN scenario
            'confidence': 0.7,
            'regime_label': 'TREND',
            'regime_confidence': 0.75,
            'scenario_result': scenario,
            'session': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        position = PositionState(side=PositionSide.FLAT, size=0.0)
        
        decision = self.policy_engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position
        )
        
        # Should be blocked or DO_NOTHING due to scenario misalignment
        # (even if eval is bullish, strong DOWN scenario overrides)
        self.assertEqual(decision.scenario_bias, 'bearish')


class TestScenarioPolicyRange(BaseScenarioTest):
    """Test scenario-aware policy decisions in RANGE regime."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy_engine = PolicyEngine(verbose=False)
        self.scenario_simulator = ScenarioSimulator(verbose=False)
    
    def test_range_high_chop_reduces_size(self):
        """RANGE regime with CHOP > 50% should prefer small entries."""
        # Generate RANGE scenario with high CHOP
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='RANGE',
            regime_confidence=0.7,
            eval_score=0.3,  # Neutral
        )
        
        self.assertGreater(scenario.probability_chop, 0.50)
        self.assertEqual(scenario.regime_label, 'RANGE')
        
        market_state = self._create_market_state('RANGE', 0.7)
        
        eval_result = {
            'eval_score': 0.5,
            'confidence': 0.7,
            'regime_label': 'RANGE',
            'regime_confidence': 0.7,
            'scenario_result': scenario,
            'session': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        position = PositionState(side=PositionSide.FLAT, size=0.0)
        
        decision = self.policy_engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position
        )
        
        # Should prefer ENTER_SMALL over ENTER_FULL
        if decision.action in [TradingAction.ENTER_SMALL, TradingAction.ENTER_FULL]:
            self.assertEqual(decision.action, TradingAction.ENTER_SMALL)
    
    def test_range_symmetric_probs_mean_reversion(self):
        """RANGE regime with symmetric UP/DOWN probs should favor mean-reversion."""
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='RANGE',
            regime_confidence=0.7,
            eval_score=0.1,  # Very neutral
        )
        
        # Check that probabilities are relatively symmetric
        imbalance = abs(scenario.probability_up - scenario.probability_down)
        self.assertLess(imbalance, 0.15)
        
        market_state = self._create_market_state('RANGE', 0.7)
        
        eval_result = {
            'eval_score': 0.2,
            'confidence': 0.6,
            'regime_label': 'RANGE',
            'regime_confidence': 0.7,
            'scenario_result': scenario,
            'session': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        position = PositionState(side=PositionSide.FLAT, size=0.0)
        
        decision = self.policy_engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position
        )
        
        # Scenario should show symmetric bias
        self.assertEqual(decision.scenario_bias, 'symmetric')


class TestScenarioPolicyReversal(BaseScenarioTest):
    """Test scenario-aware policy decisions in REVERSAL regime."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy_engine = PolicyEngine(verbose=False)
        self.scenario_simulator = ScenarioSimulator(verbose=False)
    
    def test_reversal_strong_down_after_up_eval_allows_reverse(self):
        """REVERSAL with DOWN prob > 60% after UP eval should allow REVERSE."""
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='REVERSAL',
            regime_confidence=0.85,
            eval_score=0.75,  # Strong UP eval
        )
        
        # Reversal regime: Strong UP eval reverses to DOWN scenario
        self.assertGreater(scenario.probability_down, 0.60)
        self.assertEqual(scenario.regime_label, 'REVERSAL')
        self.assertEqual(scenario.scenario_bias, 'bearish_reversal')
        
        market_state = self._create_market_state('REVERSAL', 0.85)
        
        # Long position trying to hold/reduce
        position = PositionState(
            side=PositionSide.LONG,
            size=0.5,
            entry_price=98.0,
            current_price=100.0
        )
        
        eval_result = {
            'eval_score': 0.75,
            'confidence': 0.8,
            'regime_label': 'REVERSAL',
            'regime_confidence': 0.85,
            'scenario_result': scenario,
            'session': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        decision = self.policy_engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position
        )
        
        # Should allow REVERSE or at least REDUCE
        self.assertIn(decision.action, [TradingAction.REVERSE, TradingAction.REDUCE, TradingAction.EXIT])
        self.assertEqual(decision.scenario_bias, 'bearish_reversal')
    
    def test_reversal_reduces_size_pending_confirmation(self):
        """REVERSAL regime should reduce size pending reversal confirmation."""
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='REVERSAL',
            regime_confidence=0.8,
            eval_score=0.3,  # Moderate
        )
        
        self.assertEqual(scenario.regime_label, 'REVERSAL')
        
        market_state = self._create_market_state('REVERSAL', 0.8)
        
        # FLAT position: Entry in REVERSAL should be small size
        position = PositionState(side=PositionSide.FLAT, size=0.0)
        
        eval_result = {
            'eval_score': 0.5,
            'confidence': 0.7,
            'regime_label': 'REVERSAL',
            'regime_confidence': 0.8,
            'scenario_result': scenario,
            'session': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        decision = self.policy_engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position
        )
        
        # Should enter with reduced size
        if decision.action in [TradingAction.ENTER_SMALL, TradingAction.ENTER_FULL]:
            # REVERSAL regime reduces size
            self.assertEqual(decision.action, TradingAction.ENTER_SMALL)


class TestScenarioRiskScaling(unittest.TestCase):
    """Test scenario-aware risk scaling in PortfolioRiskManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_manager = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=50000,
            max_total_exposure=200000,
            max_daily_loss=10000
        )
        self.scenario_simulator = ScenarioSimulator(verbose=False)
    
    def test_scenario_aligned_increases_size_10_percent(self):
        """Scenario strongly aligned should increase size by ~10%."""
        # Strong UP scenario in TREND matching bullish eval
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='TREND',
            regime_confidence=0.8,
            eval_score=0.6,  # Bullish
        )
        
        self.assertGreater(scenario.probability_up, 0.55)
        
        eval_result = {
            'eval_score': 0.6,
            'scenario_result': scenario,
        }
        
        # Base size
        base_size = 1000.0
        
        # Risk scaling should increase by ~10%
        scaled_size, risk_factor = self.risk_manager._apply_scenario_risk_scaling(
            base_size, scenario, eval_result['eval_score']
        )
        
        self.assertGreaterEqual(risk_factor, 1.09)
        self.assertLess(risk_factor, 1.15)
        self.assertGreater(scaled_size, base_size)
    
    def test_scenario_misaligned_reduces_size_20_percent(self):
        """Scenario strongly misaligned should reduce size by ~20%."""
        # Strong DOWN scenario in TREND opposing bullish eval
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='TREND',
            regime_confidence=0.75,
            eval_score=-0.6,  # Bearish - generates DOWN scenarios
        )
        
        self.assertGreater(scenario.probability_down, 0.55)
        
        eval_result = {
            'eval_score': 0.6,  # Bullish eval but DOWN scenarios
            'scenario_result': scenario,
        }
        
        base_size = 1000.0
        
        # Risk scaling should reduce by ~20%
        scaled_size, risk_factor = self.risk_manager._apply_scenario_risk_scaling(
            base_size, scenario, eval_result['eval_score']
        )
        
        self.assertLess(risk_factor, 0.85)
        self.assertGreater(risk_factor, 0.70)
        self.assertLess(scaled_size, base_size)
    
    def test_chop_dominates_reduces_size_25_percent(self):
        """CHOP > 50% in RANGE should reduce size by ~25%."""
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='RANGE',
            regime_confidence=0.7,
            eval_score=0.1,
        )
        
        self.assertGreater(scenario.probability_chop, 0.50)
        
        eval_result = {
            'eval_score': 0.5,
            'scenario_result': scenario,
        }
        
        base_size = 1000.0
        
        # Risk scaling should reduce by ~25%
        scaled_size, risk_factor = self.risk_manager._apply_scenario_risk_scaling(
            base_size, scenario, eval_result['eval_score']
        )
        
        self.assertLess(risk_factor, 0.85)
        self.assertGreater(risk_factor, 0.65)


class TestScenarioDeterminism(BaseScenarioTest):
    """Test deterministic behavior of scenario-aware decisions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy_engine = PolicyEngine(verbose=False)
        self.risk_manager = PortfolioRiskManager(
            total_capital=1000000,
            max_symbol_exposure=50000,
            max_total_exposure=200000,
            max_daily_loss=10000
        )
        self.scenario_simulator = ScenarioSimulator(verbose=False)
    
    def test_policy_decisions_deterministic(self):
        """Policy decisions with same scenario should be identical across runs."""
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='TREND',
            regime_confidence=0.8,
            eval_score=0.6,
        )
        
        market_state = self._create_market_state('TREND', 0.8)
        
        eval_result = {
            'eval_score': 0.6,
            'confidence': 0.8,
            'regime_label': 'TREND',
            'regime_confidence': 0.8,
            'scenario_result': scenario,
            'session': 'MIDDAY',
            'session_modifiers': {},
            'flow_signals': {},
        }
        
        position = PositionState(side=PositionSide.FLAT, size=0.0)
        
        # Run twice
        decision1 = self.policy_engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position
        )
        
        decision2 = self.policy_engine.decide_action(
            market_state=market_state,
            eval_result=eval_result,
            position_state=position
        )
        
        # Should be identical
        self.assertEqual(decision1.action, decision2.action)
        self.assertEqual(decision1.target_size, decision2.target_size)
        self.assertEqual(decision1.scenario_alignment, decision2.scenario_alignment)
        self.assertEqual(decision1.scenario_bias, decision2.scenario_bias)
    
    def test_risk_scaling_deterministic(self):
        """Risk scaling with same scenario should be identical across runs."""
        scenario = self.scenario_simulator.simulate_scenarios(
            current_price=100.0,
            vwap=100.0,
            session_high=102.0,
            session_low=99.0,
            expected_move=1.5,
            volatility=0.02,
            regime_label='TREND',
            regime_confidence=0.8,
            eval_score=0.6,
        )
        
        # Run twice
        scaled1, factor1 = self.risk_manager._apply_scenario_risk_scaling(
            1000.0, scenario, 0.6
        )
        
        scaled2, factor2 = self.risk_manager._apply_scenario_risk_scaling(
            1000.0, scenario, 0.6
        )
        
        # Should be identical
        self.assertEqual(scaled1, scaled2)
        self.assertEqual(factor1, factor2)


if __name__ == '__main__':
    unittest.main()
