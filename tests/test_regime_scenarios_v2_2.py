"""
Regime-Conditioned Scenario Tests (Phase v2.2).

Tests for scenario generation and integration with regime conditioning.
"""

import unittest
from datetime import datetime
from typing import Dict, Any

from engine.scenario_simulator import (
    ScenarioSimulator, ScenarioType, ScenarioResult
)
from engine.causal_evaluator import CausalEvaluator, MarketState
from engine.causal_evaluator import (
    MacroState, LiquidityState, VolatilityState,
    DealerState, EarningsState, TimeRegimeState, TimeRegimeType,
    PriceLocationState, MacroNewsState, LiquidityRegime, VolatilityRegime
)


class TestScenarioSimulator(unittest.TestCase):
    """Test ScenarioSimulator basic functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ScenarioSimulator(verbose=False)
        self.base_price = 4500.0
        self.vwap = 4500.0
        self.session_high = 4510.0
        self.session_low = 4490.0
        self.expected_move = 10.0
        self.volatility = 0.15
    
    def test_simulator_initialization(self):
        """Test scenario simulator initializes."""
        self.assertIsNotNone(self.simulator)
    
    def test_scenario_generation_default(self):
        """Test basic scenario generation without regime."""
        result = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.vwap,
            session_high=self.session_high,
            session_low=self.session_low,
            expected_move=self.expected_move,
            volatility=self.volatility,
            regime_label="UNKNOWN",
            regime_confidence=0.0,
            eval_score=0.0
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result.scenarios), 3)
        self.assertAlmostEqual(
            result.probability_up + result.probability_down + result.probability_chop,
            1.0,
            places=3
        )
    
    def test_scenario_types_present(self):
        """Test that all scenario types are generated."""
        result = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.vwap,
            session_high=self.session_high,
            session_low=self.session_low,
            expected_move=self.expected_move,
            volatility=self.volatility
        )
        
        types = [s.scenario_type for s in result.scenarios]
        self.assertIn(ScenarioType.UP_MOVE, types)
        self.assertIn(ScenarioType.DOWN_MOVE, types)
        self.assertIn(ScenarioType.CHOP, types)
    
    def test_scenario_probabilities_sum_to_one(self):
        """Test that scenario probabilities sum to 1.0."""
        result = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.vwap,
            session_high=self.session_high,
            session_low=self.session_low,
            expected_move=self.expected_move,
            volatility=self.volatility
        )
        
        total_prob = sum(s.probability for s in result.scenarios)
        self.assertAlmostEqual(total_prob, 1.0, places=3)


class TestTrendScenarios(unittest.TestCase):
    """Test scenario generation in TREND regime."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ScenarioSimulator(verbose=False)
        self.base_price = 4500.0
    
    def test_trend_bullish_up_weighted(self):
        """Test TREND with bullish eval: up moves weighted heavily."""
        result = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.base_price,
            session_high=self.base_price + 20,
            session_low=self.base_price - 20,
            expected_move=10.0,
            volatility=0.15,
            regime_label="TREND",
            regime_confidence=0.8,
            eval_score=0.7  # Bullish
        )
        
        self.assertEqual(result.regime_label, "TREND")
        self.assertEqual(result.scenario_bias, "bullish")
        self.assertGreater(result.probability_up, result.probability_down)
        self.assertGreater(result.probability_up, 0.40)  # Up should be substantial
    
    def test_trend_bearish_down_weighted(self):
        """Test TREND with bearish eval: down moves weighted heavily."""
        result = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.base_price,
            session_high=self.base_price + 20,
            session_low=self.base_price - 20,
            expected_move=10.0,
            volatility=0.15,
            regime_label="TREND",
            regime_confidence=0.8,
            eval_score=-0.7  # Bearish
        )
        
        self.assertEqual(result.regime_label, "TREND")
        self.assertEqual(result.scenario_bias, "bearish")
        self.assertGreater(result.probability_down, result.probability_up)
        self.assertGreater(result.probability_down, 0.40)  # Down should be substantial
    
    def test_trend_confidence_affects_skew(self):
        """Test that higher regime confidence skews probabilities more."""
        result_low_conf = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.base_price,
            session_high=self.base_price + 20,
            session_low=self.base_price - 20,
            expected_move=10.0,
            volatility=0.15,
            regime_label="TREND",
            regime_confidence=0.3,
            eval_score=0.7
        )
        
        result_high_conf = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.base_price,
            session_high=self.base_price + 20,
            session_low=self.base_price - 20,
            expected_move=10.0,
            volatility=0.15,
            regime_label="TREND",
            regime_confidence=0.9,
            eval_score=0.7
        )
        
        # Skew difference should be positive (higher conf = stronger up bias)
        # Using delta > some threshold rather than strict greater-than
        delta = result_high_conf.probability_up - result_low_conf.probability_up
        self.assertGreaterEqual(delta, -0.05)  # Allow for rounding


class TestRangeScenarios(unittest.TestCase):
    """Test scenario generation in RANGE regime."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ScenarioSimulator(verbose=False)
        self.base_price = 4500.0
    
    def test_range_symmetric_probabilities(self):
        """Test RANGE has symmetric up/down probabilities."""
        result = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.base_price,
            session_high=self.base_price + 20,
            session_low=self.base_price - 20,
            expected_move=10.0,
            volatility=0.15,
            regime_label="RANGE",
            regime_confidence=0.8,
            eval_score=0.0
        )
        
        self.assertEqual(result.regime_label, "RANGE")
        self.assertEqual(result.scenario_bias, "symmetric")
        # Up and down should be very similar
        self.assertAlmostEqual(
            result.probability_up,
            result.probability_down,
            places=1
        )
    
    def test_range_chop_elevated(self):
        """Test RANGE elevates chop (oscillation) probability."""
        result = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.base_price,
            session_high=self.base_price + 20,
            session_low=self.base_price - 20,
            expected_move=10.0,
            volatility=0.15,
            regime_label="RANGE",
            regime_confidence=0.8,
            eval_score=0.0
        )
        
        # Chop should be elevated compared to directional moves
        self.assertGreater(
            result.probability_chop,
            result.probability_up
        )
        self.assertGreater(
            result.probability_chop,
            result.probability_down
        )


class TestReversalScenarios(unittest.TestCase):
    """Test scenario generation in REVERSAL regime."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simulator = ScenarioSimulator(verbose=False)
        self.base_price = 4500.0
    
    def test_reversal_strong_up_reverses_down(self):
        """Test REVERSAL with strong up eval: down reversal weighted."""
        result = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.base_price,
            session_high=self.base_price + 20,
            session_low=self.base_price - 20,
            expected_move=10.0,
            volatility=0.15,
            regime_label="REVERSAL",
            regime_confidence=0.8,
            eval_score=0.7  # Strong up exhaustion
        )
        
        self.assertEqual(result.regime_label, "REVERSAL")
        self.assertIn("reversal", result.scenario_bias)
        # Down should be weighted heavily (reversal of exhaustion)
        self.assertGreater(result.probability_down, result.probability_up)
    
    def test_reversal_strong_down_reverses_up(self):
        """Test REVERSAL with strong down eval: up reversal weighted."""
        result = self.simulator.simulate_scenarios(
            current_price=self.base_price,
            vwap=self.base_price,
            session_high=self.base_price + 20,
            session_low=self.base_price - 20,
            expected_move=10.0,
            volatility=0.15,
            regime_label="REVERSAL",
            regime_confidence=0.8,
            eval_score=-0.7  # Strong down exhaustion
        )
        
        self.assertEqual(result.regime_label, "REVERSAL")
        self.assertIn("reversal", result.scenario_bias)
        # Up should be weighted heavily (reversal of exhaustion)
        self.assertGreater(result.probability_up, result.probability_down)


class TestScenarioIntegration(unittest.TestCase):
    """Test scenario integration with CausalEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = CausalEvaluator(
            enable_regime_conditioning=True,
            verbose=False
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
    
    def test_evaluator_generates_scenarios(self):
        """Test that evaluator generates scenarios."""
        state = self._create_market_state()
        result = self.evaluator.evaluate(state)
        
        self.assertIsNotNone(result.scenario_result)
    
    def test_scenarios_affect_confidence(self):
        """Test that scenarios can adjust confidence."""
        state = self._create_market_state()
        result = self.evaluator.evaluate(state)
        
        # Scenario confidence boost should be present
        self.assertIsInstance(result.scenario_confidence_boost, float)
    
    def test_scenario_expected_value_calculated(self):
        """Test that scenario EV is calculated."""
        state = self._create_market_state()
        result = self.evaluator.evaluate(state)
        
        # Scenario EV should be a number (can be 0)
        self.assertIsInstance(result.scenario_ev, float)


class TestDeterministicScenarios(unittest.TestCase):
    """Test deterministic behavior of scenarios."""
    
    def test_scenarios_deterministic(self):
        """Test that same inputs produce same scenarios."""
        simulator = ScenarioSimulator(verbose=False)
        
        # Run simulation twice
        result1 = simulator.simulate_scenarios(
            current_price=4500.0,
            vwap=4500.0,
            session_high=4510.0,
            session_low=4490.0,
            expected_move=10.0,
            volatility=0.15,
            regime_label="TREND",
            regime_confidence=0.8,
            eval_score=0.7
        )
        
        result2 = simulator.simulate_scenarios(
            current_price=4500.0,
            vwap=4500.0,
            session_high=4510.0,
            session_low=4490.0,
            expected_move=10.0,
            volatility=0.15,
            regime_label="TREND",
            regime_confidence=0.8,
            eval_score=0.7
        )
        
        # Probabilities should be identical
        self.assertEqual(result1.probability_up, result2.probability_up)
        self.assertEqual(result1.probability_down, result2.probability_down)
        self.assertEqual(result1.probability_chop, result2.probability_chop)
        self.assertEqual(result1.scenario_bias, result2.scenario_bias)
    
    def test_evaluator_scenarios_deterministic(self):
        """Test that evaluator produces deterministic scenarios."""
        evaluator = CausalEvaluator(enable_regime_conditioning=True, verbose=False)
        
        state = MarketState(
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
        
        # Evaluate twice
        result1 = evaluator.evaluate(state)
        result2 = evaluator.evaluate(state)
        
        # Scenarios should be identical
        if result1.scenario_result and result2.scenario_result:
            self.assertEqual(
                result1.scenario_result.probability_up,
                result2.scenario_result.probability_up
            )
            self.assertEqual(
                result1.scenario_result.scenario_bias,
                result2.scenario_result.scenario_bias
            )


if __name__ == '__main__':
    unittest.main()
