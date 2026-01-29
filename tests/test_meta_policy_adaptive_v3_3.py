"""
tests/test_meta_policy_adaptive_v3_3.py
Unit tests for Trading Stockfish v3.3 meta-policy adaptive weighting
"""

import unittest

from meta_policy_engine import (
    MetaPolicyEngine,
    MetaPolicyPerformanceTracker,
    MetaPolicyWeights,
)


class TestMetaPolicyAdaptiveV33(unittest.TestCase):
    def setUp(self):
        self.weights = MetaPolicyWeights()
        self.tracker = MetaPolicyPerformanceTracker()
        self.engine = MetaPolicyEngine(
            weights=self.weights,
            enabled=True,
            adaptive_weighting=True,
            performance_tracker=self.tracker,
        )

    def test_weights_update_deterministically(self):
        perf = {
            "pnl": 1.0,
            "risk": 0.5,
            "slippage": 0.1,
            "regime_accuracy": 0.8,
            "scenario_accuracy": 0.7,
        }
        old_weights = self.weights.as_dict()
        self.engine.update_weights(perf)
        new_weights = self.weights.as_dict()
        self.assertNotEqual(old_weights, new_weights)
        # Repeating with same input yields same output
        self.weights = MetaPolicyWeights()  # reset
        self.engine.weights = self.weights
        self.engine.update_weights(perf)
        self.assertEqual(self.weights.as_dict(), new_weights)

    def test_weights_within_bounds(self):
        perf = {
            "pnl": 100.0,
            "risk": 100.0,
            "slippage": 100.0,
            "regime_accuracy": 0.0,
            "scenario_accuracy": 0.0,
        }
        self.engine.update_weights(perf)
        for v in self.weights.as_dict().values():
            self.assertGreaterEqual(v, self.weights.min_weight)
            self.assertLessEqual(v, self.weights.max_weight)

    def test_weights_respond_to_performance(self):
        perf_good = {
            "pnl": 2.0,
            "risk": 0.1,
            "slippage": 0.0,
            "regime_accuracy": 1.0,
            "scenario_accuracy": 1.0,
        }
        perf_bad = {
            "pnl": -2.0,
            "risk": 2.0,
            "slippage": 1.0,
            "regime_accuracy": 0.0,
            "scenario_accuracy": 0.0,
        }
        self.engine.update_weights(perf_good)
        good_weights = self.weights.as_dict()
        self.engine.update_weights(perf_bad)
        bad_weights = self.weights.as_dict()
        self.assertNotEqual(good_weights, bad_weights)

    def test_disable_adaptive_restores_v32(self):
        self.engine.adaptive_weighting = False
        perf = {
            "pnl": 1.0,
            "risk": 0.5,
            "slippage": 0.1,
            "regime_accuracy": 0.8,
            "scenario_accuracy": 0.7,
        }
        old_weights = self.weights.as_dict()
        self.engine.update_weights(perf)
        self.assertEqual(self.weights.as_dict(), old_weights)

    def test_decisions_deterministic(self):
        actions = [
            {
                "EV": 0.5,
                "Risk": 0.2,
                "ScenarioRobustness": 0.5,
                "RegimeFit": 0.5,
                "ExecutionPenalty": 0.1,
            },
            {
                "EV": 0.4,
                "Risk": 0.2,
                "ScenarioRobustness": 0.5,
                "RegimeFit": 0.5,
                "ExecutionPenalty": 0.1,
            },
        ]
        self.tracker.update(
            {
                "pnl": 1.0,
                "risk": 0.5,
                "slippage": 0.1,
                "regime_accuracy": 0.8,
                "scenario_accuracy": 0.7,
            }
        )
        result1 = self.engine.select_action(actions)
        result2 = self.engine.select_action(actions)
        self.assertEqual(result1["chosen_action"], result2["chosen_action"])


if __name__ == "__main__":
    unittest.main()
