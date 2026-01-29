"""
tests/test_meta_policy_v3_2.py
Unit tests for Trading Stockfish v3.2 meta-policy integration
"""

import unittest

from meta_policy_engine import MetaPolicyEngine


class TestMetaPolicyEngineV32(unittest.TestCase):
    def setUp(self):
        self.engine = MetaPolicyEngine()
        self.engine.enabled = True
        self.engine.weights = {"w1": 1.0, "w2": 1.0, "w3": 1.0, "w4": 1.0, "w5": 1.0}

    def test_highest_ev_selected(self):
        actions = [
            {
                "EV": 0.9,
                "Risk": 0.2,
                "ScenarioRobustness": 0.5,
                "RegimeFit": 0.5,
                "ExecutionPenalty": 0.1,
            },
            {
                "EV": 0.7,
                "Risk": 0.2,
                "ScenarioRobustness": 0.5,
                "RegimeFit": 0.5,
                "ExecutionPenalty": 0.1,
            },
        ]
        result = self.engine.select_action(actions)
        self.assertEqual(result["chosen_action"], actions[0])

    def test_penalizes_high_risk(self):
        actions = [
            {
                "EV": 0.8,
                "Risk": 0.9,
                "ScenarioRobustness": 0.5,
                "RegimeFit": 0.5,
                "ExecutionPenalty": 0.1,
            },
            {
                "EV": 0.7,
                "Risk": 0.2,
                "ScenarioRobustness": 0.5,
                "RegimeFit": 0.5,
                "ExecutionPenalty": 0.1,
            },
        ]
        result = self.engine.select_action(actions)
        self.assertEqual(result["chosen_action"], actions[1])

    def test_rewards_scenario_robustness(self):
        actions = [
            {
                "EV": 0.7,
                "Risk": 0.2,
                "ScenarioRobustness": 0.9,
                "RegimeFit": 0.5,
                "ExecutionPenalty": 0.1,
            },
            {
                "EV": 0.7,
                "Risk": 0.2,
                "ScenarioRobustness": 0.2,
                "RegimeFit": 0.5,
                "ExecutionPenalty": 0.1,
            },
        ]
        result = self.engine.select_action(actions)
        self.assertEqual(result["chosen_action"], actions[0])

    def test_respects_regime_fit(self):
        actions = [
            {
                "EV": 0.7,
                "Risk": 0.2,
                "ScenarioRobustness": 0.5,
                "RegimeFit": 0.9,
                "ExecutionPenalty": 0.1,
            },
            {
                "EV": 0.7,
                "Risk": 0.2,
                "ScenarioRobustness": 0.5,
                "RegimeFit": 0.1,
                "ExecutionPenalty": 0.1,
            },
        ]
        result = self.engine.select_action(actions)
        self.assertEqual(result["chosen_action"], actions[0])

    def test_deterministic_results(self):
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
        result1 = self.engine.select_action(actions)
        result2 = self.engine.select_action(actions)
        self.assertEqual(result1["chosen_action"], result2["chosen_action"])

    def test_disable_meta_policy_restores_behavior(self):
        self.engine.enabled = False
        actions = [
            {
                "EV": 0.5,
                "Risk": 0.2,
                "ScenarioRobustness": 0.5,
                "RegimeFit": 0.5,
                "ExecutionPenalty": 0.1,
            },
        ]
        result = self.engine.select_action(actions)
        self.assertEqual(result["chosen_action"], actions[0])
        self.assertIn("meta-policy disabled", result["reason_codes"][0])


if __name__ == "__main__":
    unittest.main()
