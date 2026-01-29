import unittest

import engine.test_flags as test_flags
from engine.evaluator import EvaluatorConfig, evaluate
from engine.execution_simulator import ExecutionSimulator


class TestMicrostructureIntegrationV40(unittest.TestCase):
    def setUp(self):
        # Allow legacy evaluator path in this microstructure-only harness
        test_flags.CANONICAL_TEST_BYPASS = True
        self.state_base = {
            "spread": 1.0,
            "liquidity_score": 100.0,
            "liquidity_stress_flags": {
                "spread_spike_flag": False,
                "thin_book_flag": False,
                "one_sided_liquidity_flag": False,
            },
            "order_flow_features": {
                "buy_imbalance": 3,
                "sell_imbalance": 1,
                "net_imbalance": 2,
                "quote_pulling_score": 0,
                "sweep_flag": False,
                "spoofing_score": 0,
            },
            "indicators": {
                "rsi_14": 50,
                "sma_50": 100,
                "sma_200": 100,
                "atr_14": 1,
                "volatility": 0.5,
            },
            "trend": {"regime": "uptrend", "strength": 0.5},
            "sentiment": {"score": 0.5, "confidence": 0.8},
            "candles": {"H1": {}},
            "tick": {"bid": 100, "ask": 101, "spread": 1.0},
            "health": {"is_stale": False, "errors": []},
            "timestamp": 1234567890,
        }
        self.sim = ExecutionSimulator()
        EvaluatorConfig.ENABLE_MICROSTRUCTURE = True

    def test_evaluator_microstructure_on(self):
        state = dict(self.state_base)
        state["spread"] = 3.0
        state["liquidity_score"] = 5.0
        state["liquidity_stress_flags"] = {
            "spread_spike_flag": True,
            "thin_book_flag": True,
            "one_sided_liquidity_flag": False,
        }
        state["order_flow_features"] = {
            "buy_imbalance": 1,
            "sell_imbalance": 2,
            "net_imbalance": -1,
            "quote_pulling_score": 2,
            "sweep_flag": False,
            "spoofing_score": 1,
        }
        result = evaluate(state, enable_microstructure=True)
        self.assertIn("micro_ev_penalty", result["details"])
        self.assertIn("micro_risk_penalty", result["details"])
        self.assertGreaterEqual(result["details"]["micro_ev_penalty"], 0)
        self.assertGreaterEqual(result["details"]["micro_risk_penalty"], 0)

    def test_evaluator_microstructure_off(self):
        state = dict(self.state_base)
        EvaluatorConfig.ENABLE_MICROSTRUCTURE = False
        result = evaluate(state, enable_microstructure=False)
        self.assertNotIn("micro_ev_penalty", result["details"])
        self.assertNotIn("micro_risk_penalty", result["details"])

    def test_execution_simulator_partial_fill(self):
        state = dict(self.state_base)
        state["liquidity_score"] = 2.0
        result = self.sim.simulate_execution_v4(
            "buy", 5.0, 100.0, None, None, "ES", None, state, True
        )
        self.assertLess(result.filled_size, 5.0)
        self.assertTrue(result.liquidity_constrained)

    def test_execution_simulator_full_fill(self):
        state = dict(self.state_base)
        state["liquidity_score"] = 10.0
        result = self.sim.simulate_execution_v4(
            "buy", 5.0, 100.0, None, None, "ES", None, state, True
        )
        self.assertEqual(result.filled_size, 5.0)
        self.assertFalse(result.liquidity_constrained)

    def test_execution_simulator_slippage(self):
        state = dict(self.state_base)
        state["spread"] = 5.0
        state["liquidity_score"] = 2.0
        state["order_flow_features"] = {
            "buy_imbalance": 1,
            "sell_imbalance": 2,
            "net_imbalance": -1,
            "quote_pulling_score": 3,
            "sweep_flag": False,
            "spoofing_score": 2,
        }
        result = self.sim.simulate_execution_v4(
            "buy", 5.0, 100.0, None, None, "ES", None, state, True
        )
        self.assertGreater(result.slippage, 0)

    def test_determinism(self):
        state = dict(self.state_base)
        EvaluatorConfig.ENABLE_MICROSTRUCTURE = True
        r1 = self.sim.simulate_execution_v4(
            "buy", 5.0, 100.0, None, None, "ES", None, state, True
        )
        r2 = self.sim.simulate_execution_v4(
            "buy", 5.0, 100.0, None, None, "ES", None, state, True
        )
        self.assertEqual(r1.filled_size, r2.filled_size)
        self.assertEqual(r1.slippage, r2.slippage)


if __name__ == "__main__":
    unittest.main()
