"""
Test suite for GovernanceEngine v4.0‑E
Validates all meta-governance rules and deterministic logic.
"""

import unittest

from engine.governance_engine import GovernanceDecision, GovernanceEngine


class TestGovernanceEngineV4E(unittest.TestCase):
    def setUp(self):
        self.engine = GovernanceEngine(
            max_drawdown_threshold=-0.05,
            event_safety_factor=0.5,
            transition_threshold=0.2,
            max_trades_per_hour=3,
        )
        self.market_state = {
            "volatility_state": {"vol_regime": "NORMAL"},
            "liquidity_state": {"liquidity_shock": False},
            "regime_state": {"macro_regime": "RISK_ON", "regime_transition": False},
            "timestamp": 10000,
        }
        self.eval_output = {"eval_score": 0.5}
        self.policy_decision = {"action": "ENTER_FULL", "size": 1.0}
        self.execution = {"unrealized_pnl": 0.0}

    def test_drawdown_veto(self):
        self.execution["unrealized_pnl"] = -0.10
        decision = self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "DRAWDOWN_LIMIT")

    def test_extreme_volatility_override(self):
        self.market_state["volatility_state"]["vol_regime"] = "EXTREME"
        decision = self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "EXTREME_VOLATILITY")
        self.assertEqual(decision.adjusted_action, "FLAT")
        self.assertEqual(decision.adjusted_size, 0.0)

    def test_liquidity_shock_override(self):
        self.market_state["liquidity_state"]["liquidity_shock"] = True
        decision = self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "LIQUIDITY_SHOCK")
        self.assertEqual(decision.adjusted_action, "EXIT")
        self.assertEqual(decision.adjusted_size, 0.0)

    def test_macro_event_size_reduction(self):
        self.market_state["regime_state"]["macro_regime"] = "EVENT"
        decision = self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )
        self.assertTrue(decision.approved)
        self.assertEqual(decision.reason, "MACRO_EVENT_SIZE_REDUCED")
        self.assertEqual(decision.adjusted_size, 0.5)

    def test_regime_transition_veto(self):
        self.market_state["regime_state"]["regime_transition"] = True
        self.eval_output["eval_score"] = 0.1
        decision = self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "REGIME_TRANSITION_UNSAFE")

    def test_trade_frequency_limit(self):
        # Simulate 4 trades in last hour
        self.engine.trade_times.extend([9900, 9950, 9960, 9990])
        decision = self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "TRADE_FREQUENCY_LIMIT")

    def test_counter_trend_veto(self):
        self.market_state["swing_structure"] = "LL"
        self.market_state["trend_direction"] = "DOWN"
        self.market_state["trend_strength"] = 0.6
        self.eval_output["eval_score"] = 0.5  # Long bias against downtrend
        self.policy_decision = {"action": "ENTER_FULL", "size": 1.0}

        decision = self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "COUNTER_TREND_VETO")

    def test_determinism(self):
        # Run twice, should get same result
        d1 = self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )
        d2 = self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )
        self.assertEqual(d1.to_dict(), d2.to_dict())

    def test_no_unrelated_field_modification(self):
        # Decision should not modify unrelated fields
        before = dict(self.market_state)
        self.engine.decide(
            self.market_state, self.eval_output, self.policy_decision, self.execution
        )
        self.assertEqual(before, self.market_state)


if __name__ == "__main__":
    unittest.main()
