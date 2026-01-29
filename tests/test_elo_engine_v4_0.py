"""
Test suite for TradingELO Engine v4.0‑F
"""

import unittest

from research.elo_engine import TradingELO


class TestTradingELOEngineV4(unittest.TestCase):
    def setUp(self):
        self.elo = TradingELO(k=32)

    def test_expected_score(self):
        self.assertAlmostEqual(self.elo.expected_score(1600, 1600), 0.5, places=5)
        self.assertLess(self.elo.expected_score(1400, 1600), 0.5)
        self.assertGreater(self.elo.expected_score(1800, 1600), 0.5)

    def test_composite_metric(self):
        metrics = {
            "risk_adjusted_return": 1.0,
            "drawdown": 0.5,
            "execution_quality": 0.5,
            "survival": 1.0,
        }
        expected = 0.5 * 1.0 + 0.2 * 0.5 + 0.2 * 0.5 + 0.1 * 1.0
        self.assertAlmostEqual(self.elo.composite_metric(metrics), expected, places=5)

    def test_update_elo(self):
        old_elo = 1600
        opp_elo = 1600
        metrics = {
            "risk_adjusted_return": 1.0,
            "drawdown": 1.0,
            "execution_quality": 1.0,
            "survival": 1.0,
        }
        new_elo = self.elo.update_elo(old_elo, opp_elo, metrics)
        self.assertGreater(new_elo, old_elo)
        # Determinism
        new_elo2 = self.elo.update_elo(old_elo, opp_elo, metrics)
        self.assertEqual(new_elo, new_elo2)


if __name__ == "__main__":
    unittest.main()
