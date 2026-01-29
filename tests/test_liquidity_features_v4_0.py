import unittest

from engine.liquidity_features import LiquidityFeatures


class TestLiquidityFeaturesV40(unittest.TestCase):
    def setUp(self):
        self.liq = LiquidityFeatures(window=3, use_microstructure_realism=True)

    def test_top_of_book(self):
        ob = {"bids": [(1.1000, 100.0)], "asks": [(1.1001, 120.0)]}
        out = self.liq.compute(ob)
        self.assertEqual(out["top_depth_bid"], 100.0)
        self.assertEqual(out["top_depth_ask"], 120.0)

    def test_depth_imbalance(self):
        ob = {"bids": [(1.1000, 100.0)], "asks": [(1.1001, 50.0)]}
        out = self.liq.compute(ob)
        self.assertAlmostEqual(out["depth_imbalance"], (100 - 50) / (100 + 50))

    def test_cumulative_depth(self):
        ob = {
            "bids": [(1.1, 10), (1.09, 20), (1.08, 30)],
            "asks": [(1.11, 15), (1.12, 25), (1.13, 35)],
        }
        out = self.liq.compute(ob)
        self.assertEqual(out["cumulative_depth_bid"], 60)
        self.assertEqual(out["cumulative_depth_ask"], 75)

    def test_liquidity_resilience(self):
        ob1 = {
            "bids": [(1.1, 10), (1.09, 20), (1.08, 30)],
            "asks": [(1.11, 15), (1.12, 25), (1.13, 35)],
        }
        ob2 = {
            "bids": [(1.1, 20), (1.09, 30), (1.08, 40)],
            "asks": [(1.11, 25), (1.12, 35), (1.13, 45)],
        }
        self.liq.compute(ob1)
        out = self.liq.compute(ob2)
        self.assertGreater(out["liquidity_resilience"], 0)

    def test_liquidity_pressure(self):
        ob = {
            "bids": [(1.1, 10), (1.09, 20), (1.08, 30)],
            "asks": [(1.11, 15), (1.12, 25), (1.13, 35)],
        }
        order_flow = {"aggressive_buy": 10, "aggressive_sell": 5}
        out = self.liq.compute(ob, order_flow_inputs=order_flow)
        self.assertGreater(out["liquidity_pressure"], 0)

    def test_liquidity_shock(self):
        ob1 = {"bids": [(1.1, 100)], "asks": [(1.11, 100)]}
        ob2 = {"bids": [(1.1, 40)], "asks": [(1.11, 40)]}
        self.liq.compute(ob1)
        out = self.liq.compute(ob2)
        self.assertTrue(out["liquidity_shock"])

    def test_neutral_outputs(self):
        liq = LiquidityFeatures(window=3, use_microstructure_realism=False)
        out = liq.compute({})
        self.assertEqual(out["top_depth_bid"], 0.0)
        self.assertFalse(out["liquidity_shock"])

    def test_determinism(self):
        ob = {
            "bids": [(1.1, 10), (1.09, 20), (1.08, 30)],
            "asks": [(1.11, 15), (1.12, 25), (1.13, 35)],
        }
        out1 = self.liq.compute(ob)
        out2 = self.liq.compute(ob)
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
