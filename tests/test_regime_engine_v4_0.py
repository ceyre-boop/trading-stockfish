import unittest

from engine.regime_engine import RegimeEngine


class TestRegimeEngineV40(unittest.TestCase):
    def setUp(self):
        self.regime = RegimeEngine(window=3)

    def test_vol_regime(self):
        vstate = {"vol_regime": "HIGH", "realized_vol": 0.2}
        lstate = {"liquidity_resilience": 0.1, "depth_imbalance": 0.1}
        mstate = {"hawkishness": 0.0, "risk_sentiment": 0.6}
        out = self.regime.compute(vstate, lstate, mstate)
        self.assertEqual(out["vol_regime"], "HIGH")

    def test_liq_regime(self):
        vstate = {"vol_regime": "NORMAL", "realized_vol": 0.1}
        lstate = {"liquidity_resilience": 0.3, "depth_imbalance": 0.05}
        mstate = {"hawkishness": 0.0, "risk_sentiment": 0.0}
        out = self.regime.compute(vstate, lstate, mstate)
        self.assertEqual(out["liq_regime"], "DEEP")

    def test_macro_regime(self):
        vstate = {"vol_regime": "NORMAL", "realized_vol": 0.1}
        lstate = {"liquidity_resilience": 0.1, "depth_imbalance": 0.1}
        mstate = {"hawkishness": 0.0, "risk_sentiment": -0.6}
        out = self.regime.compute(vstate, lstate, mstate)
        self.assertEqual(out["macro_regime"], "RISK_OFF")

    def test_regime_transition(self):
        vstate = {"vol_regime": "LOW", "realized_vol": 0.05}
        lstate = {"liquidity_resilience": 0.1, "depth_imbalance": 0.1}
        mstate = {"hawkishness": 0.0, "risk_sentiment": 0.6}
        self.regime.compute(vstate, lstate, mstate)
        vstate2 = {"vol_regime": "HIGH", "realized_vol": 0.2}
        out = self.regime.compute(vstate2, lstate, mstate)
        self.assertTrue(out["regime_transition"])

    def test_regime_confidence(self):
        vstate = {"vol_regime": "NORMAL", "realized_vol": 0.2}
        lstate = {"liquidity_resilience": 0.2, "depth_imbalance": 0.1}
        mstate = {"hawkishness": 0.0, "risk_sentiment": 0.0}
        out = self.regime.compute(vstate, lstate, mstate)
        self.assertGreaterEqual(out["regime_confidence"], 0)
        self.assertLessEqual(out["regime_confidence"], 1)

    def test_determinism(self):
        vstate = {"vol_regime": "NORMAL", "realized_vol": 0.1}
        lstate = {"liquidity_resilience": 0.1, "depth_imbalance": 0.1}
        mstate = {"hawkishness": 0.0, "risk_sentiment": 0.0}
        out1 = self.regime.compute(vstate, lstate, mstate)
        out2 = self.regime.compute(vstate, lstate, mstate)
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
