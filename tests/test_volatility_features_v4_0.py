import unittest

import numpy as np

from engine.volatility_features import VolatilityFeatures


class TestVolatilityFeaturesV40(unittest.TestCase):
    def setUp(self):
        self.vol = VolatilityFeatures(window=5, use_microstructure_realism=True)

    def test_realized_vol(self):
        prices = [100, 101, 102, 101, 100]
        for p in prices:
            out = self.vol.compute(p)
        self.assertGreaterEqual(out["realized_vol"], 0)

    def test_intraday_band_width(self):
        prices = [100, 101, 102, 101, 100]
        for p in prices:
            out = self.vol.compute(p)
        self.assertGreaterEqual(out["intraday_band_width"], 0)

    def test_vol_of_vol(self):
        prices = [100, 101, 102, 101, 100]
        for p in prices:
            out = self.vol.compute(p)
        self.assertGreaterEqual(out["vol_of_vol"], 0)

    def test_regime_classification(self):
        prices = [100, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9]
        for p in prices:
            out = self.vol.compute(p)
        self.assertIn(out["vol_regime"], ["LOW", "NORMAL", "HIGH", "EXTREME"])

    def test_neutral_outputs(self):
        vol = VolatilityFeatures(window=5, use_microstructure_realism=False)
        out = vol.compute(100)
        self.assertEqual(out["realized_vol"], 0.0)
        self.assertEqual(out["vol_regime"], "NORMAL")

    def test_determinism(self):
        prices = [100, 101, 102, 101, 100]
        for p in prices:
            out1 = self.vol.compute(p)
            out2 = self.vol.compute(p)
            self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
