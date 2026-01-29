import unittest

from engine.liquidity_metrics import (
    compute_liquidity_score,
    compute_spread,
    detect_liquidity_stress,
)


class TestLiquidityMetricsV40(unittest.TestCase):
    def setUp(self):
        self.book_normal = {
            "bids": [[100, 50], [99, 30], [98, 20]],
            "asks": [[101, 50], [102, 30], [103, 20]],
        }
        self.book_wide = {"bids": [[100, 50]], "asks": [[110, 50]]}
        self.book_thin = {"bids": [[100, 2]], "asks": [[101, 2]]}
        self.book_one_sided = {"bids": [[100, 1]], "asks": [[101, 100]]}

    def test_spread(self):
        self.assertEqual(compute_spread(self.book_normal), 1)
        self.assertEqual(compute_spread(self.book_wide), 10)

    def test_liquidity_score(self):
        score = compute_liquidity_score(self.book_normal, top_n=2)
        self.assertGreater(score, 0)
        score_thin = compute_liquidity_score(self.book_thin, top_n=1)
        self.assertLess(score_thin, score)

    def test_stress_flags(self):
        flags = detect_liquidity_stress(self.book_wide)
        self.assertTrue(flags["spread_spike_flag"])
        flags = detect_liquidity_stress(self.book_thin)
        self.assertTrue(flags["thin_book_flag"])
        flags = detect_liquidity_stress(self.book_one_sided)
        self.assertTrue(flags["one_sided_liquidity_flag"])

    def test_determinism(self):
        # Same book → same metrics
        a = compute_liquidity_score(self.book_normal)
        b = compute_liquidity_score(self.book_normal)
        self.assertEqual(a, b)
        f1 = detect_liquidity_stress(self.book_normal)
        f2 = detect_liquidity_stress(self.book_normal)
        self.assertEqual(f1, f2)


if __name__ == "__main__":
    unittest.main()
