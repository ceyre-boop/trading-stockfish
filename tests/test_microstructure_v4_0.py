"""
Tests for Phase v4.0 Microstructure Realism
"""

import unittest

from engine.liquidity_metrics import compute_liquidity_metrics
from engine.order_book_model import OrderBookModel
from engine.order_flow_features import OrderFlowFeatures


class TestOrderBookModel(unittest.TestCase):
    def setUp(self):
        self.book = OrderBookModel(depth=3)

    def test_update_and_snapshot(self):
        event = {
            "type": "book",
            "bids": [{"price": 99, "volume": 100}, {"price": 98, "volume": 50}],
            "asks": [{"price": 101, "volume": 80}, {"price": 102, "volume": 60}],
        }
        self.book.update_from_event(event)
        snap = self.book.get_depth_snapshot()
        self.assertEqual(len(snap["bids"]), 2)
        self.assertEqual(len(snap["asks"]), 2)
        self.assertEqual(snap["bids"][0]["price"], 99)
        self.assertEqual(snap["asks"][0]["price"], 101)

    def test_spread_and_imbalance(self):
        event = {
            "type": "book",
            "bids": [{"price": 99, "volume": 100}],
            "asks": [{"price": 101, "volume": 80}],
        }
        self.book.update_from_event(event)
        self.assertEqual(self.book.get_spread(), 2)
        imb = self.book.get_imbalance_metrics()
        self.assertAlmostEqual(imb["imbalance"], (100 - 80) / 180)


class TestOrderFlowFeatures(unittest.TestCase):
    def setUp(self):
        self.book = OrderBookModel(depth=2)
        self.features = OrderFlowFeatures()

    def test_aggressive_trade(self):
        trade = {"type": "trade", "side": "buy", "size": 50, "price": 101}
        self.features.update(trade, self.book)
        feats = self.features.get_features()
        self.assertEqual(feats["aggressive_side"], "buy")
        self.assertEqual(feats["trade_size"], 50)

    def test_quote_pulling(self):
        book1 = {
            "type": "book",
            "bids": [{"price": 99, "volume": 200}],
            "asks": [{"price": 101, "volume": 100}],
        }
        book2 = {
            "type": "book",
            "bids": [{"price": 99, "volume": 100}],
            "asks": [{"price": 101, "volume": 100}],
        }
        self.features.update(book1, self.book)
        self.features.update(book2, self.book)
        feats = self.features.get_features()
        self.assertTrue("bid_pull" in feats)
        self.assertEqual(feats["bid_pull"], 100)


class TestLiquidityMetrics(unittest.TestCase):
    def setUp(self):
        self.book = OrderBookModel(depth=3)
        event = {
            "type": "book",
            "bids": [
                {"price": 99, "volume": 100},
                {"price": 98, "volume": 50},
                {"price": 97, "volume": 30},
            ],
            "asks": [
                {"price": 101, "volume": 80},
                {"price": 102, "volume": 60},
                {"price": 103, "volume": 40},
            ],
        }
        self.book.update_from_event(event)

    def test_liquidity_score(self):
        metrics = compute_liquidity_metrics(self.book)
        self.assertEqual(metrics["spread"], 2)
        self.assertEqual(metrics["liquidity_score"], 80)
        self.assertIn("stress_flags", metrics)


if __name__ == "__main__":
    unittest.main()
