import unittest

from engine.order_book_model import OrderBookModel


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

    def test_best_bid_ask_and_spread(self):
        event = {
            "type": "book",
            "bids": [{"price": 99, "volume": 100}],
            "asks": [{"price": 101, "volume": 80}],
        }
        self.book.update_from_event(event)
        best_bid, best_ask = self.book.get_best_bid_ask()
        self.assertEqual(best_bid, 99)
        self.assertEqual(best_ask, 101)
        self.assertEqual(self.book.get_spread(), 2)

    def test_imbalance_metrics(self):
        event = {
            "type": "book",
            "bids": [{"price": 99, "volume": 100}, {"price": 98, "volume": 50}],
            "asks": [{"price": 101, "volume": 80}],
        }
        self.book.update_from_event(event)
        metrics = self.book.get_imbalance_metrics()
        self.assertEqual(metrics["bid_volume"], 150)
        self.assertEqual(metrics["ask_volume"], 80)
        self.assertAlmostEqual(metrics["imbalance"], (150 - 80) / 230)


if __name__ == "__main__":
    unittest.main()
