import unittest

from engine.order_flow_features import OrderFlowFeatures


class TestOrderFlowFeaturesV40(unittest.TestCase):
    def setUp(self):
        self.order_book_template = lambda bid, ask: {
            "bids": [[100, bid]],
            "asks": [[101, ask]],
        }

    def test_normalized_imbalance_metrics(self):
        off = OrderFlowFeatures()
        # 60 buy, 40 sell
        trades = [
            {"type": "trade", "aggressor": "buy", "size": 60},
            {"type": "trade", "aggressor": "sell", "size": 40},
        ]
        off.update_from_events(trades, [], self.order_book_template(10, 10))
        feats = off.get_features()
        self.assertAlmostEqual(feats["buy_imbalance"], 0.6)
        self.assertAlmostEqual(feats["sell_imbalance"], 0.4)
        self.assertAlmostEqual(feats["net_imbalance"], 0.2)
        # All zeros if no aggressive volume
        off = OrderFlowFeatures()
        off.update_from_events([], [], self.order_book_template(10, 10))
        feats = off.get_features()
        self.assertEqual(feats["buy_imbalance"], 0.0)
        self.assertEqual(feats["sell_imbalance"], 0.0)
        self.assertEqual(feats["net_imbalance"], 0.0)

    def test_batch_update_behavior(self):
        off = OrderFlowFeatures()
        trades = [
            {"type": "trade", "aggressor": "buy", "size": 10},
            {"type": "trade", "aggressor": "sell", "size": 5},
        ]
        book_events = [
            {"type": "add", "side": "bid", "size": 10},
            {"type": "remove", "side": "ask", "size": 5},
        ]
        off.update_from_events(trades, book_events, self.order_book_template(10, 10))
        feats1 = off.get_features()
        # Re-run with same sequence, should be identical
        off2 = OrderFlowFeatures()
        off2.update_from_events(trades, book_events, self.order_book_template(10, 10))
        feats2 = off2.get_features()
        self.assertEqual(feats1, feats2)

    def test_quote_pulling_detection(self):
        off = OrderFlowFeatures()
        # Stable book
        off.update_from_events([], [], self.order_book_template(10, 10))
        off.update_from_events([], [], self.order_book_template(10, 10))
        feats = off.get_features()
        self.assertEqual(feats["quote_pulling_score"], 0.0)
        # Collapse bid size
        off = OrderFlowFeatures()
        off.update_from_events([], [], self.order_book_template(10, 10))
        off.update_from_events([], [], self.order_book_template(0, 10))
        feats = off.get_features()
        self.assertGreater(feats["quote_pulling_score"], 0.0)

    def test_sweep_event_detection(self):
        off = OrderFlowFeatures()
        # Normal trade
        off.update_from_events([], [], self.order_book_template(10, 10))
        off.update_from_events(
            [{"type": "trade", "aggressor": "buy", "size": 1}],
            [],
            self.order_book_template(10, 10),
        )
        feats = off.get_features()
        self.assertFalse(feats["sweep_flag"])
        # Sweep: trade, book levels drop
        off = OrderFlowFeatures()
        ob1 = {"bids": [[100, 10], [99, 10]], "asks": [[101, 10], [102, 10]]}
        ob2 = {"bids": [[100, 10]], "asks": [[101, 10]]}
        off.update_from_events([], [], ob1)
        off.update_from_events(
            [{"type": "trade", "aggressor": "buy", "size": 20}], [], ob2
        )
        feats = off.get_features()
        self.assertTrue(feats["sweep_flag"])

    def test_spoofing_heuristic(self):
        off = OrderFlowFeatures()
        # Normal liquidity
        ob1 = {"bids": [[100, 10]], "asks": [[101, 10]]}
        ob2 = {"bids": [[100, 10]], "asks": [[101, 10]]}
        off.update_from_events([], [], ob1)
        off.update_from_events([], [], ob2)
        feats = off.get_features()
        self.assertEqual(feats["spoofing_score"], 0.0)
        # Large order vanishes
        ob1 = {"bids": [[100, 100]], "asks": [[101, 10]]}
        ob2 = {"bids": [[100, 10]], "asks": [[101, 10]]}
        off = OrderFlowFeatures()
        off.update_from_events([], [], ob1)
        off.update_from_events([], [], ob2)
        feats = off.get_features()
        self.assertGreater(feats["spoofing_score"], 0.0)

    def test_neutral_output_behavior(self):
        # No events
        off = OrderFlowFeatures()
        feats = off.get_features()
        for v in feats.values():
            if isinstance(v, float):
                self.assertEqual(v, 0.0)
            else:
                self.assertFalse(v)
        # use_microstructure_realism = False
        off = OrderFlowFeatures(use_microstructure_realism=False)
        trades = [
            {"type": "trade", "aggressor": "buy", "size": 10},
        ]
        off.update_from_events(trades, [], self.order_book_template(10, 10))
        feats = off.get_features()
        for v in feats.values():
            if isinstance(v, float):
                self.assertEqual(v, 0.0)
            else:
                self.assertFalse(v)
        # Minimal data
        off = OrderFlowFeatures()
        off.update_from_events([], [], self.order_book_template(10, 10))
        feats = off.get_features()
        for v in feats.values():
            if isinstance(v, float):
                self.assertEqual(v, 0.0)
            else:
                self.assertFalse(v)

    def test_determinism(self):
        off1 = OrderFlowFeatures()
        off2 = OrderFlowFeatures()
        events = [
            {"type": "trade", "aggressor": "buy", "size": 10},
            {"type": "trade", "aggressor": "sell", "size": 5},
        ]
        book_events = [
            {"type": "add", "side": "bid", "size": 10},
            {"type": "remove", "side": "ask", "size": 5},
        ]
        ob = self.order_book_template(10, 10)
        for _ in range(3):
            off1.update_from_events(events, book_events, ob)
            off2.update_from_events(events, book_events, ob)
        feats1 = off1.get_features()
        feats2 = off2.get_features()
        self.assertEqual(feats1, feats2)


if __name__ == "__main__":
    unittest.main()
