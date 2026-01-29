"""
Test suite for ScenarioLibrary v4.0‑F
"""

import os
import unittest

from research.scenario_library import ScenarioLibrary


class TestScenarioLibraryV4(unittest.TestCase):
    def setUp(self):
        # Create a dummy scenario file for testing
        self.test_dir = os.path.join(
            os.path.dirname(__file__), "..", "research", "scenarios"
        )
        os.makedirs(self.test_dir, exist_ok=True)
        self.scenario_name = "test_scenario"
        self.file_path = os.path.join(self.test_dir, f"{self.scenario_name}.json.gz")
        import gzip
        import json

        scenario = {
            "instrument": "ES",
            "date": "2024-01-15",
            "volatility_regime": "HIGH",
            "liquidity_regime": "THIN",
            "macro_regime": "RISK_OFF",
            "event_tags": ["FOMC"],
            "ticks": [{"price": 100.0, "timestamp": 1}],
            "order_books": [{"bids": [], "asks": [], "timestamp": 1}],
            "news_events": [{"headline": "FOMC decision", "timestamp": 1}],
        }
        with gzip.open(self.file_path, "wt", encoding="utf-8") as f:
            json.dump(scenario, f)

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_list_scenarios(self):
        scenarios = ScenarioLibrary.list_scenarios()
        self.assertIn(self.scenario_name, scenarios)

    def test_load_scenario(self):
        data = ScenarioLibrary.load_scenario(self.scenario_name)
        self.assertEqual(data["instrument"], "ES")
        self.assertEqual(data["volatility_regime"], "HIGH")
        self.assertIn("ticks", data)
        self.assertIn("order_books", data)
        self.assertIn("news_events", data)

    def test_metadata(self):
        meta = ScenarioLibrary.scenario_metadata(self.scenario_name)
        self.assertEqual(meta["instrument"], "ES")
        self.assertEqual(meta["macro_regime"], "RISK_OFF")
        self.assertIn("event_tags", meta)

    def test_determinism(self):
        d1 = ScenarioLibrary.load_scenario(self.scenario_name)
        d2 = ScenarioLibrary.load_scenario(self.scenario_name)
        self.assertEqual(d1, d2)

    def test_no_missing_fields(self):
        data = ScenarioLibrary.load_scenario(self.scenario_name)
        required = [
            "instrument",
            "date",
            "volatility_regime",
            "liquidity_regime",
            "macro_regime",
            "event_tags",
            "ticks",
            "order_books",
            "news_events",
        ]
        for field in required:
            self.assertIn(field, data)


if __name__ == "__main__":
    unittest.main()
