"""
ScenarioLibrary for Trading Stockfish v4.0‑F
Deterministic, replay-safe scenario loader for historical market events.
"""

import gzip
import json
import os
from typing import Any, Dict, List


class ScenarioLibrary:
    SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")

    @classmethod
    def list_scenarios(cls) -> List[str]:
        if not os.path.exists(cls.SCENARIO_DIR):
            return []
        return [f[:-8] for f in os.listdir(cls.SCENARIO_DIR) if f.endswith(".json.gz")]

    @classmethod
    def load_scenario(cls, name: str) -> Dict[str, Any]:
        path = os.path.join(cls.SCENARIO_DIR, f"{name}.json.gz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scenario not found: {name}")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        # Validate required fields
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
            if field not in data:
                raise ValueError(f"Missing field in scenario: {field}")
        return data

    @classmethod
    def scenario_metadata(cls, name: str) -> Dict[str, Any]:
        data = cls.load_scenario(name)
        return {
            "instrument": data["instrument"],
            "date": data["date"],
            "volatility_regime": data["volatility_regime"],
            "liquidity_regime": data["liquidity_regime"],
            "macro_regime": data["macro_regime"],
            "event_tags": data["event_tags"],
        }
