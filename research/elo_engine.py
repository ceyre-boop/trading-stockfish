"""
TradingELO Engine for Trading Stockfish v4.0‑F
Deterministic ELO rating for engine version comparison.
"""

import math
from typing import Dict


class TradingELO:
    def __init__(self, k: float = 32.0):
        self.k = k

    @staticmethod
    def composite_metric(metrics: Dict[str, float]) -> float:
        # 50% risk-adjusted return, 20% drawdown, 20% execution quality, 10% survival
        return (
            0.5 * metrics.get("risk_adjusted_return", 0.0)
            + 0.2 * metrics.get("drawdown", 0.0)
            + 0.2 * metrics.get("execution_quality", 0.0)
            + 0.1 * metrics.get("survival", 0.0)
        )

    def expected_score(self, elo_self: float, elo_opponent: float) -> float:
        return 1.0 / (1.0 + 10 ** ((elo_opponent - elo_self) / 400.0))

    def update_elo(
        self, old_elo: float, elo_opponent: float, metrics: Dict[str, float]
    ) -> float:
        expected = self.expected_score(old_elo, elo_opponent)
        actual = self.composite_metric(metrics)
        return old_elo + self.k * (actual - expected)
