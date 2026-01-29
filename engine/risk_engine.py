"""
RiskEngine for Trading Stockfish v4.0‑E
Deterministic risk sizing and institutional constraints.
"""

from typing import Dict


class RiskEngine:
    def __init__(
        self,
        base_risk: float = 1.0,
        liquidity_safety_factor: float = 0.8,
        min_size_threshold: float = 0.01,
        max_exposure_per_instrument: float = 10.0,
        max_daily_loss: float = -0.1,
        max_position_age_min: int = 60,
    ):
        self.base_risk = base_risk
        self.liquidity_safety_factor = liquidity_safety_factor
        self.min_size_threshold = min_size_threshold
        self.max_exposure_per_instrument = max_exposure_per_instrument
        self.max_daily_loss = max_daily_loss
        self.max_position_age_min = max_position_age_min

    def compute_size(
        self, confidence: float, realized_vol: float, cumulative_depth: float
    ) -> float:
        size = self.base_risk * confidence / (1 + realized_vol)
        size = min(size, cumulative_depth * self.liquidity_safety_factor)
        size = max(size, self.min_size_threshold)
        return size

    def check_constraints(
        self, position: Dict, daily_pnl: float, position_age_min: int
    ) -> Dict:
        constraints = {}
        constraints["exposure_limit"] = (
            position.get("size", 0.0) <= self.max_exposure_per_instrument
        )
        constraints["daily_loss_limit"] = daily_pnl >= self.max_daily_loss
        constraints["position_age_limit"] = (
            position_age_min <= self.max_position_age_min
        )
        constraints["force_exit"] = not all(constraints.values())
        return constraints
