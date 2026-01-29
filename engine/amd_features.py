"""
AMD (Accumulation / Manipulation / Distribution) detector - deterministic, causal.
Heuristics intentionally simple and rule-based to satisfy canonical invariants.
"""

from collections import deque
from typing import Dict, List, Optional


class AMDTag:
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    MANIPULATION = "MANIPULATION"
    NEUTRAL = "NEUTRAL"


class AMDFeatures:
    def __init__(self, window: int = 50):
        self.window = window
        self.prices: deque = deque(maxlen=window)
        self.volumes: deque = deque(maxlen=window)

    def compute(
        self,
        price_series: Optional[List[float]] = None,
        volume_series: Optional[List[float]] = None,
        liquidity_state: Optional[Dict] = None,
    ) -> Dict:
        if price_series:
            for p in price_series:
                self.prices.append(p)
        if volume_series:
            for v in volume_series:
                self.volumes.append(v)

        if len(self.prices) < 10:
            return self._neutral()

        prices = list(self.prices)
        volumes = list(self.volumes) if self.volumes else [1.0] * len(self.prices)

        # Basic stats
        last_price = prices[-1]
        first_price = prices[0]
        price_change = (last_price - first_price) / first_price if first_price else 0.0
        price_range = (max(prices) - min(prices)) / first_price if first_price else 0.0
        avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
        recent_volume = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else avg_volume

        # Accumulation / distribution heuristics
        accumulation = (
            price_range < 0.01
            and price_change > 0.002
            and recent_volume > avg_volume * 1.1
        )
        distribution = (
            price_range < 0.01
            and price_change < -0.002
            and recent_volume > avg_volume * 1.1
        )

        # Manipulation heuristic: sharp move with liquidity shock
        liquidity_shock = False
        if liquidity_state:
            liquidity_shock = bool(liquidity_state.get("liquidity_shock", False))
        sharp_move = abs(price_change) > 0.01 or price_range > 0.015
        manipulation = sharp_move and liquidity_shock

        tag = AMDTag.NEUTRAL
        confidence = 0.2
        if manipulation:
            tag = AMDTag.MANIPULATION
            confidence = 0.8
        elif accumulation:
            tag = AMDTag.ACCUMULATION
            confidence = 0.6
        elif distribution:
            tag = AMDTag.DISTRIBUTION
            confidence = 0.6

        return {
            "amd_tag": tag,
            "amd_confidence": confidence,
            "amd_price_change": price_change,
            "amd_price_range": price_range,
        }

    def _neutral(self) -> Dict:
        return {
            "amd_tag": AMDTag.NEUTRAL,
            "amd_confidence": 0.0,
            "amd_price_change": 0.0,
            "amd_price_range": 0.0,
        }
