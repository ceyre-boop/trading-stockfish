from dataclasses import dataclass
from typing import Dict, Optional
from .session_logging import get_capacity_logger


@dataclass
class CapacityConfig:
    notional_caps: Dict[str, float]
    max_pct_1m: float = 0.02
    max_pct_5m: float = 0.05
    min_depth: float = 1000000.0


class PortfolioRiskManager:
    def __init__(self, config: Optional[CapacityConfig] = None):
        default_caps = {"ES": 20_000_000.0, "NQ": 15_000_000.0}
        self.config = config or CapacityConfig(notional_caps=default_caps)
        self.logger = get_capacity_logger()

    def compute_notional_limit(self, symbol: str) -> float:
        cap = self.config.notional_caps.get(symbol, min(self.config.notional_caps.values()))
        self.logger.info(f"Notional limit for {symbol}: {cap}")
        return cap

    def compute_volume_limit(self, symbol: str, volume_1m: float, volume_5m: float) -> float:
        limit_by_1m = volume_1m * self.config.max_pct_1m
        limit_by_5m = (volume_5m / 5.0) * self.config.max_pct_5m
        final = min(limit_by_1m, limit_by_5m)
        self.logger.info(f"Volume limits for {symbol}: 1m_limit={limit_by_1m} 5m_limit={limit_by_5m} chosen={final}")
        return final

    def compute_market_impact(self, symbol: str, size: float, volatility: float, depth: float) -> float:
        depth = max(depth, self.config.min_depth)
        impact = (size / depth) * volatility * 1000.0
        self.logger.info(f"Estimated market impact for {symbol}: size={size} vol={volatility} depth={depth} impact={impact}")
        return impact

    def enforce_capacity_constraints(self, symbol: str, size: float, price: float, volume_1m: float, volume_5m: float, volatility: float, depth: float):
        notional = abs(size) * price
        notional_limit = self.compute_notional_limit(symbol)
        if notional > notional_limit:
            reason = "notional_exceeds_limit"
            allowed = notional_limit / price
            self.logger.warning(f"Capacity reject {symbol}: requested_notional={notional} limit={notional_limit}")
            return {"allowed_size": allowed, "rejected": True, "reason": reason, "notional_limit": notional_limit}

        volume_limit = self.compute_volume_limit(symbol, volume_1m, volume_5m)
        if abs(size) > volume_limit:
            reason = "volume_exceeds_limit"
            allowed = volume_limit if size > 0 else -volume_limit
            self.logger.warning(f"Capacity reject {symbol}: requested_size={size} volume_limit={volume_limit}")
            return {"allowed_size": allowed, "rejected": True, "reason": reason, "volume_limit": volume_limit}

        impact = self.compute_market_impact(symbol, abs(size), volatility, depth)
        return {"allowed_size": size, "rejected": False, "reason": None, "estimated_impact": impact}
