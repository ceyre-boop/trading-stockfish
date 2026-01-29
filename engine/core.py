from typing import Any, Dict

from .data import MarketState


class TradingStockfish:
    """Simple evaluation stub for live testing harness.

    This stub computes a basic directional bias from the latest bar.
    Replace with the full engine pipeline when available.
    """

    def evaluate(self, state: MarketState) -> Dict[str, Any]:
        delta = state.close - state.open
        direction = "buy" if delta > 0 else "sell" if delta < 0 else "hold"
        strength = abs(delta) / max(abs(state.open), 1e-9)
        return {
            "direction": direction,
            "strength": round(strength, 6),
            "close": state.close,
            "open": state.open,
            "high": state.high,
            "low": state.low,
            "volume": state.volume,
            "timestamp": state.timestamp,
            "symbol": state.symbol,
        }
