"""
Deterministic liquidity features for Trading Stockfish v4.0‑C
"""

from collections import deque
from typing import Dict, List, Optional


class LiquidityFeatures:
    def __init__(self, window: int = 5, use_microstructure_realism: bool = True):
        self.window = window
        self.use_microstructure_realism = use_microstructure_realism
        self.prev_snapshots = deque(maxlen=window)
        self.prev_depth = deque(maxlen=window)

    def compute(
        self,
        order_book_snapshot: Dict,
        trade_events: Optional[List[Dict]] = None,
        order_flow_inputs: Optional[Dict] = None,
    ) -> Dict:
        if not self.use_microstructure_realism or not order_book_snapshot:
            return self._neutral()
        # Extract top-of-book volumes; support dict or tuple levels
        bids = order_book_snapshot.get("bids", [])
        asks = order_book_snapshot.get("asks", [])

        def _vol(level):
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                return float(level[1] or 0.0)
            if isinstance(level, dict):
                return float(level.get("volume", 0.0) or 0.0)
            return float(getattr(level, "volume", 0.0) or 0.0)

        top_bid = _vol(bids[0]) if bids else 0.0
        top_ask = _vol(asks[0]) if asks else 0.0
        # Depth imbalance
        depth_imbalance = 0.0
        if top_bid + top_ask > 0:
            depth_imbalance = (top_bid - top_ask) / (top_bid + top_ask)
        # Cumulative depth
        cum_bid = sum(_vol(b) for b in bids[: self.window])
        cum_ask = sum(_vol(a) for a in asks[: self.window])
        cum_imbalance = 0.0
        if cum_bid + cum_ask > 0:
            cum_imbalance = (cum_bid - cum_ask) / (cum_bid + cum_ask)
        # Resilience
        prev_cum_bid = self.prev_depth[-1][0] if self.prev_depth else cum_bid
        prev_cum_ask = self.prev_depth[-1][1] if self.prev_depth else cum_ask
        resilience_bid = (
            (cum_bid - prev_cum_bid) / prev_cum_bid if prev_cum_bid else 0.0
        )
        resilience_ask = (
            (cum_ask - prev_cum_ask) / prev_cum_ask if prev_cum_ask else 0.0
        )
        liquidity_resilience = (resilience_bid + resilience_ask) / 2
        # Pressure
        liquidity_pressure = 0.0
        if order_flow_inputs:
            aggressive_vol = order_flow_inputs.get(
                "aggressive_buy", 0.0
            ) + order_flow_inputs.get("aggressive_sell", 0.0)
            total_cum = cum_bid + cum_ask
            if total_cum > 0:
                liquidity_pressure = aggressive_vol / total_cum
        # Shock
        liquidity_shock = False
        if len(self.prev_depth) > 0:
            drop_bid = (prev_cum_bid - cum_bid) / prev_cum_bid if prev_cum_bid else 0.0
            drop_ask = (prev_cum_ask - cum_ask) / prev_cum_ask if prev_cum_ask else 0.0
            if drop_bid > 0.5 or drop_ask > 0.5:
                liquidity_shock = True
        # Update buffers
        self.prev_snapshots.append(order_book_snapshot)
        self.prev_depth.append((cum_bid, cum_ask))
        return {
            "top_depth_bid": top_bid,
            "top_depth_ask": top_ask,
            "depth_imbalance": depth_imbalance,
            "cumulative_depth_bid": cum_bid,
            "cumulative_depth_ask": cum_ask,
            "cumulative_depth_imbalance": cum_imbalance,
            "liquidity_resilience": liquidity_resilience,
            "liquidity_pressure": liquidity_pressure,
            "liquidity_shock": liquidity_shock,
        }

    def _neutral(self) -> Dict:
        return {
            "top_depth_bid": 0.0,
            "top_depth_ask": 0.0,
            "depth_imbalance": 0.0,
            "cumulative_depth_bid": 0.0,
            "cumulative_depth_ask": 0.0,
            "cumulative_depth_imbalance": 0.0,
            "liquidity_resilience": 0.0,
            "liquidity_pressure": 0.0,
            "liquidity_shock": False,
        }
