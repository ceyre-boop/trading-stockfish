"""
OrderBookModel for microstructure realism (Phase v4.0)
- Multi-level bid/ask depth
- Per-level volume and price
- Imbalance and shape metrics
- Deterministic, replay-compatible
"""

import copy
from typing import Dict, List, Optional


class OrderBookLevel:
    def __init__(self, price: float, volume: float):
        self.price = price
        self.volume = volume

    def to_dict(self):
        return {"price": self.price, "volume": self.volume}


class OrderBookModel:
    def __init__(self, depth: int = 5):
        self.depth = depth
        self.bids: List[OrderBookLevel] = []
        self.asks: List[OrderBookLevel] = []
        self.last_event = None

    def __getitem__(self, key: str):
        if key == "bids":
            return [lvl.to_dict() for lvl in self.bids]
        if key == "asks":
            return [lvl.to_dict() for lvl in self.asks]
        raise KeyError(key)

    def __len__(self):
        return len(self.bids) + len(self.asks)

    def get(self, key: str, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def update_from_event(self, event: Dict):
        # Example event: {'type': 'book', 'bids': [...], 'asks': [...], 'timestamp': ...}
        if event.get("type") == "book":

            def _coerce_levels(levels):
                out = []
                for lvl in levels[: self.depth]:
                    if isinstance(lvl, dict):
                        price = float(lvl.get("price", 0.0) or 0.0)
                        volume = float(lvl.get("volume", 0.0) or 0.0)
                    elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                        price = float(lvl[0] or 0.0)
                        volume = float(lvl[1] or 0.0)
                    else:
                        price = float(getattr(lvl, "price", 0.0) or 0.0)
                        volume = float(getattr(lvl, "volume", 0.0) or 0.0)
                    out.append(OrderBookLevel(price, volume))
                return out

            self.bids = _coerce_levels(event.get("bids", []))
            self.asks = _coerce_levels(event.get("asks", []))
            self.last_event = event

    def get_best_bid_ask(self):
        best_bid = self.bids[0].price if self.bids else None
        best_ask = self.asks[0].price if self.asks else None
        return best_bid, best_ask

    def get_spread(self):
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_depth_snapshot(self):
        return {
            "bids": [lvl.to_dict() for lvl in self.bids],
            "asks": [lvl.to_dict() for lvl in self.asks],
        }

    def get_imbalance_metrics(self):
        bid_vol = sum(lvl.volume for lvl in self.bids)
        ask_vol = sum(lvl.volume for lvl in self.asks)
        imbalance = (
            (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
        )
        return {"bid_volume": bid_vol, "ask_volume": ask_vol, "imbalance": imbalance}

    def get_shape_metrics(self):
        # Simple convexity/concavity: compare outer vs inner levels
        if len(self.bids) < 2 or len(self.asks) < 2:
            return {"bid_shape": 0, "ask_shape": 0}
        bid_shape = self.bids[0].volume - self.bids[-1].volume
        ask_shape = self.asks[0].volume - self.asks[-1].volume
        return {"bid_shape": bid_shape, "ask_shape": ask_shape}

    def clone(self):
        return copy.deepcopy(self)
