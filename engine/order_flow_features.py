"""
Order Flow Features v4.0-B
Extracts microstructure behavior features from order book and event stream.
"""

from typing import Any, Dict, List, Tuple


class OrderFlowFeatures:
    def get_liquidity_inputs(self) -> dict:
        """
        Returns aggressive buy/sell volume and net aggressive volume for liquidity features.
        """
        buy_vol, sell_vol = self._aggressive_imbalance_volumes()
        return {
            "aggressive_buy": buy_vol,
            "aggressive_sell": sell_vol,
            "net_aggressive": buy_vol - sell_vol,
        }

    def __init__(self, lookback: int = 10, use_microstructure_realism: bool = True):
        self.lookback = lookback
        self.use_microstructure_realism = use_microstructure_realism
        self.prev_snapshots = []  # List of order book snapshots
        self.prev_events = []  # List of recent trade/quote events

    def update(self, event: Dict, order_book_snapshot: Dict):
        """API-compatible update (event first) to match tests."""
        # Accept callers that accidentally swap arguments: if the first arg looks
        # like a snapshot and the second looks like an event, flip them.
        if self._is_snapshot_like(event) and self._is_event_like(order_book_snapshot):
            event, order_book_snapshot = order_book_snapshot, event

        # If we are given a live OrderBookModel and a book event, refresh it so
        # snapshots reflect the event payload before normalization.
        if (
            hasattr(order_book_snapshot, "update_from_event")
            and isinstance(event, dict)
            and event.get("type") == "book"
        ):
            order_book_snapshot.update_from_event(event)

        snap = self._normalize_snapshot(order_book_snapshot)
        self.prev_snapshots.append(snap)
        self.prev_events.append(event or {})
        if len(self.prev_snapshots) > self.lookback:
            self.prev_snapshots.pop(0)
        if len(self.prev_events) > self.lookback:
            self.prev_events.pop(0)

    def update_from_events(
        self,
        trade_events: List[Dict],
        book_events: List[Dict],
        order_book_snapshot: Dict,
    ):
        # Deterministically update buffers with batch events
        snap = self._normalize_snapshot(order_book_snapshot)
        for event in trade_events + book_events:
            self.prev_events.append(event or {})
        self.prev_snapshots.append(snap)
        # Truncate buffers to lookback
        if len(self.prev_snapshots) > self.lookback:
            self.prev_snapshots = self.prev_snapshots[-self.lookback :]
        if len(self.prev_events) > self.lookback:
            self.prev_events = self.prev_events[-self.lookback :]

    def get_features(self) -> Dict[str, Any]:
        return self.compute_features()

    def compute_features(self) -> Dict[str, Any]:
        # Config gating and neutral output
        if not self.use_microstructure_realism or len(self.prev_snapshots) < 1:
            return self._neutral_features()
        buy_vol, sell_vol = self._aggressive_imbalance_volumes()
        total = buy_vol + sell_vol
        if total == 0:
            buy_ratio = sell_ratio = net_ratio = 0.0
        else:
            buy_ratio = buy_vol / total
            sell_ratio = sell_vol / total
            net_ratio = buy_ratio - sell_ratio

        quote_pulling_score = self._quote_pulling()
        sweep_flag = self._sweep_event()
        spoofing_score = self._spoofing_heuristic()
        last_trade_side, last_trade_size = self._last_trade()
        aggressive_trade = bool(sweep_flag or quote_pulling_score > 0.0)

        # Clamp scores to deterministic, non-negative ranges.
        quote_pulling_score = self._clamp(quote_pulling_score, 0.0, 5.0)
        spoofing_score = self._clamp(spoofing_score, 0.0, 5.0)
        return {
            "buy_imbalance": buy_ratio,
            "sell_imbalance": sell_ratio,
            "net_imbalance": net_ratio,
            "quote_pulling_score": quote_pulling_score,
            "sweep_flag": sweep_flag,
            "spoofing_score": spoofing_score,
            "aggressive_side": last_trade_side,
            "trade_size": last_trade_size,
            "bid_pull": self._bid_pull_amount(),
            "aggressive_trade": aggressive_trade,
        }

    def _aggressive_imbalance_volumes(self) -> (float, float):
        buy_vol = 0.0
        sell_vol = 0.0
        for e in self.prev_events:
            if e.get("type") == "trade":
                side = e.get("aggressor") or e.get("side")
                size = float(e.get("size", 1.0) or 0.0)
                if side == "buy":
                    buy_vol += size
                elif side == "sell":
                    sell_vol += size
        return buy_vol, sell_vol

    def _quote_pulling(self) -> float:
        if len(self.prev_snapshots) < 2:
            return 0.0
        prev = self._snapshot_stats(self.prev_snapshots[-2])
        curr = self._snapshot_stats(self.prev_snapshots[-1])

        # Liquidity pulled from top of book
        bid_pull = max(0.0, prev["top_bid_size"] - curr["top_bid_size"])
        ask_pull = max(0.0, prev["top_ask_size"] - curr["top_ask_size"])
        depth_prev = max(prev["bid_depth"] + prev["ask_depth"], 1e-6)
        depth_component = (bid_pull + ask_pull) / depth_prev

        # Spread widening component
        spread_prev = prev["spread"]
        spread_curr = curr["spread"]
        spread_component = 0.0
        if spread_prev is not None and spread_curr is not None:
            spread_component = max(0.0, spread_curr - spread_prev)

        # Imbalance collapsing toward zero signals pullback
        imbalance_component = max(0.0, abs(prev["imbalance"]) - abs(curr["imbalance"]))

        score = depth_component + 0.5 * spread_component + imbalance_component
        return float(max(0.0, score))

    def _sweep_event(self) -> bool:
        # Detect large trades consuming multiple levels
        if len(self.prev_events) < 1 or len(self.prev_snapshots) < 2:
            return False
        last_event = self.prev_events[-1]
        if last_event.get("type") != "trade":
            return False
        prev = self.prev_snapshots[-2]
        curr = self.prev_snapshots[-1]
        prev_bid_count = len(prev["bids"])
        curr_bid_count = len(curr["bids"])
        prev_ask_count = len(prev["asks"])
        curr_ask_count = len(curr["asks"])
        # If a trade event and book levels drop (from 2 to 1, or more), flag as sweep
        if (prev_bid_count > curr_bid_count) or (prev_ask_count > curr_ask_count):
            return True
        return False

    def _spoofing_heuristic(self) -> float:
        score = 0.0
        for i in range(1, len(self.prev_snapshots)):
            prev = self._snapshot_stats(self.prev_snapshots[i - 1])
            curr = self._snapshot_stats(self.prev_snapshots[i])
            for side in ["bid", "ask"]:
                prev_size = prev[f"top_{side}_size"]
                curr_size = curr[f"top_{side}_size"]
                if prev_size > 0 and prev_size > 5.0 * max(curr_size, 1e-6):
                    if prev_size > 50.0:
                        score += 1.0
            # Depth imbalance spike then collapse
            if abs(prev["imbalance"]) > 0.5 and abs(curr["imbalance"]) < 0.2:
                score += 0.5
        return float(max(0.0, score))

    def _last_trade(self):
        for e in reversed(self.prev_events):
            if e.get("type") == "trade":
                side = e.get("aggressor") or e.get("side") or ""
                size = float(e.get("size", 0.0) or 0.0)
                return side, size
        return "", 0.0

    def _bid_pull_amount(self) -> float:
        if len(self.prev_snapshots) < 2:
            return 0.0
        prev = self.prev_snapshots[-2]
        curr = self.prev_snapshots[-1]
        prev_bid = prev["bids"][0][1] if prev["bids"] else 0
        curr_bid = curr["bids"][0][1] if curr["bids"] else 0
        return float(max(0, prev_bid - curr_bid))

    def _neutral_features(self) -> Dict[str, Any]:
        return {
            "buy_imbalance": 0.0,
            "sell_imbalance": 0.0,
            "net_imbalance": 0.0,
            "quote_pulling_score": 0.0,
            "sweep_flag": False,
            "spoofing_score": 0.0,
            "aggressive_side": "",
            "trade_size": 0.0,
            "bid_pull": 0.0,
            "aggressive_trade": False,
        }

    def _normalize_snapshot(self, snapshot: Dict) -> Dict:
        if not snapshot:
            return {"bids": [], "asks": []}

        # Allow OrderBookModel-like objects with bids/asks attributes or accessors.
        if hasattr(snapshot, "get_depth_snapshot"):
            snapshot = snapshot.get_depth_snapshot()
        elif hasattr(snapshot, "bids") or hasattr(snapshot, "asks"):
            snapshot = {
                "bids": getattr(snapshot, "bids", []) or [],
                "asks": getattr(snapshot, "asks", []) or [],
            }

        bids = snapshot.get("bids", []) if isinstance(snapshot, dict) else []
        asks = snapshot.get("asks", []) if isinstance(snapshot, dict) else []

        # Convert [price, vol] pairs to uniform list of tuples
        def _norm(levels):
            out = []
            for lvl in levels:
                if isinstance(lvl, dict):
                    price = float(lvl.get("price", 0.0) or 0.0)
                    volume = float(lvl.get("volume", 0.0) or 0.0)
                    out.append([price, volume])
                elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    price = float(lvl[0] or 0.0)
                    volume = float(lvl[1] or 0.0)
                    out.append([price, volume])
                elif hasattr(lvl, "price") and hasattr(lvl, "volume"):
                    price = float(getattr(lvl, "price", 0.0) or 0.0)
                    volume = float(getattr(lvl, "volume", 0.0) or 0.0)
                    out.append([price, volume])
            return out

        normalized = {"bids": _norm(bids), "asks": _norm(asks)}
        return normalized

    @staticmethod
    def _is_event_like(payload: Any) -> bool:
        return isinstance(payload, dict) and "type" in payload

    @staticmethod
    def _is_snapshot_like(payload: Any) -> bool:
        if payload is None:
            return False
        if isinstance(payload, dict):
            return "bids" in payload or "asks" in payload
        return hasattr(payload, "bids") or hasattr(payload, "asks")

    # --- helpers ---

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return float(min(hi, max(lo, val)))

    def _snapshot_stats(self, snap: Dict) -> Dict[str, float]:
        bids = snap.get("bids", []) if isinstance(snap, dict) else []
        asks = snap.get("asks", []) if isinstance(snap, dict) else []
        top_bid_size = bids[0][1] if bids else 0.0
        top_ask_size = asks[0][1] if asks else 0.0
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        bid_depth = sum(lvl[1] for lvl in bids)
        ask_depth = sum(lvl[1] for lvl in asks)
        if best_bid is not None and best_ask is not None:
            spread = float(best_ask - best_bid)
        else:
            spread = None
        denom = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / denom if denom > 0 else 0.0
        return {
            "top_bid_size": float(top_bid_size),
            "top_ask_size": float(top_ask_size),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_depth": float(bid_depth),
            "ask_depth": float(ask_depth),
            "spread": spread,
            "imbalance": float(imbalance),
        }
