from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentHealthSnapshot:
    timestamp: datetime
    feed_latency_ms: float
    tick_gap_ms: float
    spread: float
    volume: float
    anomalies: List[str]
    metadata: Dict[str, Any]


class EnvironmentHealthMonitor:
    def __init__(self, thresholds: Dict[str, Any]):
        self.thresholds = thresholds or {}
        self.last_tick_timestamp: Optional[datetime] = None

    @staticmethod
    def _parse_ts(ts: Any) -> Optional[datetime]:
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    def evaluate(
        self, market_state: Dict[str, Any], clock_state: Dict[str, Any]
    ) -> EnvironmentHealthSnapshot:
        ms = market_state or {}
        cs = clock_state or {}

        now = self._parse_ts(cs.get("timestamp_utc")) or datetime.utcnow()
        tick_ts = (
            self._parse_ts(ms.get("timestamp_utc") or cs.get("timestamp_utc")) or now
        )

        feed_latency_ms = max(0.0, (now - tick_ts).total_seconds() * 1000.0)

        last_ts = self.last_tick_timestamp or tick_ts
        tick_gap_ms = max(0.0, (tick_ts - last_ts).total_seconds() * 1000.0)
        self.last_tick_timestamp = tick_ts

        best_ask = ms.get("best_ask")
        best_bid = ms.get("best_bid")
        spread = 0.0
        try:
            if best_ask is not None and best_bid is not None:
                spread = float(best_ask) - float(best_bid)
        except Exception:
            spread = 0.0

        volume = 0.0
        try:
            volume = float(ms.get("volume", 0.0) or 0.0)
        except Exception:
            volume = 0.0

        anomalies: List[str] = []
        if feed_latency_ms > float(
            self.thresholds.get("max_feed_latency_ms", float("inf"))
        ):
            anomalies.append("feed_latency")
        if tick_gap_ms > float(self.thresholds.get("max_tick_gap_ms", float("inf"))):
            anomalies.append("tick_gap")
        if spread > float(self.thresholds.get("max_spread", float("inf"))):
            anomalies.append("spread_wide")
        if volume < float(self.thresholds.get("min_volume", 0.0)):
            anomalies.append("low_volume")

        metadata = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "raw_timestamp": ms.get("timestamp_utc"),
        }

        return EnvironmentHealthSnapshot(
            timestamp=tick_ts,
            feed_latency_ms=float(feed_latency_ms),
            tick_gap_ms=float(tick_gap_ms),
            spread=float(spread),
            volume=float(volume),
            anomalies=anomalies,
            metadata=metadata,
        )
