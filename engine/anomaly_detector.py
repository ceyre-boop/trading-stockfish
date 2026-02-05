from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, Tuple

from .environment_health import EnvironmentHealthSnapshot


@dataclass
class AnomalyDecision:
    triggered: bool
    reason: str
    details: Dict[str, Any]


class AnomalyDetector:
    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules or {}
        self.recent_anomalies: Deque[Tuple[datetime, str]] = deque()

    def record_and_evaluate(
        self, health_snapshot: EnvironmentHealthSnapshot
    ) -> AnomalyDecision:
        now = health_snapshot.timestamp
        window_seconds = float(self.rules.get("window_seconds", 60))
        max_count = int(self.rules.get("max_anomaly_count", 3))

        cutoff = now - timedelta(seconds=window_seconds)
        while self.recent_anomalies and self.recent_anomalies[0][0] < cutoff:
            self.recent_anomalies.popleft()

        for anomaly in health_snapshot.anomalies:
            self.recent_anomalies.append((now, anomaly))

        triggered = len(self.recent_anomalies) >= max_count > 0
        reason = "environment_unstable" if triggered else "stable"
        details = {
            "recent_anomalies": [
                {"timestamp": ts.isoformat(), "type": name}
                for ts, name in self.recent_anomalies
            ],
            "window_seconds": window_seconds,
            "max_anomaly_count": max_count,
            "anomaly_count": len(self.recent_anomalies),
        }

        return AnomalyDecision(triggered=triggered, reason=reason, details=details)
