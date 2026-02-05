from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANOMALY_LOG_DIR = PROJECT_ROOT / "logs" / "anomalies"
ANOMALY_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(ts: datetime) -> str:
    try:
        return ts.astimezone(timezone.utc).isoformat()
    except Exception:
        return ts.isoformat()


def _log_path(ts: datetime) -> Path:
    return ANOMALY_LOG_DIR / f"anomaly_{ts:%Y%m%d_%H%M%S_%f}.json"


@dataclass
class AnomalyEvent:
    anomaly_type: str
    severity: str
    details: Dict[str, Any]
    safe_mode_required: bool
    timestamp: datetime

    @property
    def kind(self) -> str:
        # Legacy alias for older tests/consumers expecting `kind`.
        return self.anomaly_type

    def to_json(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "details": self.details,
            "safe_mode_required": self.safe_mode_required,
            "timestamp": _iso(self.timestamp),
        }


@dataclass
class AnomalyConfig:
    drift_threshold: float = 3.0
    volatility_threshold: float = 0.05
    stale_data_threshold: timedelta = timedelta(seconds=5)
    safe_mode_window: timedelta = timedelta(minutes=5)
    safe_mode_trigger_threshold: int = 3
    feature_history: int = 50


class AnomalyDetector:
    def __init__(
        self,
        config: Optional[AnomalyConfig] = None,
        guardrail_callback: Optional[Callable[[AnomalyEvent], Any]] = None,
        safe_mode_callback: Optional[Callable[[AnomalyEvent], Any]] = None,
        health_callback: Optional[Callable[[AnomalyEvent], Any]] = None,
        *,
        volatility_threshold: Optional[float] = None,
        stale_seconds: Optional[int] = None,
        safety_event_threshold: Optional[int] = None,
    ) -> None:
        base_config = config or AnomalyConfig()
        if volatility_threshold is not None:
            base_config.volatility_threshold = volatility_threshold
        if stale_seconds is not None:
            base_config.stale_data_threshold = timedelta(seconds=stale_seconds)
        if safety_event_threshold is not None:
            base_config.safe_mode_trigger_threshold = safety_event_threshold

        self.config = base_config
        self.guardrail_callback = guardrail_callback
        self.safe_mode_callback = safe_mode_callback
        self.health_callback = health_callback

        self._feature_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.feature_history)
        )
        self._last_market_data_ts: Optional[datetime] = None
        self._last_volatility: Optional[float] = None
        self._last_regime: Optional[str] = None
        self._current_regime: Optional[str] = None
        self._safe_mode_events: Deque[datetime] = deque()
        self._adapter_anomalies: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._storage_failures: Deque[Dict[str, Any]] = deque(maxlen=100)

    def detect(self, metrics: Dict[str, Any]) -> List[AnomalyEvent]:
        """Legacy single-call interface used by older tests."""
        now = _utc_now()
        events: List[AnomalyEvent] = []

        vol = metrics.get("volatility")
        if isinstance(vol, (int, float)) and vol > self.config.volatility_threshold:
            events.append(
                AnomalyEvent(
                    anomaly_type="volatility_spike",
                    severity="high",
                    details={
                        "volatility": vol,
                        "threshold": self.config.volatility_threshold,
                    },
                    safe_mode_required=True,
                    timestamp=now,
                )
            )

        if metrics.get("regime_flip"):
            events.append(
                AnomalyEvent(
                    anomaly_type="regime_flip",
                    severity="medium",
                    details={},
                    safe_mode_required=False,
                    timestamp=now,
                )
            )

        age = metrics.get("data_age_seconds")
        if (
            isinstance(age, (int, float))
            and age > self.config.stale_data_threshold.total_seconds()
        ):
            events.append(
                AnomalyEvent(
                    anomaly_type="stale_data",
                    severity="high",
                    details={
                        "age_sec": age,
                        "threshold_sec": self.config.stale_data_threshold.total_seconds(),
                    },
                    safe_mode_required=True,
                    timestamp=now,
                )
            )

        safety_count = metrics.get("safety_event_count")
        if (
            isinstance(safety_count, (int, float))
            and safety_count > self.config.safe_mode_trigger_threshold
        ):
            events.append(
                AnomalyEvent(
                    anomaly_type="repeated_safety_events",
                    severity="high",
                    details={
                        "count": safety_count,
                        "threshold": self.config.safe_mode_trigger_threshold,
                    },
                    safe_mode_required=True,
                    timestamp=now,
                )
            )

        return events

    def update_features(self, feature_vector: Dict[str, Any]) -> None:
        for name, value in feature_vector.items():
            try:
                val = float(value)
            except Exception:
                continue
            self._feature_history[name].append(val)

    def update_market_data(
        self, timestamp: datetime, price: float, volatility: float
    ) -> None:
        self._last_market_data_ts = timestamp
        self._last_volatility = volatility

    def update_regime(self, regime_label: str) -> None:
        self._current_regime = regime_label

    def record_safe_mode_trigger(self) -> None:
        self._safe_mode_events.append(_utc_now())

    def record_adapter_event(self, event_type: str, details: Dict[str, Any]) -> None:
        if event_type in {"missing_fill", "unexpected_fill", "out_of_order"}:
            self._adapter_anomalies.append(
                {"event_type": event_type, "details": details, "timestamp": _utc_now()}
            )

    def record_storage_failure(self, details: Dict[str, Any]) -> None:
        self._storage_failures.append({"details": details, "timestamp": _utc_now()})

    def _detect_drift(self) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        for name, series in self._feature_history.items():
            if len(series) < 5:
                continue
            mean_val = sum(series) / len(series)
            var = sum((x - mean_val) ** 2 for x in series) / len(series)
            std = var**0.5
            latest = series[-1]
            if std == 0:
                continue
            z = abs((latest - mean_val) / std)
            if z > self.config.drift_threshold:
                events.append(
                    AnomalyEvent(
                        anomaly_type="drift_spike",
                        severity="medium",
                        details={
                            "feature": name,
                            "z_score": z,
                            "mean": mean_val,
                            "std": std,
                            "value": latest,
                        },
                        safe_mode_required=False,
                        timestamp=_utc_now(),
                    )
                )
        return events

    def _detect_volatility(self, now: datetime) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        if (
            self._last_volatility is not None
            and self._last_volatility > self.config.volatility_threshold
        ):
            events.append(
                AnomalyEvent(
                    anomaly_type="volatility_shock",
                    severity="high",
                    details={
                        "volatility": self._last_volatility,
                        "threshold": self.config.volatility_threshold,
                    },
                    safe_mode_required=True,
                    timestamp=now,
                )
            )
        return events

    def _detect_regime(self, now: datetime) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        if self._current_regime is not None and self._last_regime is not None:
            if self._current_regime != self._last_regime:
                events.append(
                    AnomalyEvent(
                        anomaly_type="regime_flip",
                        severity="medium",
                        details={
                            "regime_before": self._last_regime,
                            "regime_after": self._current_regime,
                        },
                        safe_mode_required=False,
                        timestamp=now,
                    )
                )
        if self._current_regime is not None:
            self._last_regime = self._current_regime
        return events

    def _detect_stale_data(self, now: datetime) -> List[AnomalyEvent]:
        if self._last_market_data_ts is None:
            return []
        age = now - self._last_market_data_ts
        if age > self.config.stale_data_threshold:
            return [
                AnomalyEvent(
                    anomaly_type="stale_data",
                    severity="high",
                    details={
                        "age_sec": age.total_seconds(),
                        "threshold_sec": self.config.stale_data_threshold.total_seconds(),
                    },
                    safe_mode_required=True,
                    timestamp=now,
                )
            ]
        return []

    def _detect_repeated_safe_mode(self, now: datetime) -> List[AnomalyEvent]:
        window_start = now - self.config.safe_mode_window
        while self._safe_mode_events and self._safe_mode_events[0] < window_start:
            self._safe_mode_events.popleft()
        if len(self._safe_mode_events) > self.config.safe_mode_trigger_threshold:
            return [
                AnomalyEvent(
                    anomaly_type="repeated_safe_mode",
                    severity="high",
                    details={
                        "count": len(self._safe_mode_events),
                        "window_sec": self.config.safe_mode_window.total_seconds(),
                    },
                    safe_mode_required=True,
                    timestamp=now,
                )
            ]
        return []

    def _drain_adapter_anomalies(self) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        while self._adapter_anomalies:
            anomaly = self._adapter_anomalies.popleft()
            events.append(
                AnomalyEvent(
                    anomaly_type="adapter_anomaly",
                    severity="medium",
                    details=anomaly,
                    safe_mode_required=False,
                    timestamp=_utc_now(),
                )
            )
        return events

    def _drain_storage_failures(self) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        while self._storage_failures:
            failure = self._storage_failures.popleft()
            events.append(
                AnomalyEvent(
                    anomaly_type="storage_failure",
                    severity="high",
                    details=failure,
                    safe_mode_required=True,
                    timestamp=_utc_now(),
                )
            )
        return events

    def evaluate(self, now: datetime) -> List[AnomalyEvent]:
        events: List[AnomalyEvent] = []
        events.extend(self._detect_drift())
        events.extend(self._detect_volatility(now))
        events.extend(self._detect_regime(now))
        events.extend(self._detect_stale_data(now))
        events.extend(self._detect_repeated_safe_mode(now))
        events.extend(self._drain_adapter_anomalies())
        events.extend(self._drain_storage_failures())

        if not events:
            return []

        log_anomalies(events)

        for event in events:
            if self.guardrail_callback:
                try:
                    self.guardrail_callback(event)
                except Exception:
                    logger.exception("Guardrail callback failed")

            if event.safe_mode_required and self.safe_mode_callback:
                try:
                    self.safe_mode_callback(event)
                except Exception:
                    logger.exception("SAFE_MODE callback failed")

            if self.health_callback:
                try:
                    self.health_callback(event)
                except Exception:
                    logger.exception("Health callback failed")

        return events


def log_anomalies(events: List[AnomalyEvent]) -> None:
    ANOMALY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for event in events:
        path = _log_path(event.timestamp)
        payload = event.to_json() | {
            "triggering_metric": event.details,
            "action_taken": "logged",
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Failed to write anomaly log: %s", path)
