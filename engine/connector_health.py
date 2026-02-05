from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEALTH_LOG_DIR = PROJECT_ROOT / "logs" / "health"
HEALTH_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(ts: datetime) -> str:
    try:
        return ts.astimezone(timezone.utc).isoformat()
    except Exception:
        return ts.isoformat()


def _timestamp_for_log(ts: datetime) -> str:
    return ts.strftime("%Y%m%d_%H%M%S_%f")


@dataclass
class HealthEvent:
    connector_name: str
    event_type: str
    details: Dict[str, Any]
    safe_mode_required: bool
    timestamp: datetime

    def to_json(self) -> Dict[str, Any]:
        return {
            "connector_name": self.connector_name,
            "event_type": self.event_type,
            "details": self.details,
            "safe_mode_required": self.safe_mode_required,
            "timestamp": _iso(self.timestamp),
        }


@dataclass
class ConnectorHealthConfig:
    heartbeat_threshold: timedelta = timedelta(seconds=5)
    latency_threshold_ms: float = 1500.0
    failure_threshold: int = 3
    stale_data_threshold: timedelta = timedelta(seconds=5)


class ConnectorHealthMonitor:
    def __init__(self, config: Optional[ConnectorHealthConfig] = None) -> None:
        self.config = config or ConnectorHealthConfig()
        self._last_heartbeat: Dict[str, datetime] = {}
        self._last_latency_ms: Dict[str, float] = {}
        self._send_failures: Dict[str, int] = {}
        self._order_rejections: Dict[str, int] = {}
        self._last_data_timestamp: Dict[str, datetime] = {}

    def record_heartbeat(self, connector_name: str, timestamp: datetime) -> None:
        self._last_heartbeat[connector_name] = timestamp

    def record_latency(self, connector_name: str, latency_ms: float) -> None:
        self._last_latency_ms[connector_name] = float(latency_ms)

    def record_send_failure(self, connector_name: str) -> None:
        self._send_failures[connector_name] = (
            self._send_failures.get(connector_name, 0) + 1
        )

    def record_order_rejection(self, connector_name: str) -> None:
        self._order_rejections[connector_name] = (
            self._order_rejections.get(connector_name, 0) + 1
        )

    def record_market_data_timestamp(
        self, connector_name: str, data_timestamp: datetime
    ) -> None:
        self._last_data_timestamp[connector_name] = data_timestamp

    def evaluate_health(self, connector_name: str, now: datetime) -> List[HealthEvent]:
        events: List[HealthEvent] = []
        cfg = self.config

        last_hb = self._last_heartbeat.get(connector_name)
        heartbeat_age = None
        missed_heartbeats = 0
        if last_hb is not None:
            heartbeat_age = now - last_hb
            if heartbeat_age > cfg.heartbeat_threshold:
                missed_heartbeats = max(int(heartbeat_age / cfg.heartbeat_threshold), 1)
                events.append(
                    HealthEvent(
                        connector_name=connector_name,
                        event_type="heartbeat_timeout",
                        details={
                            "heartbeat_age_sec": heartbeat_age.total_seconds(),
                            "threshold_sec": cfg.heartbeat_threshold.total_seconds(),
                            "missed_heartbeats": missed_heartbeats,
                        },
                        safe_mode_required=True,
                        timestamp=now,
                    )
                )

        last_latency = self._last_latency_ms.get(connector_name)
        if last_latency is not None and last_latency > cfg.latency_threshold_ms:
            events.append(
                HealthEvent(
                    connector_name=connector_name,
                    event_type="latency_degraded",
                    details={
                        "latency_ms": last_latency,
                        "threshold_ms": cfg.latency_threshold_ms,
                    },
                    safe_mode_required=False,
                    timestamp=now,
                )
            )

        send_failures = self._send_failures.get(connector_name, 0)
        order_rejections = self._order_rejections.get(connector_name, 0)
        failure_counts = {
            "send_failures": send_failures,
            "order_rejections": order_rejections,
            "missed_heartbeats": missed_heartbeats,
            "failure_threshold": cfg.failure_threshold,
        }
        if (
            send_failures > cfg.failure_threshold
            or order_rejections > cfg.failure_threshold
            or missed_heartbeats > cfg.failure_threshold
        ):
            events.append(
                HealthEvent(
                    connector_name=connector_name,
                    event_type="failure_threshold_exceeded",
                    details=failure_counts,
                    safe_mode_required=True,
                    timestamp=now,
                )
            )

        last_data_ts = self._last_data_timestamp.get(connector_name)
        if last_data_ts is not None:
            data_age = now - last_data_ts
            if data_age > cfg.stale_data_threshold:
                events.append(
                    HealthEvent(
                        connector_name=connector_name,
                        event_type="stale_data",
                        details={
                            "data_age_sec": data_age.total_seconds(),
                            "threshold_sec": cfg.stale_data_threshold.total_seconds(),
                        },
                        safe_mode_required=True,
                        timestamp=now,
                    )
                )

        return events


def _write_health_log(
    event: HealthEvent, action_taken: str, safe_mode_state: str
) -> Path:
    HEALTH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = HEALTH_LOG_DIR / f"health_{_timestamp_for_log(event.timestamp)}.json"
    payload = event.to_json() | {
        "action_taken": action_taken,
        "safe_mode_state": safe_mode_state,
        "triggering_metric": event.details,
    }
    try:
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to write health log: %s", log_path)
    return log_path


def propagate_health_events(
    events: List[HealthEvent],
    guardrail_callback: Optional[Callable[[HealthEvent], Any]] = None,
    safe_mode_callback: Optional[Callable[[HealthEvent], Any]] = None,
    anomaly_callback: Optional[Callable[[HealthEvent], Any]] = None,
) -> None:
    for event in events:
        action_taken = "logged"
        safe_mode_state = "NOT_REQUIRED" if not event.safe_mode_required else "UNKNOWN"

        if guardrail_callback is not None:
            try:
                guardrail_callback(event)
                action_taken = "guardrail_notified"
            except Exception:
                logger.exception(
                    "Guardrail callback failed for %s", event.connector_name
                )

        if event.safe_mode_required and safe_mode_callback is not None:
            try:
                state = safe_mode_callback(event)
                safe_mode_state = state if isinstance(state, str) else "REQUESTED"
                action_taken = "safe_mode_notified"
            except Exception:
                logger.exception(
                    "SAFE_MODE callback failed for %s", event.connector_name
                )

        if anomaly_callback is not None and event.event_type in {
            "stale_data",
            "latency_degraded",
            "order_adapter_anomaly",
        }:
            try:
                anomaly_callback(event)
            except Exception:
                logger.exception("Anomaly callback failed for %s", event.connector_name)

        _write_health_log(event, action_taken, safe_mode_state)
