from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from engine.connector_health import (
    ConnectorHealthConfig,
    ConnectorHealthMonitor,
    HealthEvent,
    propagate_health_events,
)

BASE_TS = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def ts(seconds: int) -> datetime:
    return BASE_TS + timedelta(seconds=seconds)


@pytest.fixture
def config() -> ConnectorHealthConfig:
    return ConnectorHealthConfig(
        heartbeat_threshold=timedelta(seconds=5),
        latency_threshold_ms=100.0,
        failure_threshold=2,
        stale_data_threshold=timedelta(seconds=5),
    )


@pytest.fixture
def monitor(config: ConnectorHealthConfig) -> ConnectorHealthMonitor:
    return ConnectorHealthMonitor(config)


def test_heartbeat_timeout_event(monitor: ConnectorHealthMonitor) -> None:
    connector = "C1"
    monitor.record_heartbeat(connector, ts(0))

    events = monitor.evaluate_health(connector, ts(10))

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "heartbeat_timeout"
    assert event.safe_mode_required is True


def test_latency_degraded_event(monitor: ConnectorHealthMonitor) -> None:
    connector = "C2"
    monitor.record_latency(connector, 200.0)

    events = monitor.evaluate_health(connector, ts(0))

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "latency_degraded"
    assert event.safe_mode_required is False


def test_failure_threshold_exceeded(monitor: ConnectorHealthMonitor) -> None:
    connector = "C3"
    monitor.record_send_failure(connector)
    monitor.record_send_failure(connector)
    monitor.record_send_failure(connector)

    events = monitor.evaluate_health(connector, ts(0))

    assert any(evt.event_type == "failure_threshold_exceeded" for evt in events)
    assert any(evt.safe_mode_required for evt in events)


def test_stale_data_event(monitor: ConnectorHealthMonitor) -> None:
    connector = "C4"
    monitor.record_market_data_timestamp(connector, ts(0))

    events = monitor.evaluate_health(connector, ts(10))

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "stale_data"
    assert event.safe_mode_required is True


def test_multiple_events_emitted(monitor: ConnectorHealthMonitor) -> None:
    connector = "C5"
    monitor.record_latency(connector, 250.0)
    monitor.record_market_data_timestamp(connector, ts(-10))

    events = monitor.evaluate_health(connector, ts(0))

    event_types = {evt.event_type for evt in events}
    assert {"latency_degraded", "stale_data"}.issubset(event_types)
    assert len(events) >= 2


def test_no_events_when_healthy(monitor: ConnectorHealthMonitor) -> None:
    connector = "C6"
    monitor.record_heartbeat(connector, ts(0))
    monitor.record_latency(connector, 50.0)
    monitor.record_market_data_timestamp(connector, ts(0))

    events = monitor.evaluate_health(connector, ts(2))

    assert events == []


def test_propagation_callbacks_invoked(
    monkeypatch: pytest.MonkeyPatch, tmp_path, monitor: ConnectorHealthMonitor
) -> None:
    connector = "C7"
    monitor.record_market_data_timestamp(connector, ts(-10))
    monitor.record_latency(connector, 250.0)
    events = monitor.evaluate_health(connector, ts(0))

    guardrail_cb = Mock()
    safe_mode_cb = Mock(return_value="REQUESTED")
    anomaly_cb = Mock()

    write_calls = []

    def fake_write(event: HealthEvent, action_taken: str, safe_mode_state: str):
        write_calls.append((event.event_type, action_taken, safe_mode_state))
        return tmp_path / f"{event.event_type}.json"

    monkeypatch.setattr("engine.connector_health._write_health_log", fake_write)

    propagate_health_events(
        events,
        guardrail_callback=guardrail_cb,
        safe_mode_callback=safe_mode_cb,
        anomaly_callback=anomaly_cb,
    )

    assert guardrail_cb.call_count == len(events)
    assert safe_mode_cb.call_count == sum(1 for evt in events if evt.safe_mode_required)
    assert anomaly_cb.call_count == sum(
        1
        for evt in events
        if evt.event_type in {"stale_data", "latency_degraded", "order_adapter_anomaly"}
    )
    assert len(write_calls) == len(events)
