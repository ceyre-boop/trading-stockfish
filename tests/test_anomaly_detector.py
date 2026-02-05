from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engine.anomaly_detector import AnomalyConfig, AnomalyDetector, AnomalyEvent


@pytest.fixture(autouse=True)
def disable_anomaly_logging(monkeypatch):
    monkeypatch.setattr("engine.anomaly_detector.log_anomalies", lambda events: None)
    monkeypatch.setattr("engine.anomaly_detector.ANOMALY_LOG_DIR", Path("unused"))


def _fixed_time(ts: datetime):
    def _inner():
        return ts

    return _inner


def _time_sequence(times):
    iterator = iter(times)

    def _next():
        return next(iterator)

    return _next


@pytest.fixture
def base_time():
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_drift_spike_detection(monkeypatch, base_time):
    config = AnomalyConfig(drift_threshold=2.0, feature_history=10)
    detector = AnomalyDetector(config=config)
    monkeypatch.setattr("engine.anomaly_detector._utc_now", _fixed_time(base_time))

    for _ in range(6):
        detector.update_features({"feature_a": 1.0})
    detector.update_features({"feature_a": 10.0})

    events = detector.evaluate(base_time)
    assert len(events) == 1
    event = events[0]
    assert event.anomaly_type == "drift_spike"
    assert event.severity == "medium"
    assert event.safe_mode_required is False


def test_volatility_shock_detection(monkeypatch, base_time):
    config = AnomalyConfig(volatility_threshold=0.02)
    detector = AnomalyDetector(config=config)
    monkeypatch.setattr("engine.anomaly_detector._utc_now", _fixed_time(base_time))

    detector.update_market_data(base_time, price=100.0, volatility=0.05)
    events = detector.evaluate(base_time)

    assert len(events) == 1
    event = events[0]
    assert event.anomaly_type == "volatility_shock"
    assert event.severity == "high"
    assert event.safe_mode_required is True


def test_regime_flip_detection(monkeypatch, base_time):
    detector = AnomalyDetector()
    monkeypatch.setattr("engine.anomaly_detector._utc_now", _fixed_time(base_time))

    detector.update_regime("REGIME_A")
    detector.evaluate(base_time)

    detector.update_regime("REGIME_B")
    events = detector.evaluate(base_time + timedelta(seconds=1))

    assert len(events) == 1
    event = events[0]
    assert event.anomaly_type == "regime_flip"
    assert event.severity == "medium"


def test_stale_data_detection(monkeypatch, base_time):
    config = AnomalyConfig(stale_data_threshold=timedelta(seconds=5))
    detector = AnomalyDetector(config=config)
    monkeypatch.setattr("engine.anomaly_detector._utc_now", _fixed_time(base_time))

    stale_ts = base_time - timedelta(seconds=10)
    detector.update_market_data(stale_ts, price=100.0, volatility=0.01)

    events = detector.evaluate(base_time)
    assert len(events) == 1
    event = events[0]
    assert event.anomaly_type == "stale_data"
    assert event.severity == "high"
    assert event.safe_mode_required is True


def test_repeated_safe_mode_triggers(monkeypatch, base_time):
    config = AnomalyConfig(
        safe_mode_window=timedelta(seconds=60), safe_mode_trigger_threshold=2
    )
    detector = AnomalyDetector(config=config)

    times = [
        base_time,
        base_time + timedelta(seconds=10),
        base_time + timedelta(seconds=20),
    ]
    monkeypatch.setattr("engine.anomaly_detector._utc_now", _time_sequence(times))

    detector.record_safe_mode_trigger()
    detector.record_safe_mode_trigger()
    detector.record_safe_mode_trigger()

    events = detector.evaluate(base_time + timedelta(seconds=30))
    assert len(events) == 1
    event = events[0]
    assert event.anomaly_type == "repeated_safe_mode"
    assert event.severity == "high"
    assert event.safe_mode_required is True


def test_adapter_anomaly_detection(monkeypatch, base_time):
    detector = AnomalyDetector()
    monkeypatch.setattr("engine.anomaly_detector._utc_now", _fixed_time(base_time))

    detector.record_adapter_event("missing_fill", {"order_id": "abc"})
    events = detector.evaluate(base_time)

    assert len(events) == 1
    event = events[0]
    assert event.anomaly_type == "adapter_anomaly"
    assert event.severity == "medium"


def test_storage_failure_detection(monkeypatch, base_time):
    detector = AnomalyDetector()
    monkeypatch.setattr("engine.anomaly_detector._utc_now", _fixed_time(base_time))

    detector.record_storage_failure({"reason": "fs_error"})
    events = detector.evaluate(base_time)

    assert len(events) == 1
    event = events[0]
    assert event.anomaly_type == "storage_failure"
    assert event.severity == "high"
    assert event.safe_mode_required is True


def test_multiple_anomalies_emitted(monkeypatch, base_time):
    config = AnomalyConfig(
        volatility_threshold=0.02, stale_data_threshold=timedelta(seconds=5)
    )
    detector = AnomalyDetector(config=config)
    monkeypatch.setattr("engine.anomaly_detector._utc_now", _fixed_time(base_time))

    detector.update_market_data(
        base_time - timedelta(seconds=10), price=100.0, volatility=0.05
    )
    events = detector.evaluate(base_time)

    kinds = {e.anomaly_type for e in events}
    assert len(events) >= 2
    assert {"volatility_shock", "stale_data"}.issubset(kinds)


def test_no_anomalies_when_healthy(monkeypatch, base_time):
    config = AnomalyConfig(
        volatility_threshold=0.02, stale_data_threshold=timedelta(seconds=5)
    )
    detector = AnomalyDetector(config=config)
    monkeypatch.setattr("engine.anomaly_detector._utc_now", _fixed_time(base_time))

    for _ in range(10):
        detector.update_features({"feature_a": 1.0})
    detector.update_market_data(base_time, price=100.0, volatility=0.01)

    events = detector.evaluate(base_time + timedelta(seconds=1))
    assert events == []


def test_propagation_callbacks_invoked(monkeypatch, base_time):
    config = AnomalyConfig(drift_threshold=2.0, volatility_threshold=0.02)
    guardrail_calls = []
    safe_mode_calls = []
    health_calls = []

    def guardrail_cb(evt: AnomalyEvent):
        guardrail_calls.append(evt)

    def safe_mode_cb(evt: AnomalyEvent):
        safe_mode_calls.append(evt)

    def health_cb(evt: AnomalyEvent):
        health_calls.append(evt)

    detector = AnomalyDetector(
        config=config,
        guardrail_callback=guardrail_cb,
        safe_mode_callback=safe_mode_cb,
        health_callback=health_cb,
    )

    monkeypatch.setattr("engine.anomaly_detector._utc_now", _fixed_time(base_time))

    for _ in range(6):
        detector.update_features({"feature_a": 1.0})
    detector.update_features({"feature_a": 10.0})
    detector.update_market_data(base_time, price=100.0, volatility=0.05)

    events = detector.evaluate(base_time)

    assert len(events) >= 2
    assert len(guardrail_calls) == len(events)
    assert len(safe_mode_calls) == 1
    assert len(health_calls) == len(events)
    assert any(evt.safe_mode_required for evt in safe_mode_calls)


def test_detects_repeated_safety_events():
    detector = AnomalyDetector(safety_event_threshold=2)
    events = detector.detect({"safety_event_count": 3})
    assert any(e.kind == "repeated_safety_events" for e in events)
