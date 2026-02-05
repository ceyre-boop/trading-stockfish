from datetime import datetime, timedelta

from engine.anomaly_detector import AnomalyDecision, AnomalyDetector
from engine.environment_health import EnvironmentHealthSnapshot


def make_snap(ts, anomalies):
    return EnvironmentHealthSnapshot(
        timestamp=ts,
        feed_latency_ms=0,
        tick_gap_ms=0,
        spread=0,
        volume=0,
        anomalies=anomalies,
        metadata={},
    )


def test_anomaly_accumulation_triggers():
    detector = AnomalyDetector({"max_anomaly_count": 2, "window_seconds": 10})
    t0 = datetime(2025, 1, 1, 0, 0, 0)
    d1 = detector.record_and_evaluate(make_snap(t0, ["feed_latency"]))
    assert d1.triggered is False

    d2 = detector.record_and_evaluate(
        make_snap(t0 + timedelta(seconds=1), ["tick_gap"])
    )
    assert d2.triggered is True
    assert d2.reason == "environment_unstable"


def test_window_eviction_resets():
    detector = AnomalyDetector({"max_anomaly_count": 2, "window_seconds": 1})
    t0 = datetime(2025, 1, 1, 0, 0, 0)
    detector.record_and_evaluate(make_snap(t0, ["feed_latency"]))
    detector.record_and_evaluate(make_snap(t0 + timedelta(seconds=2), []))
    d3 = detector.record_and_evaluate(
        make_snap(t0 + timedelta(seconds=2), ["tick_gap"])
    )
    assert d3.triggered is False
