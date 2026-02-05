from datetime import datetime, timedelta

from engine.environment_health import EnvironmentHealthMonitor


def test_environment_health_anomalies_detected():
    thresholds = {
        "max_feed_latency_ms": 100,
        "max_tick_gap_ms": 50,
        "max_spread": 0.5,
        "min_volume": 100,
    }
    monitor = EnvironmentHealthMonitor(thresholds)

    now = datetime(2025, 1, 1, 0, 0, 1)
    prev = now - timedelta(milliseconds=200)

    market_state = {
        "timestamp_utc": prev.isoformat(),
        "best_bid": 100.0,
        "best_ask": 100.6,
        "volume": 50,
    }
    clock_state = {"timestamp_utc": now.isoformat()}

    snap = monitor.evaluate(market_state, clock_state)

    assert snap.feed_latency_ms >= 0
    assert snap.tick_gap_ms >= 0
    assert "spread_wide" in snap.anomalies
    assert "low_volume" in snap.anomalies


def test_tick_gap_updates():
    thresholds = {"max_tick_gap_ms": 10_000}
    monitor = EnvironmentHealthMonitor(thresholds)

    t1 = datetime(2025, 1, 1, 0, 0, 0)
    t2 = datetime(2025, 1, 1, 0, 0, 1)

    snap1 = monitor.evaluate(
        {"timestamp_utc": t1.isoformat()}, {"timestamp_utc": t1.isoformat()}
    )
    snap2 = monitor.evaluate(
        {"timestamp_utc": t2.isoformat()}, {"timestamp_utc": t2.isoformat()}
    )

    assert snap1.tick_gap_ms == 0
    assert snap2.tick_gap_ms == 1000
