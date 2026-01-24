import os
from datetime import datetime, timezone, timedelta

from engine.session_context import SessionContext
from engine.portfolio_risk import PortfolioRiskManager


def test_session_flow_and_capacity_preview(tmp_path):
    # Initialize contexts
    sc = SessionContext()
    prm = PortfolioRiskManager()

    base = datetime(2026, 1, 15, tzinfo=timezone.utc)
    # Sequence: Globex (03:00) -> PreMarket (07:00) -> RTH Open (09:00) -> Midday (13:00) -> Power Hour (16:00) -> Close (20:00)
    times = [
        base.replace(hour=3),
        base.replace(hour=7),
        base.replace(hour=9),
        base.replace(hour=13),
        base.replace(hour=16),
        base.replace(hour=20),
    ]

    prices = [4995.0, 5001.0, 5005.0, 4998.0, 5003.0]
    vols = [100, 200, 150, 100, 120]

    expected_sessions = [
        sc._identify_session(times[0]),
        sc._identify_session(times[1]),
        sc._identify_session(times[2]),
        sc._identify_session(times[3]),
        sc._identify_session(times[4]),
        sc._identify_session(times[5]),
    ]

    # Feed updates and assert transitions
    for t, sess in zip(times, expected_sessions):
        sc.update(t, recent_prices=prices, recent_volumes=vols, prior_high=5010.0, prior_low=4980.0,
                  overnight_high=5020.0, overnight_low=4975.0, round_levels=[5000.0, 18000.0])
        assert sc.get_session() == sess
        assert sc.flow.vwap is not None

    # Stop-run scenario: big ticks around round number
    recent_prices = [4998.0, 5003.0, 4990.0, 5006.0]
    stop = sc.flow.detect_stop_run(recent_prices, threshold_ticks=5.0)
    assert stop is True

    # Initiative move scenario: sustained move
    short_ma = 5015.0
    long_ma = 5000.0
    bias = sc.flow.compute_flow_bias(short_ma, long_ma)
    assert bias == "buy"

    # Capacity checks
    # 1) Notional exceed for ES
    res1 = prm.enforce_capacity_constraints("ES", size=5000, price=4500.0, volume_1m=200000.0, volume_5m=900000.0, volatility=0.02, depth=2000000.0)
    assert res1["rejected"] is True

    # 2) Volume exceed for small depth
    res2 = prm.enforce_capacity_constraints("NQ", size=10000, price=13000.0, volume_1m=1000.0, volume_5m=2000.0, volatility=0.03, depth=50000.0)
    assert res2["rejected"] is True

    # 3) Acceptable trade
    res3 = prm.enforce_capacity_constraints("ES", size=10, price=5000.0, volume_1m=200000.0, volume_5m=900000.0, volatility=0.015, depth=2000000.0)
    assert res3["rejected"] is False

    # Check logs were created and contain key entries
    log_dir = os.path.join("logs", "session")
    assert os.path.isdir(log_dir)
    found = False
    combined = ""
    for fn in os.listdir(log_dir):
        if fn.endswith('.log'):
            with open(os.path.join(log_dir, fn), 'r', encoding='utf-8') as f:
                combined += f.read()
    assert "Session transition" in combined
    assert "Flow update" in combined or "vwap" in combined
    assert "Stop-run detected" in combined
    assert "Capacity reject" in combined or "notional_exceeds_limit" in combined
