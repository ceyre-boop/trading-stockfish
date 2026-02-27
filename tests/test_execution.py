from trading_stockfish.execution.tracker import ExecutionTracker, Fill, ExecutionStats


def _make_fill(side="BUY", intended=100.0, filled=100.1, qty=1.0, latency=5.0, oid="1"):
    return Fill(
        order_id=oid,
        intended_price=intended,
        filled_price=filled,
        quantity=qty,
        side=side,
        latency_ms=latency,
    )


class TestExecutionTracker:
    def test_empty_stats(self):
        tracker = ExecutionTracker()
        stats = tracker.stats()
        assert stats.total_fills == 0
        assert stats.fill_rate == 0.0

    def test_single_fill_stats(self):
        tracker = ExecutionTracker()
        tracker.record_attempt()
        tracker.record_fill(_make_fill())
        stats = tracker.stats()
        assert stats.total_fills == 1
        assert stats.fill_rate == 1.0

    def test_slippage_buy(self):
        tracker = ExecutionTracker()
        fill = _make_fill(side="BUY", intended=100.0, filled=100.1)
        slip = tracker.slippage_bps(fill)
        assert abs(slip - 10.0) < 1e-6   # (100.1 - 100.0) / 100.0 * 10000 = 10 bps

    def test_slippage_sell(self):
        tracker = ExecutionTracker()
        fill = _make_fill(side="SELL", intended=100.0, filled=99.9)
        slip = tracker.slippage_bps(fill)
        # adverse for seller: filled below intended
        assert slip > 0

    def test_partial_fill_rate(self):
        tracker = ExecutionTracker()
        for i in range(5):
            tracker.record_attempt()
        for i in range(3):
            tracker.record_fill(_make_fill(oid=str(i)))
        stats = tracker.stats()
        assert abs(stats.fill_rate - 0.6) < 1e-9

    def test_reset(self):
        tracker = ExecutionTracker()
        tracker.record_attempt()
        tracker.record_fill(_make_fill())
        tracker.reset()
        stats = tracker.stats()
        assert stats.total_fills == 0
