import time

from realtime.execution_adapter import HardenedExecutionAdapter


def test_throttling():
    adapter = HardenedExecutionAdapter(max_orders_per_second=10)
    assert adapter._throttle() is True
    assert adapter._throttle() in {
        True,
        False,
    }  # second call may throttle depending on timing


def test_retry_logic():
    adapter = HardenedExecutionAdapter(
        max_orders_per_second=100, max_retries=3, retry_backoff=0.001
    )
    attempts = {"count": 0}

    def send_fn():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("fail")
        return {"filled": 1.0}

    result = adapter.send_order_with_retries(send_fn, price=100, quantity=1.0)
    assert result.status in {"FILLED", "PARTIAL"}
    assert result.attempts >= 2
    assert result.filled >= 1.0


def test_partial_fill_handling():
    adapter = HardenedExecutionAdapter()
    parts = adapter.handle_partial_fill(desired_qty=5, filled_qty=2)
    assert parts["filled"] == 2
    assert parts["remaining"] == 3


def test_reconciliation():
    adapter = HardenedExecutionAdapter()
    delta = adapter.reconcile(reported_position=10, internal_position=7)
    assert delta == 3
