"""Execution adapter hardening utilities for Phase 11.

Provides throttling, retry, slippage simulation (PAPER), and reconciliation helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class ExecutionResult:
    status: str
    attempts: int
    filled: float
    remaining: float
    message: str = ""


class HardenedExecutionAdapter:
    def __init__(
        self,
        max_orders_per_second: float = 5.0,
        max_retries: int = 3,
        retry_backoff: float = 0.05,
        slippage_bps: float = 5.0,
    ):
        self.max_orders_per_second = max_orders_per_second
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.slippage_bps = slippage_bps
        self.last_order_ts = 0.0

    def _throttle(self) -> bool:
        now = time.time()
        interval = 1.0 / self.max_orders_per_second if self.max_orders_per_second else 0
        if interval and now - self.last_order_ts < interval:
            return False
        self.last_order_ts = now
        return True

    def _simulate_slippage(self, price: float) -> float:
        return price * (1 + self.slippage_bps / 10000.0)

    def send_order_with_retries(
        self, send_fn: Callable[[], Dict], price: float, quantity: float
    ) -> ExecutionResult:
        attempts = 0
        filled = 0.0
        remaining = quantity
        last_error = ""

        while attempts < self.max_retries:
            if not self._throttle():
                time.sleep(self.retry_backoff)
                continue
            attempts += 1
            try:
                result = send_fn()
                filled += result.get("filled", 0.0)
                remaining = max(quantity - filled, 0.0)
                if remaining <= 0:
                    return ExecutionResult(
                        status="FILLED",
                        attempts=attempts,
                        filled=filled,
                        remaining=0.0,
                        message="filled",
                    )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                time.sleep(self.retry_backoff)
                continue
        status = "PARTIAL" if filled else "FAILED"
        return ExecutionResult(
            status=status,
            attempts=attempts,
            filled=filled,
            remaining=remaining,
            message=last_error,
        )

    def reconcile(self, reported_position: float, internal_position: float) -> float:
        return reported_position - internal_position

    def handle_partial_fill(
        self, desired_qty: float, filled_qty: float
    ) -> Dict[str, float]:
        return {"filled": filled_qty, "remaining": max(desired_qty - filled_qty, 0.0)}
