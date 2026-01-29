import unittest

from engine.execution_simulator import (
    ExecutionSimulator,
    LiquidityState,
    PositionState,
    VolatilityState,
)


class TestExecutionSimulatorV4(unittest.TestCase):
    def setUp(self):
        self.executor = ExecutionSimulator(config_path="execution_config.yaml")
        self.liquidity = LiquidityState(
            volume_per_minute=1000, bid_size=500, ask_size=500, typical_atr=1.0
        )
        self.volatility = VolatilityState(
            current_atr=1.0, volatility_percentile=50, regime="moderate"
        )
        self.position = PositionState(
            symbol="ES",
            side="flat",
            quantity=0.0,
            entry_price=0.0,
            current_price=4500.0,
            entry_cost=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
        )

    def test_simulate_execution_v4(self):
        result = self.executor.simulate_execution_v4(
            action="enter",
            target_size=10.0,
            mid_price=4500.0,
            liquidity_state=self.liquidity,
            volatility_state=self.volatility,
            symbol="ES",
            current_position=self.position,
        )
        # Assert result is not None and has expected attributes
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "fill_price"))
        self.assertTrue(
            hasattr(result, "actual_filled_size") or hasattr(result, "filled_size")
        )
        # fill_price is deterministic for same input
        result2 = self.executor.simulate_execution_v4(
            action="enter",
            target_size=10.0,
            mid_price=4500.0,
            liquidity_state=self.liquidity,
            volatility_state=self.volatility,
            symbol="ES",
            current_position=self.position,
        )
        self.assertEqual(result.fill_price, result2.fill_price)
        # fill_size <= order_size
        fill_size = getattr(
            result, "actual_filled_size", getattr(result, "filled_size", None)
        )
        self.assertIsNotNone(fill_size)
        self.assertLessEqual(fill_size, 10.0)
        # slippage >= 0
        self.assertGreaterEqual(result.slippage, 0.0)


if __name__ == "__main__":
    unittest.main()

