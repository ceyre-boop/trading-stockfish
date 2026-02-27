import pytest
from trading_stockfish.risk.governor import RiskGovernor, RiskConfig


class TestRiskGovernor:
    def test_position_size_basic(self):
        gov = RiskGovernor(RiskConfig(risk_per_trade=0.01, max_position=1.0))
        size = gov.position_size(equity=1.0, stop_distance=0.01)
        assert abs(size - 1.0) < 1e-9   # 0.01 / 0.01 = 1.0, clipped

    def test_position_size_zero_stop(self):
        gov = RiskGovernor()
        assert gov.position_size(1.0, 0.0) == 0.0

    def test_drawdown_zero_at_peak(self):
        gov = RiskGovernor()
        dd = gov.drawdown(1.0)
        assert dd == 0.0

    def test_drawdown_increases(self):
        gov = RiskGovernor()
        gov.drawdown(1.0)
        dd = gov.drawdown(0.9)
        assert abs(dd - 0.1) < 1e-9

    def test_halt_triggered(self):
        gov = RiskGovernor(RiskConfig(max_drawdown=0.10))
        gov.drawdown(1.0)
        assert gov.is_halted(0.89)   # drawdown ≈ 11 % > 10 %

    def test_halt_not_triggered_at_peak(self):
        gov = RiskGovernor(RiskConfig(max_drawdown=0.10))
        assert not gov.is_halted(1.0)

    def test_historical_var(self):
        gov = RiskGovernor(RiskConfig(var_confidence=0.95, var_window=100))
        pnl = [-i for i in range(1, 101)]  # losses: -1 … -100
        var = gov.historical_var(pnl)
        assert var > 0   # should be a positive loss estimate

    def test_reset(self):
        gov = RiskGovernor()
        gov.drawdown(0.5)
        gov.reset()
        assert gov.drawdown(1.0) == 0.0
