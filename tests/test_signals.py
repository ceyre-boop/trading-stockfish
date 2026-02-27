import numpy as np
import pytest
from trading_stockfish.signals.generator import (
    SignalGenerator,
    SignalConfig,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    SIGNAL_FLAT,
)


def _rising_prices(n=50):
    return (100.0 + np.arange(n) * 0.5).tolist()


def _flat_prices(n=50):
    return [100.0] * n


class TestSignalGenerator:
    def test_momentum_length(self):
        gen = SignalGenerator()
        sigs = gen.momentum(_rising_prices())
        assert len(sigs) == 50

    def test_momentum_long_on_rising(self):
        gen = SignalGenerator(SignalConfig(momentum_window=5))
        sigs = gen.momentum(_rising_prices())
        assert np.any(sigs == SIGNAL_LONG)

    def test_mean_reversion_length(self):
        gen = SignalGenerator()
        prices = _rising_prices(100)
        sigs = gen.mean_reversion(prices)
        assert len(sigs) == 100

    def test_flat_prices_mean_rev(self):
        gen = SignalGenerator()
        sigs = gen.mean_reversion(_flat_prices(60))
        # flat prices have zero std → all flat signals
        assert np.all(sigs == SIGNAL_FLAT)

    def test_combined_regime_aware(self):
        gen = SignalGenerator()
        prices = _rising_prices(60)
        for regime in (0, 1, 2):
            sigs = gen.combined(prices, regime=regime)
            assert len(sigs) == 60
            assert set(sigs.tolist()).issubset({SIGNAL_LONG, SIGNAL_FLAT, SIGNAL_SHORT})
