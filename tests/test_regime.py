import numpy as np
import pytest
from trading_stockfish.regime.detector import (
    RegimeDetector,
    RegimeConfig,
    REGIME_LOW,
    REGIME_NORMAL,
    REGIME_HIGH,
)


def _trending_prices(n: int = 100, drift: float = 0.0, vol: float = 0.01) -> list:
    rng = np.random.default_rng(42)
    log_returns = rng.normal(drift, vol, n - 1)
    log_prices = np.concatenate([[0.0], np.cumsum(log_returns)])
    return (100.0 * np.exp(log_prices)).tolist()


class TestRegimeDetector:
    def test_detect_length(self):
        prices = _trending_prices(50)
        det = RegimeDetector()
        regimes = det.detect(prices)
        assert len(regimes) == 50

    def test_detect_values_in_range(self):
        prices = _trending_prices(100)
        det = RegimeDetector()
        regimes = det.detect(prices)
        assert set(regimes).issubset({REGIME_LOW, REGIME_NORMAL, REGIME_HIGH})

    def test_low_vol_regime(self):
        # very low-vol prices → should produce mostly LOW_VOL
        prices = _trending_prices(200, vol=0.0005)
        cfg = RegimeConfig(window=10, low_threshold=0.15, high_threshold=0.30)
        det = RegimeDetector(cfg)
        regimes = det.detect(prices)
        # after warm-up, most bars should be low-vol
        assert np.mean(regimes[10:] == REGIME_LOW) > 0.5

    def test_high_vol_regime(self):
        # very high-vol prices → should produce mostly HIGH_VOL
        prices = _trending_prices(200, vol=0.05)
        cfg = RegimeConfig(window=10, low_threshold=0.10, high_threshold=0.20)
        det = RegimeDetector(cfg)
        regimes = det.detect(prices)
        assert np.mean(regimes[10:] == REGIME_HIGH) > 0.5

    def test_single_price(self):
        det = RegimeDetector()
        regimes = det.detect([100.0])
        assert len(regimes) == 1

    def test_rolling_volatility_shape(self):
        prices = _trending_prices(30)
        det = RegimeDetector()
        vol = det.rolling_volatility(prices)
        assert vol.shape == (30,)
        assert np.all(np.isfinite(vol))
