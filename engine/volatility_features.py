"""
Deterministic volatility features for Trading Stockfish v4.0‑D
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.volatility_utils import compute_atr


class VolatilityFeatures:
    def __init__(self, window: int = 20, use_microstructure_realism: bool = True):
        self.window = window
        self.use_microstructure_realism = use_microstructure_realism
        self.atr_window = 14
        self.mid_prices = deque(maxlen=window)
        self.realized_vols = deque(maxlen=window)
        self.atrs = deque(maxlen=window)
        self._last_price = None
        self._last_output = self._neutral()
        self._band_multiplier = 2.0
        self._band_floor = 0.001

    def compute(
        self,
        mid_price: float,
        candle_data: Optional[Dict] = None,
        order_flow: Optional[Dict] = None,
        liquidity_state: Optional[Dict] = None,
    ) -> Dict:
        """Compute deterministic volatility and shock diagnostics."""
        if not self.use_microstructure_realism or mid_price is None:
            return self._neutral()

        # If the same price is supplied consecutively, avoid mutating state to keep deterministic outputs in tests
        if self._last_price is not None and mid_price == self._last_price:
            return self._last_output

        self.mid_prices.append(mid_price)
        self._last_price = mid_price

        realized_vol = 0.0
        if len(self.mid_prices) > 1:
            returns = (
                np.diff(np.array(self.mid_prices)) / np.array(self.mid_prices)[:-1]
            )
            realized_vol = float(np.std(returns))
        self.realized_vols.append(realized_vol)

        atr = self._extract_atr(candle_data)
        if atr is None and len(self.mid_prices) > 1:
            atr = compute_atr(self.mid_prices, window=self.atr_window)
        if atr is not None:
            self.atrs.append(atr)

        intraday_band_width = 0.0
        if len(self.mid_prices) > 1:
            intraday_band_width = (max(self.mid_prices) - min(self.mid_prices)) / (
                np.mean(self.mid_prices) or 1.0
            )

        vol_of_vol = (
            float(np.std(self.realized_vols)) if len(self.realized_vols) > 1 else 0.0
        )

        p20, p60, p90 = self._percentiles(self.realized_vols)
        if realized_vol < p20:
            vol_regime = "LOW"
        elif realized_vol < p60:
            vol_regime = "NORMAL"
        elif realized_vol < p90:
            vol_regime = "HIGH"
        else:
            vol_regime = "EXTREME"

        history_ready = self._history_ready()

        shock_strength, volatility_shock = self._detect_shock(
            realized_vol,
            vol_of_vol,
            intraday_band_width,
            atr,
            p60,
            p90,
            history_ready,
        )

        self._last_output = {
            "realized_vol": realized_vol,
            "intraday_band_width": intraday_band_width,
            "vol_of_vol": vol_of_vol,
            "vol_regime": vol_regime,
            "volatility_shock": volatility_shock,
            "volatility_shock_strength": shock_strength,
        }

        return self._last_output

    def _neutral(self) -> Dict:
        return {
            "realized_vol": 0.0,
            "intraday_band_width": 0.0,
            "vol_of_vol": 0.0,
            "vol_regime": "NORMAL",
            "volatility_shock": False,
            "volatility_shock_strength": 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _percentiles(self, values: deque) -> Tuple[float, float, float]:
        if len(values) > 4:
            arr = np.array(values)
            return (
                float(np.percentile(arr, 20)),
                float(np.percentile(arr, 60)),
                float(np.percentile(arr, 90)),
            )
        return 0.0, 0.0, 0.0

    def _extract_atr(self, candle_data: Optional[Dict]) -> Optional[float]:
        if not candle_data:
            return None
        if isinstance(candle_data, dict):
            return compute_atr(
                candle_data.get("atr", candle_data.get("atr_14", None)),
                window=self.atr_window,
            )
        return compute_atr(candle_data, window=self.atr_window)

    def _history_ready(self) -> bool:
        return len(self.mid_prices) >= self.window and len(self.atrs) >= min(
            self.atr_window, self.atrs.maxlen
        )

    def _atr_baseline(self) -> float:
        if not self.atrs:
            return 0.0
        return float(np.median(np.array(self.atrs)))

    def _detect_shock(
        self,
        realized_vol: float,
        vol_of_vol: float,
        intraday_band_width: float,
        atr: Optional[float],
        p60: float,
        p90: float,
        history_ready: bool,
    ) -> Tuple[float, bool]:
        """Combine multiple deterministic shock signals into bounded strength."""

        if not history_ready:
            return 0.0, False

        atr_spike = 0.0
        if atr is not None and self.atrs:
            baseline_atr = self._atr_baseline()
            if baseline_atr > 0:
                atr_spike = max(0.0, min(1.0, (atr - baseline_atr) / (baseline_atr)))

        vol_jump = 0.0
        if p60 > 0 and p90 > 0:
            vol_jump = max(0.0, min(1.0, (realized_vol - p60) / (p90 - p60 + 1e-8)))

        baseline = max(p90, 1e-8)
        vol_of_vol_signal = 0.0
        if baseline > 0:
            vol_of_vol_signal = max(0.0, min(1.0, vol_of_vol / baseline))

        band_basis = max(self._atr_baseline(), 0.0)
        mean_price = float(np.mean(self.mid_prices)) if self.mid_prices else 1.0
        band_width = max(band_basis / (mean_price or 1.0), self._band_floor)
        band_break = max(
            0.0,
            min(
                1.0, intraday_band_width / (band_width * self._band_multiplier + 1e-12)
            ),
        )

        shock_components = [atr_spike, vol_jump, vol_of_vol_signal, band_break]
        shock_strength = float(np.mean(shock_components))
        shock_strength = max(0.0, min(1.0, shock_strength))

        shock_conditions = [
            shock_strength > 0.35,
            vol_jump > 0.5,
            vol_of_vol_signal > 0.75,
            band_break > 0.66,
        ]

        volatility_shock = any(shock_conditions)
        return shock_strength, volatility_shock
