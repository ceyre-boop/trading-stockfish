import numpy as np

from .decision_frame import DecisionFrame
from .mcr_scenarios import PricePath


def _extract_vol(frame: DecisionFrame) -> float:
    try:
        if hasattr(frame, "market_state") and frame.market_state is not None:
            vol = getattr(frame.market_state, "volatility", None)
            if vol is not None:
                return float(vol)
    except Exception:
        pass
    try:
        evidence = getattr(frame, "market_profile_evidence", {}) or {}
        vol = evidence.get("volatility")
        if vol is not None:
            return float(vol)
    except Exception:
        pass
    return 0.001


def generate_price_paths(
    frame: DecisionFrame,
    *,
    n_paths: int,
    horizon_bars: int,
    seed: int,
) -> list[PricePath]:
    vol = _extract_vol(frame)
    start_price = 100.0
    paths: list[PricePath] = []

    for i in range(n_paths):
        rng = np.random.default_rng(seed + i)
        prices = [start_price]
        vols = []
        liq = []
        for _ in range(horizon_bars):
            r_t = rng.normal(0.0, 1.0) * vol
            new_price = prices[-1] * (1.0 + r_t)
            prices.append(float(new_price))
            vols.append(vol)
            liq.append(1.0)
        bar_indices = list(range(len(prices)))
        path = PricePath(
            bar_indices=bar_indices,
            prices=prices,
            volatility=vols,
            liquidity=liq,
            metadata={"seed": seed, "path_index": i},
        )
        paths.append(path)

    return paths
