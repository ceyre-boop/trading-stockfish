"""
Deterministic volume profile engine v1.
Builds per-range profiles, POC/HVN/LVN detection, value area, and context flags.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class VolumeProfileConfig:
    price_bin_size: float = 1.0
    range_mode: str = "ROLLING_WINDOW"  # or "SESSION"
    rolling_window_size: int = 200
    value_area_target: float = 0.7
    hvn_threshold_ratio: float = 0.7
    lvn_threshold_ratio: float = 0.3
    proximity_band: float = 1.5  # multiplier on bin size for near HVN/LVN


@dataclass(frozen=True)
class VolumeBin:
    price_level: float
    volume: float


@dataclass(frozen=True)
class VolumeProfile:
    bins: List[VolumeBin]
    poc_price: float
    hvn_levels: List[float]
    lvn_levels: List[float]
    value_area_low: float
    value_area_high: float
    value_area_coverage: float


def _bin_price(price: float, bin_size: float) -> float:
    if bin_size <= 0:
        return price
    return round(price / bin_size) * bin_size


def _select_range(
    bars: Sequence[Tuple[float, float]], config: VolumeProfileConfig
) -> List[Tuple[float, float]]:
    if config.range_mode.upper() == "ROLLING_WINDOW":
        return list(bars[-config.rolling_window_size :])
    return list(bars)


def _build_bins(
    bars: Iterable[Tuple[float, float]], bin_size: float
) -> List[VolumeBin]:
    buckets: Dict[float, float] = {}
    for price, vol in bars:
        if vol is None:
            continue
        bin_price = _bin_price(price, bin_size)
        buckets[bin_price] = buckets.get(bin_price, 0.0) + float(vol)
    return [VolumeBin(price_level=p, volume=v) for p, v in sorted(buckets.items())]


def _value_area(bins: List[VolumeBin], target: float) -> Tuple[float, float, float]:
    if not bins:
        return 0.0, 0.0, 0.0
    total_vol = sum(b.volume for b in bins)
    if total_vol <= 0:
        return bins[0].price_level, bins[-1].price_level, 0.0
    sorted_bins = sorted(bins, key=lambda b: (-b.volume, b.price_level))
    accum = 0.0
    selected: List[VolumeBin] = []
    for b in sorted_bins:
        selected.append(b)
        accum += b.volume
        if accum >= total_vol * target:
            break
    low = min(b.price_level for b in selected)
    high = max(b.price_level for b in selected)
    coverage = accum / total_vol if total_vol else 0.0
    return low, high, coverage


def _hvn_lvn_levels(
    bins: List[VolumeBin], config: VolumeProfileConfig
) -> Tuple[List[float], List[float]]:
    if not bins:
        return [], []
    max_vol = max(b.volume for b in bins)
    hvn_cut = max_vol * config.hvn_threshold_ratio
    lvn_cut = max_vol * config.lvn_threshold_ratio
    hvn = [b.price_level for b in bins if b.volume >= hvn_cut and b.volume > 0]
    lvn = [b.price_level for b in bins if b.volume <= lvn_cut and b.volume > 0]
    hvn_sorted = sorted(hvn)
    lvn_sorted = sorted(l for l in lvn if l not in hvn_sorted)
    return hvn_sorted, lvn_sorted


def build_volume_profile(
    bars: Sequence[Tuple[float, float]],
    config: VolumeProfileConfig | None = None,
) -> VolumeProfile:
    cfg = config or VolumeProfileConfig()
    if not bars:
        return VolumeProfile(
            bins=[],
            poc_price=0.0,
            hvn_levels=[],
            lvn_levels=[],
            value_area_low=0.0,
            value_area_high=0.0,
            value_area_coverage=0.0,
        )

    selected = _select_range(bars, cfg)
    bins = _build_bins(selected, cfg.price_bin_size)
    if not bins:
        return VolumeProfile(
            bins=[],
            poc_price=0.0,
            hvn_levels=[],
            lvn_levels=[],
            value_area_low=0.0,
            value_area_high=0.0,
            value_area_coverage=0.0,
        )

    poc_bin = max(bins, key=lambda b: (b.volume, -b.price_level))
    hvn_levels, lvn_levels = _hvn_lvn_levels(bins, cfg)
    va_low, va_high, va_cov = _value_area(bins, cfg.value_area_target)

    return VolumeProfile(
        bins=bins,
        poc_price=poc_bin.price_level,
        hvn_levels=hvn_levels,
        lvn_levels=lvn_levels,
        value_area_low=va_low,
        value_area_high=va_high,
        value_area_coverage=va_cov,
    )


def compute_volume_profile_features(
    prices: Sequence[float],
    volumes: Sequence[float] | None = None,
    config: VolumeProfileConfig | None = None,
) -> Dict[str, object]:
    cfg = config or VolumeProfileConfig()
    if volumes is None:
        volumes = [1.0 for _ in prices]
    pair_count = min(len(prices), len(volumes))
    bars: List[Tuple[float, float]] = []
    for i in range(pair_count):
        price = prices[i]
        vol = volumes[i]
        bars.append((price, float(vol)))

    # Ignore tiny trailing update to preserve replay/live parity
    if len(bars) >= 2:
        last_price = bars[-1][0]
        prev_price = bars[-2][0]
        delta = abs(last_price - prev_price)
        threshold = max(abs(prev_price) * 0.001, cfg.price_bin_size * 0.1)
        if delta <= threshold and bars[-1][1] <= max(bars[-2][1] * 0.1, 1e-8):
            bars = bars[:-1]

    profile = build_volume_profile(bars, cfg)
    if not bars:
        return {
            "poc_price": 0.0,
            "hvn_levels": [],
            "lvn_levels": [],
            "value_area_low": 0.0,
            "value_area_high": 0.0,
            "value_area_coverage": 0.0,
            "price_vs_value_area_state": "UNKNOWN",
            "near_hvn": False,
            "near_lvn": False,
        }

    current_price = bars[-1][0]
    if current_price > profile.value_area_high:
        va_state = "ABOVE"
    elif current_price < profile.value_area_low:
        va_state = "BELOW"
    else:
        va_state = "INSIDE"

    def _near(levels: List[float]) -> bool:
        band = cfg.price_bin_size * cfg.proximity_band
        return any(abs(current_price - lvl) <= band for lvl in levels)

    return {
        "poc_price": profile.poc_price,
        "hvn_levels": list(profile.hvn_levels),
        "lvn_levels": list(profile.lvn_levels),
        "value_area_low": profile.value_area_low,
        "value_area_high": profile.value_area_high,
        "value_area_coverage": profile.value_area_coverage,
        "price_vs_value_area_state": va_state,
        "near_hvn": _near(profile.hvn_levels),
        "near_lvn": _near(profile.lvn_levels),
    }
