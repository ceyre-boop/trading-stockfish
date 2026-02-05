from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict

import numpy as np


@dataclass
class MarketProfileFeatures:
    timestamp_utc: datetime
    session_context: str
    time_of_day_bucket: str

    dist_pdh: float
    dist_pdl: float
    dist_prev_session_high: float
    dist_prev_session_low: float
    dist_weekly_high: float
    dist_weekly_low: float
    nearest_draw_side: str  # "UP" / "DOWN" / "NONE"

    atr: float
    atr_vs_session_baseline: float
    realized_vol: float
    intraday_range_vs_typical: float

    trend_slope_htf: float
    trend_dir_htf: str  # "UP" / "DOWN" / "FLAT"
    trend_slope_ltf: float
    trend_dir_ltf: str

    displacement_score: float
    num_impulsive_bars: int

    swept_pdh: bool
    swept_pdl: bool
    swept_session_high: bool
    swept_session_low: bool
    swept_equal_highs: bool
    swept_equal_lows: bool

    fvg_created: bool
    fvg_filled: bool
    fvg_respected: bool
    ob_created: bool
    ob_respected: bool
    ob_violated: bool

    volume_spike: bool
    volume_vs_mean: float

    def to_vector(self) -> Dict[str, Any]:
        # Convert to ML-friendly dict (boolean -> int, timestamp -> float epoch)
        data = asdict(self)
        data["timestamp_utc"] = self.timestamp_utc.timestamp()
        for key, val in list(data.items()):
            if isinstance(val, bool):
                data[key] = int(val)
        return data

    def to_numpy(self, feature_order: list[str]) -> np.ndarray:
        vec = self.to_vector()
        return np.array([vec.get(k) for k in feature_order], dtype=float)
