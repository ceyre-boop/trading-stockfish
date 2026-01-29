from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class RegimeSignal:
    vol: str
    liq: str
    macro: str
    confidence: float
    hysteresis_applied: bool = False
    transition: str = ""
    amd: str = "NEUTRAL"
    amd_confidence: float = 0.0
    volatility_shock: bool = False
    volatility_shock_strength: float = 0.0
    trend_regime: str = "RANGE"
    swing_structure: str = "NEUTRAL"
    trend_direction: str = "RANGE"
    trend_strength: float = 0.0
    session: str = "UNKNOWN"

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d
