"""
Deterministic session regime classification.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Union


class SessionRegime(str, Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OFF_HOURS = "OFF_HOURS"


def compute_session_regime(timestamp: Union[int, float]) -> SessionRegime:
    dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    if 13.0 <= hour < 21.0:
        return SessionRegime.NEW_YORK
    if 8.0 <= hour < 16.0:
        return SessionRegime.LONDON
    if 0.0 <= hour < 8.0:
        return SessionRegime.ASIA
    return SessionRegime.OFF_HOURS
