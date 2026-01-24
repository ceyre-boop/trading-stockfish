from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time, timezone
from typing import List, Optional, Dict, Tuple
from statistics import mean
from .session_logging import get_session_logger, get_flow_logger


class SessionName(Enum):
    GLOBEX = "GLOBEX"
    PREMARKET = "PREMARKET"
    RTH_OPEN = "RTH_OPEN"
    MIDDAY = "MIDDAY"
    POWER_HOUR = "POWER_HOUR"
    CLOSE = "CLOSE"


@dataclass
class SessionModifiers:
    volatility_scale: float = 1.0
    liquidity_scale: float = 1.0
    trade_freq_scale: float = 1.0
    risk_scale: float = 1.0


class FlowContext:
    def __init__(self):
        self.vwap: Optional[float] = None
        self.prior_high: Optional[float] = None
        self.prior_low: Optional[float] = None
        self.overnight_high: Optional[float] = None
        self.overnight_low: Optional[float] = None
        self.round_levels: List[float] = []
        self.stop_run_detected: bool = False
        self.initiative_move: bool = False
        self.logger = get_flow_logger()

    def update_from_price_series(self, prices: List[float], volumes: Optional[List[float]] = None,
                                 prior_high: Optional[float] = None, prior_low: Optional[float] = None,
                                 overnight_high: Optional[float] = None, overnight_low: Optional[float] = None,
                                 round_levels: Optional[List[float]] = None):
        if not prices:
            return
        self.vwap = self._compute_vwap(prices, volumes)
        self.prior_high = prior_high
        self.prior_low = prior_low
        self.overnight_high = overnight_high
        self.overnight_low = overnight_low
        self.round_levels = round_levels or []
        self.logger.info(f"Flow update: vwap={self.vwap} prior_high={self.prior_high} prior_low={self.prior_low} overnight_h={self.overnight_high} overnight_l={self.overnight_low} rounds={self.round_levels}")

    def _compute_vwap(self, prices: List[float], volumes: Optional[List[float]] = None) -> float:
        if volumes and len(volumes) == len(prices):
            num = sum(p * v for p, v in zip(prices, volumes))
            den = sum(volumes) if sum(volumes) > 0 else len(prices)
            return num / den
        return mean(prices)

    def detect_stop_run(self, recent_prices: List[float], threshold_ticks: float) -> bool:
        if len(recent_prices) < 3:
            return False
        moves = [recent_prices[i+1] - recent_prices[i] for i in range(len(recent_prices)-1)]
        large_moves = [m for m in moves if abs(m) >= threshold_ticks]
        self.stop_run_detected = len(large_moves) >= 2
        if self.stop_run_detected:
            self.logger.warning(f"Stop-run detected moves={large_moves}")
        return self.stop_run_detected

    def compute_flow_bias(self, short_ma: float, long_ma: float, chop_threshold: float = 0.001) -> str:
        diff = (short_ma - long_ma) / long_ma if long_ma else 0.0
        if abs(diff) < chop_threshold:
            bias = "neutral"
        elif diff > 0:
            bias = "buy"
        else:
            bias = "sell"
        self.logger.info(f"Flow bias computed: short_ma={short_ma} long_ma={long_ma} bias={bias}")
        return bias


class SessionContext:
    def __init__(self, tz=timezone.utc):
        self.tz = tz
        self.current_session: SessionName = SessionName.GLOBEX
        self.modifiers = SessionModifiers()
        self.flow = FlowContext()
        self.logger = get_session_logger()

    def _identify_session(self, dt: datetime) -> SessionName:
        # Use UTC times for deterministic behavior
        t = dt.time()
        if t < time(6, 30):
            return SessionName.GLOBEX
        if t < time(8, 30):
            return SessionName.PREMARKET
        if t < time(11, 30):
            return SessionName.RTH_OPEN
        if t < time(15, 0):
            return SessionName.MIDDAY
        if t < time(19, 0):
            return SessionName.POWER_HOUR
        return SessionName.CLOSE

    def _compute_modifiers(self, session: SessionName) -> SessionModifiers:
        base = SessionModifiers()
        if session == SessionName.GLOBEX:
            base.volatility_scale = 0.8
            base.liquidity_scale = 0.7
            base.risk_scale = 0.9
        elif session == SessionName.PREMARKET:
            base.volatility_scale = 0.9
            base.liquidity_scale = 0.8
        elif session == SessionName.RTH_OPEN:
            base.volatility_scale = 1.2
            base.liquidity_scale = 1.2
            base.trade_freq_scale = 1.3
            base.risk_scale = 1.1
        elif session == SessionName.MIDDAY:
            base.volatility_scale = 0.9
            base.liquidity_scale = 1.0
        elif session == SessionName.POWER_HOUR:
            base.volatility_scale = 1.3
            base.liquidity_scale = 0.9
            base.risk_scale = 1.2
        elif session == SessionName.CLOSE:
            base.volatility_scale = 1.1
            base.liquidity_scale = 0.8
        return base

    def update(self, timestamp: datetime, recent_prices: Optional[List[float]] = None, recent_volumes: Optional[List[float]] = None,
               prior_high: Optional[float] = None, prior_low: Optional[float] = None,
               overnight_high: Optional[float] = None, overnight_low: Optional[float] = None,
               round_levels: Optional[List[float]] = None):
        session = self._identify_session(timestamp)
        if session != self.current_session:
            self.logger.info(f"Session transition {self.current_session.value} -> {session.value} at {timestamp.isoformat()}")
            self.current_session = session
            self.modifiers = self._compute_modifiers(session)
        # Update flow context
        if recent_prices is not None:
            self.flow.update_from_price_series(recent_prices, recent_volumes, prior_high, prior_low, overnight_high, overnight_low, round_levels)

    def get_session(self) -> SessionName:
        return self.current_session

    def get_session_modifiers(self) -> SessionModifiers:
        return self.modifiers

    def is_high_risk_period(self) -> bool:
        return self.modifiers.risk_scale > 1.1

    def is_low_liquidity_period(self) -> bool:
        return self.modifiers.liquidity_scale < 0.9
