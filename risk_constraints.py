"""
Deterministic risk constraints for trade gating.

Provides immutable config/state and pure helpers to decide whether proposed
risk is allowed, plus state updates after fills and at day end.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float = 0.01  # fraction of equity
    max_risk_per_day: float = 0.02  # fraction of equity
    max_drawdown: float = 0.1  # fraction from peak equity
    max_positions: Optional[int] = None
    max_correlation_bucket_risk: Optional[float] = None


@dataclass(frozen=True)
class RiskState:
    current_equity: float = 0.0
    open_risk: float = 0.0
    realized_pnl_today: float = 0.0
    peak_equity: float = 0.0
    risk_used_today: float = 0.0
    current_positions: int = 0

    @property
    def current_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str = ""
    approved_risk: float = 0.0


def _safe_equity(state: RiskState) -> float:
    return max(state.current_equity, 1e-9)


def check_trade_risk_allowed(
    risk_config: RiskConfig, risk_state: RiskState, proposed_risk: float
) -> RiskDecision:
    equity = _safe_equity(risk_state)
    # Normalize to fraction of equity if values are given in absolute terms.
    proposed_fraction = proposed_risk / equity

    if proposed_fraction > risk_config.max_risk_per_trade:
        return RiskDecision(False, "MAX_RISK_PER_TRADE", approved_risk=0.0)

    if (risk_state.risk_used_today + proposed_fraction) > risk_config.max_risk_per_day:
        return RiskDecision(False, "MAX_RISK_PER_DAY", approved_risk=0.0)

    if risk_state.current_drawdown > risk_config.max_drawdown:
        return RiskDecision(False, "MAX_DRAWDOWN", approved_risk=0.0)

    if (
        risk_config.max_positions is not None
        and risk_state.current_positions >= risk_config.max_positions
    ):
        return RiskDecision(False, "MAX_POSITIONS", approved_risk=0.0)

    return RiskDecision(True, "APPROVED", approved_risk=proposed_fraction)


def update_risk_state_after_fill(
    risk_state: RiskState,
    fill_risk: float,
    fill_realized_pnl: float = 0.0,
    fill_unrealized_risk_delta: float = 0.0,
    position_delta: int = 0,
) -> RiskState:
    equity = risk_state.current_equity + fill_realized_pnl
    peak_equity = max(risk_state.peak_equity, equity)
    new_open_risk = max(0.0, risk_state.open_risk + fill_unrealized_risk_delta)
    new_risk_used_today = risk_state.risk_used_today + max(0.0, fill_risk)
    new_positions = max(0, risk_state.current_positions + position_delta)

    return RiskState(
        current_equity=equity,
        open_risk=new_open_risk,
        realized_pnl_today=risk_state.realized_pnl_today + fill_realized_pnl,
        peak_equity=peak_equity,
        risk_used_today=new_risk_used_today,
        current_positions=new_positions,
    )


def update_risk_state_end_of_day(risk_state: RiskState) -> RiskState:
    # Reset day-specific counters; preserve equity/peak/open risk.
    peak_equity = max(risk_state.peak_equity, risk_state.current_equity)
    return RiskState(
        current_equity=risk_state.current_equity,
        open_risk=risk_state.open_risk,
        realized_pnl_today=0.0,
        peak_equity=peak_equity,
        risk_used_today=0.0,
        current_positions=risk_state.current_positions,
    )
