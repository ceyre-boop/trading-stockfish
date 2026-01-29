from dataclasses import dataclass


@dataclass
class PositionState:
    """Position tracking."""

    symbol: str
    side: str  # long/short/flat
    quantity: float  # Absolute value
    entry_price: float  # Average entry price
    current_price: float  # Current mark price
    entry_cost: float  # Total commission + slippage at entry
    unrealized_pnl: float  # Current unrealized P&L
    realized_pnl: float  # Closed P&L
