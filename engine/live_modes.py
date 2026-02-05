from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LiveMode(str, Enum):
    SIM_REPLAY = "sim_replay"
    SIM_LIVE_FEED = "sim_live_feed"
    PAPER_TRADING = "paper_trading"
    LIVE_THROTTLED = "live_throttled"
    LIVE_FULL = "live_full"


@dataclass(frozen=True)
class ModeCapabilities:
    allow_order_routing: bool
    allow_position_updates: bool
    allow_live_prices: bool
    allow_execution_reports: bool
    description: str


MODE_CAPABILITIES = {
    LiveMode.SIM_REPLAY: ModeCapabilities(
        allow_order_routing=False,
        allow_position_updates=False,
        allow_live_prices=False,
        allow_execution_reports=False,
        description="Offline replay only.",
    ),
    LiveMode.SIM_LIVE_FEED: ModeCapabilities(
        allow_order_routing=False,
        allow_position_updates=False,
        allow_live_prices=True,
        allow_execution_reports=False,
        description="Live feed, no orders.",
    ),
    LiveMode.PAPER_TRADING: ModeCapabilities(
        allow_order_routing=False,
        allow_position_updates=True,
        allow_live_prices=True,
        allow_execution_reports=True,
        description="Simulated orders only.",
    ),
    LiveMode.LIVE_THROTTLED: ModeCapabilities(
        allow_order_routing=True,
        allow_position_updates=True,
        allow_live_prices=True,
        allow_execution_reports=True,
        description="Real orders with strict caps.",
    ),
    LiveMode.LIVE_FULL: ModeCapabilities(
        allow_order_routing=True,
        allow_position_updates=True,
        allow_live_prices=True,
        allow_execution_reports=True,
        description="Full live trading (future).",
    ),
}


VALID_MODE_TRANSITIONS = {
    LiveMode.SIM_REPLAY: {LiveMode.SIM_LIVE_FEED},
    LiveMode.SIM_LIVE_FEED: {LiveMode.PAPER_TRADING},
    LiveMode.PAPER_TRADING: {LiveMode.LIVE_THROTTLED},
    LiveMode.LIVE_THROTTLED: {LiveMode.LIVE_FULL},
    LiveMode.LIVE_FULL: set(),
}
