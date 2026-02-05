from typing import Any

from .decision_actions import ActionType
from .mcr_scenarios import MCRActionContext, MCRRolloutResult, PricePath


def _determine_levels(
    entry_price: float, action_ctx: MCRActionContext
) -> tuple[float, float, int]:
    stop_structure = action_ctx.decision_action.stop_structure or {}
    tp_structure = action_ctx.decision_action.tp_structure or {}

    stop_pct = None
    if "pct" in stop_structure:
        try:
            stop_pct = float(stop_structure["pct"])
        except Exception:
            stop_pct = None
    if stop_pct is None:
        stop_pct = 0.01  # default 1%

    stop_distance = entry_price * stop_pct

    rr = None
    if "rr" in tp_structure:
        try:
            rr = float(tp_structure["rr"])
        except Exception:
            rr = None
    if rr is None:
        rr = 2.0  # default 2R target

    tp_distance = stop_distance * rr

    max_bars = None
    try:
        max_bars = int(action_ctx.risk_envelope.get("max_bars"))
    except Exception:
        max_bars = None
    return stop_distance, tp_distance, max_bars if max_bars is not None else 0


def _pnL_R(
    current_price: float, entry_price: float, stop_distance: float, direction: int
) -> float:
    if stop_distance == 0:
        return 0.0
    move = current_price - entry_price
    pnl_price = move if direction > 0 else -move
    return pnl_price / stop_distance


def simulate_trade_on_path(
    price_path: PricePath, action_ctx: MCRActionContext
) -> MCRRolloutResult:
    prices = price_path.prices or []
    metadata = price_path.metadata or {}

    action = action_ctx.decision_action
    action_type = action.action_type

    if not prices:
        return MCRRolloutResult(
            realized_R=0.0,
            max_adverse_excursion=0.0,
            max_favorable_excursion=0.0,
            time_in_trade_bars=0,
            hit_stop=False,
            hit_tp=False,
            closed_by_rule=True,
            path_metadata=metadata,
        )

    if action_type in (ActionType.NO_TRADE, ActionType.MANAGE_POSITION):
        return MCRRolloutResult(
            realized_R=0.0,
            max_adverse_excursion=0.0,
            max_favorable_excursion=0.0,
            time_in_trade_bars=0,
            hit_stop=False,
            hit_tp=False,
            closed_by_rule=True,
            path_metadata=metadata,
        )

    entry_price = float(prices[0])
    stop_distance, tp_distance, max_bars = _determine_levels(entry_price, action_ctx)
    direction = 1 if action_type == ActionType.OPEN_LONG else -1

    hit_stop = False
    hit_tp = False
    closed_by_rule = False

    max_favorable = 0.0
    max_adverse = 0.0
    realized_R = 0.0
    time_in_trade = 0

    horizon = max_bars if max_bars > 0 else len(prices) - 1

    stop_level = (
        entry_price - stop_distance if direction > 0 else entry_price + stop_distance
    )
    tp_level = entry_price + tp_distance if direction > 0 else entry_price - tp_distance

    for idx, price in enumerate(prices[1 : horizon + 1], start=1):
        pnl_r = _pnL_R(price, entry_price, stop_distance, direction)
        max_favorable = max(max_favorable, pnl_r)
        max_adverse = min(max_adverse, pnl_r)
        time_in_trade = idx

        if direction > 0:
            if price <= stop_level:
                hit_stop = True
                realized_R = _pnL_R(price, entry_price, stop_distance, direction)
                break
            if price >= tp_level:
                hit_tp = True
                realized_R = _pnL_R(price, entry_price, stop_distance, direction)
                break
        else:
            if price >= stop_level:
                hit_stop = True
                realized_R = _pnL_R(price, entry_price, stop_distance, direction)
                break
            if price <= tp_level:
                hit_tp = True
                realized_R = _pnL_R(price, entry_price, stop_distance, direction)
                break
    else:
        # Horizon reached or only entry price available
        closed_by_rule = True
        exit_price = prices[min(len(prices) - 1, horizon)] if prices else entry_price
        realized_R = _pnL_R(exit_price, entry_price, stop_distance, direction)

    return MCRRolloutResult(
        realized_R=float(realized_R),
        max_adverse_excursion=float(max_adverse),
        max_favorable_excursion=float(max_favorable),
        time_in_trade_bars=int(time_in_trade),
        hit_stop=hit_stop,
        hit_tp=hit_tp,
        closed_by_rule=closed_by_rule,
        path_metadata=metadata,
    )
