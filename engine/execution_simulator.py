import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Data classes and enums
# ---------------------------------------------------------------------------


class TradeAction(Enum):
    """Trade action types."""

    ENTER = "enter"
    ADD = "add"
    REDUCE = "reduce"
    EXIT = "exit"
    REVERSE = "reverse"


@dataclass
class LiquidityState:
    """Market liquidity snapshot."""

    volume_per_minute: float
    bid_size: float
    ask_size: float
    typical_atr: float


@dataclass
class VolatilityState:
    """Market volatility snapshot."""

    current_atr: float
    volatility_percentile: float
    regime: str


@dataclass
class PositionState:
    """Position tracking."""

    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    entry_cost: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class ExecutionResult:
    action: str = ""
    target_size: float = 0.0
    actual_filled_size: float = 0.0
    fill_price: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    transaction_cost: float = 0.0
    total_cost: float = 0.0
    liquidity_constraint_applied: bool = False
    filled_percentage: float = 0.0
    updated_position: Optional[PositionState] = None

    # Session/flow context (v1.1.1)
    session_name: str = ""
    session_modifiers: Dict[str, Any] = field(default_factory=dict)
    flow_signals: Dict[str, Any] = field(default_factory=dict)
    slippage_components: Dict[str, Any] = field(default_factory=dict)
    fill_probability: float = 1.0
    partial_fill_ratio: float = 1.0

    # Microstructure/v4 compatibility
    requested_size: float = 0.0
    filled_size: float = 0.0
    liquidity_constrained: bool = False
    fills: List[Dict[str, Any]] = field(default_factory=list)
    raw_state: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Execution Simulator (deterministic, v1.1.1 API)
# ---------------------------------------------------------------------------


class ExecutionSimulator:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "execution_config.yaml"
        self.symbol_configs = self._load_symbol_configs(self.config_path)
        self.slippage_config = self.symbol_configs.get("__slippage__", {})
        self.partial_fill_config = self.symbol_configs.get("__partial_fills__", {})
        self.position_limits = self.symbol_configs.get("__position_limits__", {})
        self.trade_log: List[ExecutionResult] = []
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    def _load_symbol_configs(self, config_path: str) -> Dict[str, Any]:
        if not config_path or not os.path.exists(config_path):
            return {}
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        symbols = cfg.get("symbols", {})
        symbols["__slippage__"] = cfg.get("slippage", {})
        symbols["__partial_fills__"] = cfg.get("partial_fills", {})
        symbols["__position_limits__"] = cfg.get("position_limits", {})
        return symbols

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------
    def _calculate_costs(self, filled_size: float, symbol_cfg: Dict[str, Any]) -> float:
        commission_per_contract = symbol_cfg.get("commission_per_contract", 0.0)
        return filled_size * commission_per_contract * 2  # round-trip

    def _update_position(
        self,
        action: str,
        current_position: Optional[PositionState],
        filled_size: float,
        fill_price: float,
        transaction_cost: float,
        symbol: str,
        current_mark: float,
    ) -> PositionState:
        if current_position is None:
            current_position = PositionState(
                symbol=symbol,
                side="flat",
                quantity=0.0,
                entry_price=0.0,
                current_price=current_mark,
                entry_cost=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )
        current_qty = current_position.quantity
        side = current_position.side
        if action == TradeAction.ENTER.value:
            new_qty = filled_size
            new_side = "long"
            new_entry_price = fill_price
            new_entry_cost = transaction_cost
            new_realized = 0.0
        elif action == TradeAction.ADD.value:
            if side == "flat":
                new_qty = filled_size
                new_side = "long"
                new_entry_price = fill_price
                new_entry_cost = transaction_cost
            else:
                total_cost = current_position.entry_cost + transaction_cost
                new_entry_price = (
                    current_position.entry_price * current_qty
                    + fill_price * filled_size
                ) / max(current_qty + filled_size, 1e-9)
                new_qty = current_qty + filled_size
                new_side = side
                new_entry_cost = total_cost
            new_realized = 0.0
        elif action == TradeAction.REDUCE.value:
            pnl_per_contract = (current_mark - current_position.entry_price) * (
                1 if side == "long" else -1
            )
            realized = pnl_per_contract * filled_size - transaction_cost
            new_qty = max(0.0, current_qty - filled_size)
            new_side = side if new_qty > 0 else "flat"
            new_entry_price = current_position.entry_price
            new_entry_cost = (
                current_position.entry_cost * (new_qty / current_qty)
                if current_qty > 0
                else 0.0
            )
            new_realized = realized
        elif action == TradeAction.EXIT.value:
            pnl_per_contract = (current_mark - current_position.entry_price) * (
                1 if side == "long" else -1
            )
            realized = pnl_per_contract * current_qty - transaction_cost
            new_qty = 0.0
            new_side = "flat"
            new_entry_price = 0.0
            new_entry_cost = 0.0
            new_realized = realized
        elif action == TradeAction.REVERSE.value:
            pnl_per_contract = (current_mark - current_position.entry_price) * (
                1 if side == "long" else -1
            )
            exit_pnl = pnl_per_contract * current_qty
            new_qty = filled_size
            new_side = "short" if side == "long" else "long"
            new_entry_price = fill_price
            new_entry_cost = transaction_cost
            new_realized = exit_pnl - transaction_cost
        else:
            return current_position

        if new_side == "flat":
            unrealized = 0.0
        elif new_side == "long":
            unrealized = (current_mark - new_entry_price) * new_qty - new_entry_cost
        else:
            unrealized = (new_entry_price - current_mark) * new_qty - new_entry_cost

        return PositionState(
            symbol=symbol,
            side=new_side,
            quantity=new_qty,
            entry_price=new_entry_price,
            current_price=current_mark,
            entry_cost=new_entry_cost,
            unrealized_pnl=unrealized,
            realized_pnl=current_position.realized_pnl + new_realized,
        )

    def _calculate_spread(
        self, symbol_cfg: Dict[str, Any], volatility_state: VolatilityState
    ) -> float:
        fixed_spread = symbol_cfg.get("fixed_spread", 1.0)
        volatility_scale = symbol_cfg.get("spread_volatility_scale", 0.0)
        volatility_adjustment = (
            volatility_scale * volatility_state.volatility_percentile / 100.0
        )
        return fixed_spread + volatility_adjustment

    def _calculate_slippage(
        self,
        trade_size: float,
        volatility_state: VolatilityState,
        liquidity_state: LiquidityState,
        symbol_cfg: Dict[str, Any],
    ) -> float:
        k = symbol_cfg.get("slippage_coefficient", 0.15)
        liquidity_scale = symbol_cfg.get("liquidity_scale", 1000.0)
        pessimism = self.slippage_config.get("pessimism_factor", 1.0)
        min_slippage = self.slippage_config.get("min_slippage", 0.0)
        atr = volatility_state.current_atr
        slippage = k * atr * (trade_size / max(liquidity_scale, 1e-6)) * pessimism
        return max(slippage, min_slippage)

    def _get_fill_price(
        self,
        action: str,
        bid_price: float,
        ask_price: float,
        slippage: float,
        tick: float,
    ) -> float:
        if action in [TradeAction.ENTER.value, TradeAction.ADD.value]:
            fill_price = ask_price + slippage
        elif action in [
            TradeAction.REDUCE.value,
            TradeAction.EXIT.value,
            TradeAction.REVERSE.value,
        ]:
            fill_price = bid_price - slippage
        else:
            fill_price = ask_price
        return round(fill_price / tick) * tick

    def _check_partial_fill(
        self,
        target_size: float,
        liquidity_state: LiquidityState,
        symbol_cfg: Dict[str, Any],
    ) -> Tuple[float, bool]:
        cfg = self.partial_fill_config or {}
        if not cfg.get("enabled", True):
            return target_size, False
        liquidity_scale = symbol_cfg.get("liquidity_scale", 1000.0)
        low_threshold = cfg.get("low_liquidity_threshold", 0.3)
        fill_ratio = cfg.get("fill_ratio_base", 0.8)
        liquidity_metric = liquidity_state.volume_per_minute / (
            max(1.0, liquidity_state.typical_atr) * liquidity_scale
        )
        if liquidity_metric < low_threshold:
            return target_size * fill_ratio, True
        return target_size, False

    # ------------------------------------------------------------------
    # Session/flow adjustments (deterministic)
    # ------------------------------------------------------------------
    def _apply_session_slippage(
        self,
        session_name: str,
        base_slippage: float,
        session_modifiers: Dict[str, float],
    ) -> Tuple[float, Dict[str, float], float]:
        components: Dict[str, float] = {"base": base_slippage}
        fill_prob = 1.0
        if session_name == "GLOBEX":
            session_multiplier = 1.8
            fill_prob = 0.70
        elif session_name == "PREMARKET":
            session_multiplier = 1.3
            fill_prob = 0.80
        elif session_name == "RTH_OPEN":
            session_multiplier = 2.0
            fill_prob = 0.65
        elif session_name == "MIDDAY":
            session_multiplier = 0.6
            fill_prob = 0.95
        elif session_name == "POWER_HOUR":
            session_multiplier = 1.2
            fill_prob = 0.88
        elif session_name == "CLOSE":
            session_multiplier = 1.4
            fill_prob = 0.82
        else:
            session_multiplier = 1.0
        components["session_factor"] = session_multiplier
        liq_scale = (
            session_modifiers.get("liquidity_scale", 1.0) if session_modifiers else 1.0
        )
        vol_scale = (
            session_modifiers.get("volatility_scale", 1.0) if session_modifiers else 1.0
        )
        modifier_factor = vol_scale / max(liq_scale, 0.5)
        components["modifier_factor"] = modifier_factor
        adjusted_slippage = base_slippage * session_multiplier * modifier_factor
        return adjusted_slippage, components, fill_prob

    def _apply_flow_adjustments(
        self,
        flow_signals: Dict[str, Any],
        base_slippage: float,
        base_fill_prob: float,
        session_name: str,
    ) -> Tuple[float, float, Dict[str, float]]:
        components: Dict[str, Any] = {}
        slippage_adj = base_slippage
        fill_prob_adj = base_fill_prob
        if not flow_signals:
            return slippage_adj, fill_prob_adj, components
        if flow_signals.get("stop_run_detected", False):
            slippage_adj *= 1.5
            fill_prob_adj *= 0.80
            components["stop_run"] = {"slippage_mult": 1.5, "fill_prob_mult": 0.80}
        if flow_signals.get("initiative_move_detected", False):
            if session_name == "POWER_HOUR":
                slippage_adj *= 0.85
                fill_prob_adj *= 1.1
                components["initiative"] = {
                    "slippage_mult": 0.85,
                    "fill_prob_mult": 1.1,
                }
            else:
                slippage_adj *= 1.1
                fill_prob_adj *= 0.90
                components["initiative"] = {
                    "slippage_mult": 1.1,
                    "fill_prob_mult": 0.90,
                }
        level_reaction_score = flow_signals.get("level_reaction_score", 0.0)
        if level_reaction_score is not None and level_reaction_score != 0:
            if level_reaction_score > 0.5:
                fill_prob_adj *= 1.05
                components["level_reaction"] = {"fill_prob_mult": 1.05}
            elif level_reaction_score < -0.5:
                fill_prob_adj *= 0.95
                components["level_reaction"] = {"fill_prob_mult": 0.95}
        vwap_distance = flow_signals.get("vwap_distance", 0.0)
        if vwap_distance is not None:
            abs_distance = abs(vwap_distance)
            if abs_distance > 0.02:
                vwap_penalty = min(1.3, 1.0 + abs_distance * 5)
                slippage_adj *= vwap_penalty
                components["vwap_distance"] = {"slippage_mult": vwap_penalty}
        round_level_proximity = flow_signals.get("round_level_proximity", 0.0)
        if round_level_proximity is not None and round_level_proximity > 0.8:
            slippage_adj *= 1.1
            components["round_level"] = {"slippage_mult": 1.1}
        return slippage_adj, fill_prob_adj, components

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def simulate_execution_v4(
        self,
        action: str,
        target_size: float,
        mid_price: float,
        liquidity_state: Optional[LiquidityState],
        volatility_state: Optional[VolatilityState],
        symbol: str,
        current_position: Optional[PositionState] = None,
        state: Optional[Dict[str, Any]] = None,
        enable_microstructure: bool = True,
    ) -> ExecutionResult:
        """Compatibility wrapper for legacy v4 microstructure tests.

        Maps the v4 call signature to the deterministic v1.1.1 semantics using
        simple, state-driven fill and slippage heuristics. No new randomness is
        introduced; this is a thin shim for older harness expectations.
        """

        del (
            liquidity_state,
            volatility_state,
            symbol,
            enable_microstructure,
        )  # unused in shim
        state = state or {}

        spread = float(state.get("spread", 1.0) or 0.0)
        liquidity_score = float(state.get("liquidity_score", 1.0) or 0.0)
        order_flow = state.get("order_flow_features", {}) or {}
        quote_pull = float(order_flow.get("quote_pulling_score", 0.0) or 0.0)
        spoofing = float(order_flow.get("spoofing_score", 0.0) or 0.0)

        slippage = max(0.0, spread * 0.1 + 0.01 * (quote_pull + spoofing))
        fill_ratio = min(1.0, max(0.0, liquidity_score / 10.0))
        filled = target_size * fill_ratio
        liquidity_constrained = fill_ratio < 0.999

        action_lower = str(action).lower()
        price_sign = 1 if action_lower in {"buy", "enter", "add"} else -1
        fill_price = mid_price + price_sign * (spread / 2 + slippage)

        result = ExecutionResult(
            action=str(action),
            target_size=target_size,
            actual_filled_size=filled,
            fill_price=fill_price,
            spread=spread,
            slippage=slippage,
            transaction_cost=0.0,
            total_cost=slippage,
            liquidity_constraint_applied=liquidity_constrained,
            filled_percentage=(filled / target_size if target_size > 0 else 0.0),
            updated_position=current_position,
            requested_size=target_size,
            filled_size=filled,
            liquidity_constrained=liquidity_constrained,
            fills=[
                {
                    "size": filled,
                    "price": fill_price,
                    "slippage": slippage,
                    "transaction_cost": 0.0,
                }
            ],
            raw_state=state,
            fill_probability=fill_ratio,
            partial_fill_ratio=fill_ratio,
        )
        return result

    def simulate_execution(
        self,
        action: str,
        target_size: float,
        mid_price: float,
        liquidity_state: LiquidityState,
        volatility_state: VolatilityState,
        symbol: str,
        current_position: Optional[PositionState] = None,
    ) -> ExecutionResult:
        if symbol not in self.symbol_configs:
            raise ValueError(f"Unknown symbol: {symbol}")
        symbol_cfg = self.symbol_configs[symbol]
        spread = self._calculate_spread(symbol_cfg, volatility_state)
        bid_price = mid_price - (spread / 2)
        ask_price = mid_price + (spread / 2)
        slippage = self._calculate_slippage(
            target_size, volatility_state, liquidity_state, symbol_cfg
        )
        tick = symbol_cfg.get("tick_size", 0.01)
        fill_price = self._get_fill_price(action, bid_price, ask_price, slippage, tick)
        actual_filled_size, liquidity_constrained = self._check_partial_fill(
            target_size, liquidity_state, symbol_cfg
        )
        transaction_cost = self._calculate_costs(actual_filled_size, symbol_cfg)
        updated_position = self._update_position(
            action,
            current_position,
            actual_filled_size,
            fill_price,
            transaction_cost,
            symbol,
            mid_price,
        )
        result = ExecutionResult(
            action=action,
            target_size=target_size,
            actual_filled_size=actual_filled_size,
            fill_price=fill_price,
            spread=spread,
            slippage=slippage,
            transaction_cost=transaction_cost,
            total_cost=slippage + transaction_cost,
            liquidity_constraint_applied=liquidity_constrained,
            filled_percentage=(
                actual_filled_size / target_size if target_size > 0 else 0.0
            ),
            updated_position=updated_position,
            requested_size=target_size,
            filled_size=actual_filled_size,
            liquidity_constrained=liquidity_constrained,
            fills=[
                {
                    "size": actual_filled_size,
                    "price": fill_price,
                    "slippage": slippage,
                    "transaction_cost": transaction_cost,
                }
            ],
        )
        self.trade_log.append(result)
        return result

    def execute_order(
        self,
        action: str,
        target_size: float,
        mid_price: float,
        liquidity_state: LiquidityState,
        volatility_state: VolatilityState,
        symbol: str,
        current_position: Optional[PositionState] = None,
        policy_decision: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        result = self.simulate_execution(
            action,
            target_size,
            mid_price,
            liquidity_state,
            volatility_state,
            symbol,
            current_position,
        )
        if not policy_decision:
            return result

        session_name = policy_decision.get("session_name", "") or ""
        session_modifiers = policy_decision.get("session_modifiers", {}) or {}
        flow_signals = policy_decision.get("flow_signals", {}) or {}

        session_slippage, session_components, session_fill_prob = (
            self._apply_session_slippage(
                session_name, result.slippage, session_modifiers
            )
        )
        final_slippage, final_fill_prob, flow_components = self._apply_flow_adjustments(
            flow_signals, session_slippage, session_fill_prob, session_name
        )
        components = {**session_components, **flow_components}

        bid_price = mid_price - (result.spread / 2)
        ask_price = mid_price + (result.spread / 2)
        tick = self.symbol_configs.get(symbol, {}).get("tick_size", 0.01)
        if action in [TradeAction.ENTER.value, TradeAction.ADD.value]:
            fill_price = ask_price + final_slippage
        elif action in [
            TradeAction.REDUCE.value,
            TradeAction.EXIT.value,
            TradeAction.REVERSE.value,
        ]:
            fill_price = bid_price - final_slippage
        else:
            fill_price = ask_price
        fill_price = round(fill_price / tick) * tick

        partial_fill_ratio = min(1.0, max(0.0, final_fill_prob))
        actual_filled = result.actual_filled_size * partial_fill_ratio

        result.action = action
        result.target_size = target_size
        result.actual_filled_size = actual_filled
        result.filled_size = actual_filled
        result.fill_price = fill_price
        result.slippage = final_slippage
        result.total_cost = final_slippage + result.transaction_cost
        result.session_name = session_name
        result.session_modifiers = session_modifiers
        result.flow_signals = flow_signals
        result.slippage_components = components
        result.fill_probability = final_fill_prob
        result.partial_fill_ratio = partial_fill_ratio
        result.liquidity_constrained = result.liquidity_constraint_applied
        result.filled_percentage = (
            actual_filled / target_size if target_size > 0 else 0.0
        )
        result.fills = [
            {
                "size": actual_filled,
                "price": fill_price,
                "slippage": final_slippage,
                "transaction_cost": result.transaction_cost,
            }
        ]
        return result

    # ------------------------------------------------------------------
    # Logging and utilities
    # ------------------------------------------------------------------
    def _log_execution(
        self,
        result: ExecutionResult,
        action: str,
        mid_price: float,
        liquidity_state: LiquidityState,
        volatility_state: VolatilityState,
        symbol: str,
    ) -> None:
        log_msg = (
            f"[{symbol}] Action={action.upper()} | "
            f"Target={result.target_size:.2f} -> Filled={result.actual_filled_size:.2f} "
            f"({result.filled_percentage*100:.1f}%) | "
            f"FillPrice={result.fill_price:.4f} | "
            f"Mid={mid_price:.4f} | "
            f"Spread={result.spread:.4f} | "
            f"Slippage={result.slippage:.4f} | "
            f"Cost={result.transaction_cost:.2f} | "
            f"TotalCost={result.total_cost:.2f} | "
            f"Liquidity=Vol{liquidity_state.volume_per_minute:.0f}/min "
            f"ATR{liquidity_state.typical_atr:.4f} | "
            f"Volatility={volatility_state.volatility_percentile:.0f}pct "
            f"Regime={volatility_state.regime} | "
            f"ConstrainedFill={result.liquidity_constraint_applied}"
        )
        self.logger.info(log_msg)

    def get_trade_log_summary(self) -> Dict[str, Any]:
        if not self.trade_log:
            return {}
        total_trades = len(self.trade_log)
        total_slippage = sum(t.slippage for t in self.trade_log)
        total_cost = sum(t.transaction_cost for t in self.trade_log)
        avg_fill_pct = sum(t.filled_percentage for t in self.trade_log) / total_trades
        constrained_fills = sum(
            1 for t in self.trade_log if t.liquidity_constraint_applied
        )
        return {
            "total_trades": total_trades,
            "total_slippage": round(total_slippage, 2),
            "total_costs": round(total_cost, 2),
            "average_fill_percentage": round(avg_fill_pct * 100, 1),
            "constrained_fills": constrained_fills,
            "constrained_fill_rate": (
                round(constrained_fills / total_trades * 100, 1)
                if total_trades > 0
                else 0.0
            ),
        }

    def save_trade_log(self, output_path: str = "execution_log.json") -> None:
        trade_records = []
        for trade in self.trade_log:
            trade_dict = (
                asdict(trade.updated_position) if trade.updated_position else {}
            )
            trade_dict.update(
                {
                    "action": trade.action,
                    "target_size": trade.target_size,
                    "actual_filled_size": trade.actual_filled_size,
                    "fill_price": trade.fill_price,
                    "spread": trade.spread,
                    "slippage": trade.slippage,
                    "transaction_cost": trade.transaction_cost,
                    "total_cost": trade.total_cost,
                    "filled_percentage": trade.filled_percentage,
                    "liquidity_constrained": trade.liquidity_constraint_applied,
                    "session_name": trade.session_name,
                    "session_modifiers": trade.session_modifiers,
                    "flow_signals": trade.flow_signals,
                    "slippage_components": trade.slippage_components,
                    "fill_probability": trade.fill_probability,
                    "partial_fill_ratio": trade.partial_fill_ratio,
                }
            )
            trade_records.append(trade_dict)
        with open(output_path, "w") as f:
            json.dump(trade_records, f, indent=2, default=str)

    def get_execution_config(self, symbol: str) -> Dict[str, Any]:
        return self.symbol_configs.get(symbol, {})
