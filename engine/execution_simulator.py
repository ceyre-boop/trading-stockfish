"""
ExecutionSimulator v1

Minimal, pessimistic execution model for honest PnL and ELO.
Routes all trades through realistic spread, slippage, and cost models.

No fantasy fills. No mid-price fills in official tournaments.
"""

import os
import yaml
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, Optional, Any
from enum import Enum


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
    volume_per_minute: float     # Contracts/pips per minute
    bid_size: float              # Available at bid
    ask_size: float              # Available at ask
    typical_atr: float           # Average True Range


@dataclass
class VolatilityState:
    """Market volatility snapshot."""
    current_atr: float           # Current Average True Range
    volatility_percentile: float # 0-100 (100=most volatile)
    regime: str                  # strong/moderate/weak


@dataclass
class PositionState:
    """Position tracking."""
    symbol: str
    side: str                    # long/short/flat
    quantity: float              # Absolute value
    entry_price: float           # Average entry price
    current_price: float         # Current mark price
    entry_cost: float            # Total commission + slippage at entry
    unrealized_pnl: float        # Current unrealized P&L
    realized_pnl: float          # Closed P&L


@dataclass
class ExecutionResult:
    """Result of execution attempt."""
    action: str
    target_size: float
    actual_filled_size: float
    fill_price: float
    spread: float
    slippage: float
    transaction_cost: float
    total_cost: float            # slippage + commission
    liquidity_constraint_applied: bool
    filled_percentage: float     # actual_filled_size / target_size
    updated_position: PositionState
    
    # Session and Flow Context (v1.1.1)
    session_name: str = ""
    session_modifiers: Dict[str, float] = field(default_factory=dict)
    flow_signals: Dict[str, Any] = field(default_factory=dict)
    slippage_components: Dict[str, float] = field(default_factory=dict)
    fill_probability: float = 1.0
    partial_fill_ratio: float = 1.0


class ExecutionSimulator:
    """Minimal, pessimistic execution simulator for trading engine."""

    def __init__(self, config_path: str = "execution_config.yaml"):
        """Initialize executor with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.trade_log = []
        self._setup_logging()
        self.symbol_configs = self.config.get('symbols', {})
        self.slippage_config = self.config.get('slippage', {})
        self.partial_fill_config = self.config.get('partial_fills', {})

    def _load_config(self) -> Dict[str, Any]:
        """Load execution configuration from YAML."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_logging(self):
        """Setup execution logging."""
        log_dir = os.path.join(os.getcwd(), 'logs', 'execution')
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(
            log_dir,
            f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        self.logger = logging.getLogger('ExecutionSimulator')
        self.logger.setLevel(logging.DEBUG)
        
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s | %(message)s')
        )
        self.logger.addHandler(handler)

    def _apply_session_slippage(
        self,
        session_name: str,
        base_slippage: float,
        session_modifiers: Dict[str, float]
    ) -> Tuple[float, Dict[str, float], float]:
        """
        Apply session-specific slippage adjustments.
        
        Returns:
            (adjusted_slippage, slippage_components, fill_probability)
        """
        components = {'base': base_slippage}
        fill_prob = 1.0
        
        if session_name == "GLOBEX":
            # Overnight low liquidity: widen spreads, increase slippage, reduce fills
            session_multiplier = 1.8
            fill_prob = 0.70
            components['session_factor'] = session_multiplier
        elif session_name == "PREMARKET":
            # Moderate slippage, partial fills more likely
            session_multiplier = 1.3
            fill_prob = 0.80
            components['session_factor'] = session_multiplier
        elif session_name == "RTH_OPEN":
            # Highest slippage of day: chaotic, partial fills, rejections
            session_multiplier = 2.0
            fill_prob = 0.65
            components['session_factor'] = session_multiplier
        elif session_name == "MIDDAY":
            # Tight spreads, low slippage, high fill probability
            session_multiplier = 0.6
            fill_prob = 0.95
            components['session_factor'] = session_multiplier
        elif session_name == "POWER_HOUR":
            # Increased volatility, moderate slippage, strong continuation fills
            session_multiplier = 1.2
            fill_prob = 0.88
            components['session_factor'] = session_multiplier
        elif session_name == "CLOSE":
            # Heavy flow, potential impact on large orders
            session_multiplier = 1.4
            fill_prob = 0.82
            components['session_factor'] = session_multiplier
        else:
            session_multiplier = 1.0
            components['session_factor'] = 1.0
        
        # Apply session modifiers if present
        liq_scale = session_modifiers.get('liquidity_scale', 1.0) if session_modifiers else 1.0
        vol_scale = session_modifiers.get('volatility_scale', 1.0) if session_modifiers else 1.0
        
        # Liquidity scale reduces slippage (more liquidity = less slippage)
        # Volatility scale increases slippage (more volatility = more slippage)
        modifier_factor = (vol_scale / max(liq_scale, 0.5))
        components['modifier_factor'] = modifier_factor
        
        adjusted_slippage = base_slippage * session_multiplier * modifier_factor
        
        return adjusted_slippage, components, fill_prob

    def _apply_flow_adjustments(
        self,
        flow_signals: Dict[str, Any],
        base_slippage: float,
        base_fill_prob: float,
        session_name: str
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Apply flow-aware execution adjustments.
        
        Returns:
            (adjusted_slippage, adjusted_fill_prob, flow_components)
        """
        components = {}
        slippage_adj = base_slippage
        fill_prob_adj = base_fill_prob
        
        if not flow_signals:
            return slippage_adj, fill_prob_adj, components
        
        # Stop-run detection: increase slippage for market orders, reduce fill prob for limits
        stop_run_detected = flow_signals.get('stop_run_detected', False)
        if stop_run_detected:
            slippage_adj *= 1.5
            fill_prob_adj *= 0.80
            components['stop_run'] = {'slippage_mult': 1.5, 'fill_prob_mult': 0.80}
        
        # Initiative move detection: faster fills in direction, worse against
        initiative_detected = flow_signals.get('initiative_move_detected', False)
        if initiative_detected:
            # If session supports initiative, allow faster fills
            if session_name == "POWER_HOUR":
                slippage_adj *= 0.85
                fill_prob_adj *= 1.1
                components['initiative'] = {'slippage_mult': 0.85, 'fill_prob_mult': 1.1}
            else:
                slippage_adj *= 1.1
                fill_prob_adj *= 0.90
                components['initiative'] = {'slippage_mult': 1.1, 'fill_prob_mult': 0.90}
        
        # Level reaction score: adjust fill probability
        level_reaction_score = flow_signals.get('level_reaction_score', 0.0)
        if level_reaction_score is not None and level_reaction_score != 0:
            if level_reaction_score > 0.5:
                # Strong reaction in favor: better fills
                fill_prob_adj *= 1.05
                components['level_reaction'] = {'fill_prob_mult': 1.05}
            elif level_reaction_score < -0.5:
                # Strong reaction against: worse fills
                fill_prob_adj *= 0.95
                components['level_reaction'] = {'fill_prob_mult': 0.95}
        
        # VWAP distance: extreme distances increase slippage
        vwap_distance = flow_signals.get('vwap_distance', 0.0)
        if vwap_distance is not None:
            abs_distance = abs(vwap_distance)
            if abs_distance > 0.02:  # More than 2% from VWAP
                vwap_penalty = min(1.3, 1.0 + abs_distance * 5)
                slippage_adj *= vwap_penalty
                components['vwap_distance'] = {'slippage_mult': vwap_penalty}
        
        # Round-number proximity: widen spreads slightly
        round_level_proximity = flow_signals.get('round_level_proximity', 0.0)
        if round_level_proximity is not None and round_level_proximity > 0.8:
            slippage_adj *= 1.1
            components['round_level'] = {'slippage_mult': 1.1}
        
        return slippage_adj, fill_prob_adj, components

    def _log(self, message: str, session_name: str = "", flow_context: str = ""):
        """Log with optional session/flow context."""
        if session_name or flow_context:
            prefix = f"[{session_name}] {flow_context}: " if session_name else f"{flow_context}: "
            self.logger.info(f"{prefix}{message}")
        else:
            self.logger.info(message)

    def execute_order(
        self,
        action: str,
        target_size: float,
        mid_price: float,
        liquidity_state: LiquidityState,
        volatility_state: VolatilityState,
        symbol: str,
        current_position: Optional[PositionState] = None,
        policy_decision: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute order with session and flow awareness.
        
        This wrapper around simulate_execution applies:
        - Session-specific slippage adjustments
        - Flow-aware execution adjustments
        - Session modifier scaling
        
        Args:
            policy_decision: Optional PolicyDecision dict containing:
                - session_name
                - session_modifiers
                - flow_signals
        """
        # Execute baseline
        result = self.simulate_execution(
            action, target_size, mid_price, liquidity_state,
            volatility_state, symbol, current_position
        )
        
        if not policy_decision:
            return result
        
        # Extract session/flow context
        session_name = policy_decision.get('session_name', '')
        session_modifiers = policy_decision.get('session_modifiers', {})
        flow_signals = policy_decision.get('flow_signals', {})
        
        # Apply session-aware slippage
        session_slippage, slippage_comps, session_fill_prob = self._apply_session_slippage(
            session_name, result.slippage, session_modifiers
        )
        
        # Apply flow-aware adjustments
        final_slippage, final_fill_prob, flow_comps = self._apply_flow_adjustments(
            flow_signals, session_slippage, session_fill_prob, session_name
        )
        
        # Combine components
        all_components = {**slippage_comps, **flow_comps}
        
        # Update result with session/flow context
        result.session_name = session_name
        result.session_modifiers = session_modifiers
        result.flow_signals = flow_signals
        result.slippage_components = all_components
        result.fill_probability = final_fill_prob
        
        # Re-calculate fill price with adjusted slippage if it differs significantly
        if abs(final_slippage - result.slippage) > 0.0001:
            # Recalculate fill price with new slippage
            bid_price = mid_price - (result.spread / 2)
            ask_price = mid_price + (result.spread / 2)
            
            if action in [TradeAction.ENTER.value, TradeAction.ADD.value]:
                result.fill_price = ask_price + final_slippage
            elif action in [TradeAction.REDUCE.value, TradeAction.EXIT.value]:
                result.fill_price = bid_price - final_slippage
            elif action == TradeAction.REVERSE.value:
                result.fill_price = bid_price - final_slippage
            
            symbol_cfg = self.symbol_configs.get(symbol, {})
            tick_size = symbol_cfg.get('tick_size', 0.01)
            result.fill_price = round(result.fill_price / tick_size) * tick_size
            
            result.slippage = final_slippage
            result.total_cost = final_slippage + result.transaction_cost
        
        # Apply fill probability to actual fill (simulate partial fills)
        if final_fill_prob < 1.0:
            result.partial_fill_ratio = final_fill_prob
            # Potentially reduce actual filled size based on fill probability
            import random
            if random.random() > final_fill_prob:
                # Simulate rejection: reduce to 0 or partial
                reduction = random.uniform(0.0, 1.0 - final_fill_prob)
                result.actual_filled_size *= (1.0 - reduction)
                result.filled_percentage = result.actual_filled_size / target_size if target_size > 0 else 0
        
        # Log with session/flow context
        flow_summary = ""
        if flow_signals:
            if flow_signals.get('stop_run_detected'):
                flow_summary += "StopRun "
            if flow_signals.get('initiative_move_detected'):
                flow_summary += "Initiative "
            if flow_signals.get('vwap_distance', 0) and abs(flow_signals['vwap_distance']) > 0.02:
                flow_summary += f"VWAP{flow_signals['vwap_distance']:.1%} "
        
        self._log(
            f"Action={action} Target={target_size:.0f} Filled={result.actual_filled_size:.0f} "
            f"Price={result.fill_price:.2f} Slippage={result.slippage:.4f} "
            f"FillProb={final_fill_prob:.0%}",
            session_name,
            flow_summary.strip()
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
        current_position: Optional[PositionState] = None
    ) -> ExecutionResult:
        """
        Simulate realistic execution with spreads, slippage, and costs.
        
        Args:
            action: Trade action (ENTER/ADD/REDUCE/EXIT/REVERSE)
            target_size: Intended quantity
            mid_price: Current mid price
            liquidity_state: Market liquidity snapshot
            volatility_state: Market volatility snapshot
            symbol: Trading symbol (ES/NQ/EURUSD)
            current_position: Current position state (for updates)
        
        Returns:
            ExecutionResult with actual fill and updated position
        """
        
        if symbol not in self.symbol_configs:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        symbol_cfg = self.symbol_configs[symbol]
        
        # Step 1: Calculate bid/ask with volatility adjustment
        spread = self._calculate_spread(
            symbol, volatility_state, symbol_cfg
        )
        bid_price = mid_price - (spread / 2)
        ask_price = mid_price + (spread / 2)
        
        # Step 2: Calculate slippage
        slippage = self._calculate_slippage(
            target_size, volatility_state, liquidity_state, symbol_cfg
        )
        
        # Step 3: Determine fill price (pessimistic)
        fill_price = self._get_fill_price(
            action, bid_price, ask_price, slippage, symbol_cfg
        )
        
        # Step 4: Check for partial fills
        actual_filled_size, liquidity_constrained = self._check_partial_fill(
            target_size, liquidity_state, symbol_cfg
        )
        
        # Step 5: Calculate costs
        transaction_cost = self._calculate_costs(
            actual_filled_size, symbol_cfg, fill_price, mid_price
        )
        
        # Step 6: Update position
        updated_position = self._update_position(
            action, current_position, actual_filled_size, fill_price,
            transaction_cost, symbol, mid_price
        )
        
        # Create result
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
            filled_percentage=actual_filled_size / target_size if target_size > 0 else 0,
            updated_position=updated_position
        )
        
        # Log execution
        self._log_execution(
            result, action, mid_price, liquidity_state, volatility_state, symbol
        )
        self.trade_log.append(result)
        
        return result

    def _calculate_spread(
        self,
        symbol: str,
        volatility_state: VolatilityState,
        symbol_cfg: Dict
    ) -> float:
        """Calculate bid-ask spread with volatility adjustment."""
        fixed_spread = symbol_cfg.get('fixed_spread', 1.0)
        volatility_scale = symbol_cfg.get('spread_volatility_scale', 0.0)
        
        # Spread increases with volatility
        volatility_adjustment = volatility_scale * volatility_state.volatility_percentile / 100.0
        spread = fixed_spread + volatility_adjustment
        
        return spread

    def _calculate_slippage(
        self,
        trade_size: float,
        volatility_state: VolatilityState,
        liquidity_state: LiquidityState,
        symbol_cfg: Dict
    ) -> float:
        """
        Calculate pessimistic slippage.
        
        Formula: slippage = k * ATR * (trade_size / liquidity_scale) * pessimism_factor
        """
        k = symbol_cfg.get('slippage_coefficient', 0.15)
        liquidity_scale = symbol_cfg.get('liquidity_scale', 1000.0)
        pessimism = self.slippage_config.get('pessimism_factor', 1.0)
        min_slippage = self.slippage_config.get('min_slippage', 0.0)
        
        atr = volatility_state.current_atr
        
        # Slippage increases with trade size relative to liquidity
        slippage = k * atr * (trade_size / liquidity_scale) * pessimism
        slippage = max(slippage, min_slippage)
        
        return slippage

    def _get_fill_price(
        self,
        action: str,
        bid_price: float,
        ask_price: float,
        slippage: float,
        symbol_cfg: Dict
    ) -> float:
        """Get pessimistic fill price against the trader."""
        tick_size = symbol_cfg.get('tick_size', 0.01)
        
        if action in [TradeAction.ENTER.value, TradeAction.ADD.value]:
            # Buys: fill worse than ask (higher price)
            fill_price = ask_price + slippage
        elif action in [TradeAction.REDUCE.value, TradeAction.EXIT.value]:
            # Sells: fill worse than bid (lower price)
            fill_price = bid_price - slippage
        elif action == TradeAction.REVERSE.value:
            # Reverse: assume exit + enter, take exit price (pessimistic)
            fill_price = bid_price - slippage
        else:
            fill_price = ask_price if action in [TradeAction.ENTER.value, TradeAction.ADD.value] else bid_price
        
        # Round to tick size
        fill_price = round(fill_price / tick_size) * tick_size
        
        return fill_price

    def _check_partial_fill(
        self,
        target_size: float,
        liquidity_state: LiquidityState,
        symbol_cfg: Dict
    ) -> Tuple[float, bool]:
        """Check if liquidity constraints cause partial fills."""
        if not self.partial_fill_config.get('enabled', True):
            return target_size, False
        
        liquidity_scale = symbol_cfg.get('liquidity_scale', 1000.0)
        low_threshold = self.partial_fill_config.get('low_liquidity_threshold', 0.3)
        fill_ratio = self.partial_fill_config.get('fill_ratio_base', 0.8)
        
        # Liquidity metric: volume available relative to typical ATR * scale
        # Only trigger partial fills if TRULY constrained (very low volume relative to volatility)
        liquidity_metric = liquidity_state.volume_per_minute / (
            max(1.0, liquidity_state.typical_atr) * liquidity_scale
        )
        
        # Only apply constraint if SIGNIFICANTLY below threshold
        if liquidity_metric < low_threshold:
            # Low liquidity: only fill a fraction
            actual_size = target_size * fill_ratio
            return actual_size, True
        
        return target_size, False

    def _calculate_costs(
        self,
        filled_size: float,
        symbol_cfg: Dict,
        fill_price: float,
        mid_price: float
    ) -> float:
        """Calculate commission and fees."""
        commission_per_contract = symbol_cfg.get('commission_per_contract', 0.0)
        
        # Round-trip commission (assume we'll exit)
        total_commission = filled_size * commission_per_contract * 2
        
        # Optional: per-notional fee (typically for EURUSD/forex)
        # For now, keep simple with just commission
        
        return total_commission

    def _update_position(
        self,
        action: str,
        current_position: Optional[PositionState],
        filled_size: float,
        fill_price: float,
        transaction_cost: float,
        symbol: str,
        current_mark: float
    ) -> PositionState:
        """Update position state after execution."""
        
        if current_position is None:
            # Opening position
            current_position = PositionState(
                symbol=symbol,
                side='flat',
                quantity=0,
                entry_price=0,
                current_price=current_mark,
                entry_cost=0,
                unrealized_pnl=0,
                realized_pnl=0
            )
        
        current_qty = current_position.quantity
        side = current_position.side
        
        if action == TradeAction.ENTER.value:
            # New position
            new_qty = filled_size
            new_side = 'long'
            new_entry_price = fill_price
            new_entry_cost = transaction_cost
            new_realized = 0
        
        elif action == TradeAction.ADD.value:
            # Add to existing
            if side == 'flat':
                # Starting new position via ADD
                new_qty = filled_size
                new_side = 'long'
                new_entry_price = fill_price
                new_entry_cost = transaction_cost
            else:
                total_cost = current_position.entry_cost + transaction_cost
                new_entry_price = (
                    (current_position.entry_price * current_qty + fill_price * filled_size) /
                    (current_qty + filled_size)
                )
                new_qty = current_qty + filled_size
                new_side = side
                new_entry_cost = total_cost
            new_realized = 0
        
        elif action == TradeAction.REDUCE.value:
            # Reduce position
            pnl_per_contract = (current_mark - current_position.entry_price) * (
                1 if side == 'long' else -1
            )
            realized = pnl_per_contract * filled_size - transaction_cost
            new_qty = max(0, current_qty - filled_size)
            new_side = side if new_qty > 0 else 'flat'
            new_entry_price = current_position.entry_price
            new_entry_cost = current_position.entry_cost * (new_qty / current_qty) if current_qty > 0 else 0
            new_realized = realized
        
        elif action == TradeAction.EXIT.value:
            # Close entire position
            pnl_per_contract = (current_mark - current_position.entry_price) * (
                1 if side == 'long' else -1
            )
            realized = pnl_per_contract * current_qty - transaction_cost
            new_qty = 0
            new_side = 'flat'
            new_entry_price = 0
            new_entry_cost = 0
            new_realized = realized
        
        elif action == TradeAction.REVERSE.value:
            # Close and open opposite
            # First realize P&L on exit
            pnl_per_contract = (current_mark - current_position.entry_price) * (
                1 if side == 'long' else -1
            )
            exit_pnl = pnl_per_contract * current_qty
            
            # Then open opposite position
            new_qty = filled_size
            new_side = 'short' if side == 'long' else 'long'
            new_entry_price = fill_price
            new_entry_cost = transaction_cost
            new_realized = exit_pnl - transaction_cost
        
        else:
            # Unknown action: return unchanged
            return current_position
        
        # Calculate unrealized P&L
        if new_side == 'flat':
            unrealized = 0
        elif new_side == 'long':
            unrealized = (current_mark - new_entry_price) * new_qty - new_entry_cost
        else:  # short
            unrealized = (new_entry_price - current_mark) * new_qty - new_entry_cost
        
        updated = PositionState(
            symbol=symbol,
            side=new_side,
            quantity=new_qty,
            entry_price=new_entry_price,
            current_price=current_mark,
            entry_cost=new_entry_cost,
            unrealized_pnl=unrealized,
            realized_pnl=current_position.realized_pnl + new_realized
        )
        
        return updated

    def _log_execution(
        self,
        result: ExecutionResult,
        action: str,
        mid_price: float,
        liquidity_state: LiquidityState,
        volatility_state: VolatilityState,
        symbol: str
    ):
        """Log execution details."""
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
        """Get summary statistics of all executions."""
        if not self.trade_log:
            return {}
        
        total_trades = len(self.trade_log)
        total_slippage = sum(t.slippage for t in self.trade_log)
        total_cost = sum(t.transaction_cost for t in self.trade_log)
        avg_fill_pct = sum(t.filled_percentage for t in self.trade_log) / total_trades
        constrained_fills = sum(1 for t in self.trade_log if t.liquidity_constraint_applied)
        
        return {
            'total_trades': total_trades,
            'total_slippage': round(total_slippage, 2),
            'total_costs': round(total_cost, 2),
            'average_fill_percentage': round(avg_fill_pct * 100, 1),
            'constrained_fills': constrained_fills,
            'constrained_fill_rate': round(constrained_fills / total_trades * 100, 1) if total_trades > 0 else 0
        }

    def save_trade_log(self, output_path: str = "execution_log.json"):
        """Save trade log to JSON."""
        trade_records = []
        for trade in self.trade_log:
            trade_dict = asdict(trade.updated_position)
            trade_dict.update({
                'action': trade.action,
                'target_size': trade.target_size,
                'actual_filled_size': trade.actual_filled_size,
                'fill_price': trade.fill_price,
                'spread': trade.spread,
                'slippage': trade.slippage,
                'transaction_cost': trade.transaction_cost,
                'total_cost': trade.total_cost,
                'filled_percentage': trade.filled_percentage,
                'liquidity_constrained': trade.liquidity_constraint_applied,
                'session_name': trade.session_name,
                'session_modifiers': trade.session_modifiers,
                'flow_signals': trade.flow_signals,
                'slippage_components': trade.slippage_components,
                'fill_probability': trade.fill_probability,
                'partial_fill_ratio': trade.partial_fill_ratio
            })
            trade_records.append(trade_dict)
        
        with open(output_path, 'w') as f:
            json.dump(trade_records, f, indent=2, default=str)

    def get_execution_config(self, symbol: str) -> Dict[str, Any]:
        """Get execution configuration for a symbol."""
        return self.symbol_configs.get(symbol, {})
