"""
Portfolio Risk Manager - Portfolio-level risk controls and exposure limits.

Tracks:
- Per-symbol exposure limits
- Total portfolio exposure
- Daily P&L (realized + unrealized)
- Position sizing constraints
- Session-aware and flow-aware risk adjustments
- ES/NQ notional and volume constraints

All operations are deterministic with explicit state tracking.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from risk_constraints import RiskConfig
from risk_constraints import RiskDecision as ConstraintDecision
from risk_constraints import (
    RiskState,
    check_trade_risk_allowed,
    update_risk_state_after_fill,
)


@dataclass
class PortfolioState:
    """Immutable snapshot of portfolio state."""

    total_capital: float
    current_exposure_per_symbol: Dict[str, float]
    current_total_exposure: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime


@dataclass
class RiskDecision:
    """Risk decision with session, flow, and capacity context."""

    action: str  # ALLOW, REDUCE_SIZE, BLOCK, FORCE_EXIT
    approved_size: float  # 0-1 normalized position size
    confidence: float  # 0-1 confidence in decision
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: str = ""

    # Session and Flow Context (v1.1.1)
    session_name: str = ""
    session_modifiers: Dict[str, float] = field(default_factory=dict)
    flow_signals: Dict[str, Any] = field(default_factory=dict)
    capacity_flags: Dict[str, bool] = field(default_factory=dict)
    risk_scaling_factors: Dict[str, float] = field(default_factory=dict)

    # Regime Context (v2.1)
    regime_label: str = ""  # TREND, RANGE, REVERSAL
    regime_confidence: float = 0.0  # [0, 1] confidence in regime
    regime_adjustments: Dict[str, Any] = field(
        default_factory=dict
    )  # What was adjusted

    # Scenario Context (v2.3)
    scenario_risk_factor: float = 1.0  # Risk scaling factor based on scenario alignment
    scenario_alignment: float = 0.0  # [0, 1] how well scenarios align with position

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "action": self.action,
            "approved_size": self.approved_size,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
            "session_name": self.session_name,
            "session_modifiers": self.session_modifiers or {},
            "flow_signals": self.flow_signals or {},
            "capacity_flags": self.capacity_flags or {},
            "risk_scaling_factors": self.risk_scaling_factors or {},
            "regime_label": self.regime_label,
            "regime_confidence": self.regime_confidence,
            "regime_adjustments": self.regime_adjustments or {},
            "scenario_risk_factor": self.scenario_risk_factor,
            "scenario_alignment": self.scenario_alignment,
        }


class PortfolioRiskManager:
    """
    Portfolio-level risk control system.

    Enforces:
    - Per-symbol exposure limits
    - Total portfolio exposure limits
    - Daily loss limits
    - Position sizing constraints

    All methods are deterministic with no hidden side effects.
    """

    def __init__(
        self,
        total_capital: float,
        max_symbol_exposure: float,
        max_total_exposure: float,
        max_daily_loss: float,
        logger: Optional[logging.Logger] = None,
        risk_config: Optional[RiskConfig] = None,
    ):
        """
        Initialize portfolio risk manager.

        Args:
            total_capital: Total capital available for trading
            max_symbol_exposure: Maximum $ exposure per symbol
            max_total_exposure: Maximum total $ exposure across all symbols
            max_daily_loss: Maximum loss allowed per day
            logger: Optional logger for state tracking
        """
        self.total_capital = total_capital
        self.max_symbol_exposure = max_symbol_exposure
        self.max_total_exposure = max_total_exposure
        self.max_daily_loss = max_daily_loss
        self.logger = logger
        self.risk_config = risk_config or RiskConfig()

        self.risk_state = RiskState(
            current_equity=total_capital,
            open_risk=0.0,
            realized_pnl_today=0.0,
            peak_equity=total_capital,
            risk_used_today=0.0,
            current_positions=0,
        )
        self.last_risk_veto_reason: str = ""

        # State tracking (updated by explicit methods)
        self.current_exposure_per_symbol: Dict[str, float] = {}
        self.current_total_exposure: float = 0.0
        self.daily_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0
        self.session_start_time: Optional[datetime] = None

    def update_exposure(self, symbol: str, position_size: int, price: float) -> None:
        """
        Update exposure for a symbol.

        Args:
            symbol: Trading symbol
            position_size: Number of contracts (can be negative for short)
            price: Current price

        Side effects:
            - Updates current_exposure_per_symbol[symbol]
            - Updates current_total_exposure
        """
        exposure_change = abs(position_size * price)

        if position_size == 0:
            # Closing position
            self.current_exposure_per_symbol[symbol] = 0.0
        else:
            # Opening/adjusting position
            self.current_exposure_per_symbol[symbol] = exposure_change

        # Recalculate total exposure
        self.current_total_exposure = sum(self.current_exposure_per_symbol.values())

        # Update risk_state open risk approximation using exposure as proxy
        exposure_fraction = self.current_total_exposure / max(self.total_capital, 1e-9)
        position_delta = 0
        if position_size == 0:
            position_delta = -1 if self.risk_state.current_positions > 0 else 0
        else:
            position_delta = 1 if self.current_exposure_per_symbol[symbol] > 0 else 0

        self.risk_state = update_risk_state_after_fill(
            self.risk_state,
            fill_risk=0.0,
            fill_realized_pnl=0.0,
            fill_unrealized_risk_delta=exposure_fraction - self.risk_state.open_risk,
            position_delta=position_delta,
        )

        if self.logger:
            self.logger.info(
                f"[ExposureUpdate] Symbol: {symbol}, "
                f"Size: {position_size}, Price: {price:.2f}, "
                f"Exposure: ${exposure_change:.2f}, "
                f"Total: ${self.current_total_exposure:.2f}"
            )

    def update_pnl(self, realized: float, unrealized: float) -> None:
        """
        Update P&L tracking.

        Args:
            realized: Realized P&L (from closed positions)
            unrealized: Unrealized P&L (from open positions)

        Side effects:
            - Updates realized_pnl, unrealized_pnl, daily_pnl
        """
        self.realized_pnl = realized
        self.unrealized_pnl = unrealized
        self.daily_pnl = realized + unrealized
        self.risk_state = update_risk_state_after_fill(
            self.risk_state,
            fill_risk=0.0,
            fill_realized_pnl=realized,
            fill_unrealized_risk_delta=0.0,
            position_delta=0,
        )

        if self.logger:
            self.logger.info(
                f"[PnLUpdate] Realized: ${realized:.2f}, "
                f"Unrealized: ${unrealized:.2f}, "
                f"Daily: ${self.daily_pnl:.2f}"
            )

    def can_open_position(self, symbol: str, target_size: int, price: float) -> bool:
        """
        Check if position can be opened given exposure limits.

        Returns:
            True if position respects all limits, False otherwise

        Side effects: None (read-only check)
        """
        # Calculate exposure if we open this position
        target_exposure = abs(target_size * price)

        # Check 1: Symbol-level limit
        if target_exposure > self.max_symbol_exposure:
            if self.logger:
                self.logger.warning(
                    f"[PositionBlocked] {symbol}: Exposure ${target_exposure:.2f} "
                    f"exceeds symbol limit ${self.max_symbol_exposure:.2f}"
                )
            return False

        # Check 2: Total exposure limit
        # Current total - old exposure for this symbol + new exposure
        current_symbol_exposure = self.current_exposure_per_symbol.get(symbol, 0.0)
        projected_total = (
            self.current_total_exposure - current_symbol_exposure + target_exposure
        )

        if projected_total > self.max_total_exposure:
            if self.logger:
                self.logger.warning(
                    f"[PositionBlocked] {symbol}: Total exposure ${projected_total:.2f} "
                    f"would exceed limit ${self.max_total_exposure:.2f}"
                )
            return False

        # Check 3: Daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            if self.logger:
                self.logger.warning(
                    f"[PositionBlocked] {symbol}: Daily loss ${abs(self.daily_pnl):.2f} "
                    f"exceeds limit ${self.max_daily_loss:.2f}"
                )
            return False

        # Risk constraint gate (deterministic, fraction of equity)
        proposed_risk = target_exposure / max(self.total_capital, 1e-9)
        decision: ConstraintDecision = check_trade_risk_allowed(
            self.risk_config, self.risk_state, proposed_risk
        )
        if not decision.allowed:
            self.last_risk_veto_reason = decision.reason
            if self.logger:
                self.logger.warning(
                    f"[RiskBlocked] {symbol}: {decision.reason}, proposed_risk={proposed_risk:.6f}"
                )
            return False
        self.last_risk_veto_reason = ""

        return True

    def should_force_exit(self) -> bool:
        """
        Check if daily loss threshold exceeded (force exit all).

        Returns:
            True if daily_pnl < -max_daily_loss, False otherwise

        Side effects: None (read-only check)
        """
        if self.daily_pnl < -self.max_daily_loss:
            if self.logger:
                self.logger.error(
                    f"[ForceExit] Daily loss ${abs(self.daily_pnl):.2f} "
                    f"exceeds threshold ${self.max_daily_loss:.2f}"
                )
            return True
        return False

    def reset_daily_limits(self) -> None:
        """
        Reset daily P&L tracking at session boundaries.

        Side effects:
            - Resets daily_pnl, realized_pnl, unrealized_pnl to 0
            - Preserves current_exposure tracking
        """
        self.daily_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.session_start_time = datetime.now()
        from risk_constraints import update_risk_state_end_of_day

        self.risk_state = update_risk_state_end_of_day(self.risk_state)

        if self.logger:
            self.logger.info(
                f"[DailyReset] Daily limits reset. Session start: {self.session_start_time}"
            )

    def get_available_capital(self) -> float:
        """
        Get available capital for new positions.

        Returns:
            Amount of capital available (max_total_exposure - current_total_exposure)

        Side effects: None
        """
        return max(0.0, self.max_total_exposure - self.current_total_exposure)

    def get_capital_utilization_percent(self) -> float:
        """
        Get portfolio capital utilization as percentage.

        Returns:
            (current_total_exposure / max_total_exposure) * 100

        Side effects: None
        """
        if self.max_total_exposure == 0:
            return 0.0
        return (self.current_total_exposure / self.max_total_exposure) * 100.0

    def get_state_snapshot(self) -> PortfolioState:
        """
        Get immutable snapshot of current portfolio state.

        Returns:
            PortfolioState with all current metrics

        Side effects: None
        """
        return PortfolioState(
            total_capital=self.total_capital,
            current_exposure_per_symbol=dict(self.current_exposure_per_symbol),
            current_total_exposure=self.current_total_exposure,
            daily_pnl=self.daily_pnl,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            timestamp=datetime.now(),
        )

    def flatten_all_positions(self) -> None:
        """
        Clear all position tracking (for forced exit scenario).

        Side effects:
            - Clears current_exposure_per_symbol
            - Sets current_total_exposure to 0
        """
        self.current_exposure_per_symbol.clear()
        self.current_total_exposure = 0.0

        if self.logger:
            self.logger.error("[FlattenAll] All positions forcefully closed")

    def _apply_session_risk_scaling(
        self, session_name: str, base_size: float, session_modifiers: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply session-specific risk scaling to position size.

        Returns:
            (scaled_size, scaling_factors)
        """
        factors = {}
        size_multiplier = 1.0

        if session_name == "GLOBEX":
            # Overnight: reduce size, increase risk penalty
            size_multiplier = 0.5
            factors["session_factor"] = 0.5
        elif session_name == "PREMARKET":
            # Moderate reduction before macro events
            size_multiplier = 0.7
            factors["session_factor"] = 0.7
        elif session_name == "RTH_OPEN":
            # Strictest limits at open
            size_multiplier = 0.4
            factors["session_factor"] = 0.4
        elif session_name == "MIDDAY":
            # Allow controlled scaling
            size_multiplier = 1.0
            factors["session_factor"] = 1.0
        elif session_name == "POWER_HOUR":
            # Allow trend-following scaling
            size_multiplier = 1.1
            factors["session_factor"] = 1.1
        elif session_name == "CLOSE":
            # Tighten new entries
            size_multiplier = 0.6
            factors["session_factor"] = 0.6
        else:
            factors["session_factor"] = 1.0

        # Apply session modifiers if present
        liq_scale = (
            session_modifiers.get("liquidity_scale", 1.0) if session_modifiers else 1.0
        )
        risk_scale = (
            session_modifiers.get("risk_scale", 1.0) if session_modifiers else 1.0
        )

        # Risk scale affects position size directly
        modifier_factor = (1.0 / max(liq_scale, 0.5)) * risk_scale
        factors["modifier_factor"] = modifier_factor

        scaled_size = base_size * size_multiplier * modifier_factor

        return scaled_size, factors

    def _apply_flow_risk_adjustments(
        self, flow_signals: Dict[str, Any], base_size: float, session_name: str
    ) -> Tuple[float, Dict[str, float], list]:
        """
        Apply flow-aware risk adjustments to position size.

        Returns:
            (adjusted_size, adjustments, warnings)
        """
        adjustments = {}
        warnings = []
        size_adj = base_size

        if not flow_signals:
            return size_adj, adjustments, warnings

        # Stop-run detection: reduce size, increase caution
        stop_run_detected = flow_signals.get("stop_run_detected", False)
        if stop_run_detected:
            size_adj *= 0.6
            adjustments["stop_run"] = 0.6
            warnings.append("Stop-run detected: size reduced 40%")

        # Initiative move: allow scaling in direction, penalize against
        initiative_detected = flow_signals.get("initiative_move_detected", False)
        if initiative_detected:
            if session_name == "POWER_HOUR":
                size_adj *= 1.15
                adjustments["initiative"] = 1.15
            else:
                size_adj *= 0.75
                adjustments["initiative"] = 0.75
                warnings.append(
                    "Initiative detected outside POWER_HOUR: caution applied"
                )

        # Level reaction score: adjust based on reaction strength
        level_reaction_score = flow_signals.get("level_reaction_score", 0.0)
        if level_reaction_score is not None and level_reaction_score != 0:
            if level_reaction_score > 0.6:
                # Strong reaction in favor: allow scaling
                size_adj *= 1.1
                adjustments["level_reaction"] = 1.1
            elif level_reaction_score < -0.6:
                # Strong reaction against: reduce size
                size_adj *= 0.7
                adjustments["level_reaction"] = 0.7
                warnings.append("Strong adverse level reaction: size reduced")

        # VWAP distance: penalize mean-reversion attempts when extreme
        vwap_distance = flow_signals.get("vwap_distance", 0.0)
        if vwap_distance is not None:
            abs_distance = abs(vwap_distance)
            if abs_distance > 0.03:  # More than 3% from VWAP
                size_adj *= 0.5
                adjustments["vwap_distance"] = 0.5
                warnings.append(
                    f"Extreme VWAP distance ({abs_distance:.1%}): size halved"
                )

        # Round-level proximity: widen risk buffers
        round_level_proximity = flow_signals.get("round_level_proximity", 0.0)
        if round_level_proximity is not None and round_level_proximity > 0.85:
            size_adj *= 0.85
            adjustments["round_level"] = 0.85
            warnings.append("Near round level: size reduced for wider buffer")

        return size_adj, adjustments, warnings

    def _enforce_capacity_limits(
        self,
        symbol: str,
        target_size: float,
        price: float,
        volume_state: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, Dict[str, bool], list]:
        """
        Enforce ES/NQ notional and volume capacity limits.

        Returns:
            (allowed, capacity_flags, warnings)
        """
        capacity_flags = {}
        warnings = []
        allowed = True

        # Notional limits (per symbol) - much higher for institutional trading
        es_notional_limit = 5000000 if symbol == "ES" else 3000000
        nq_notional_limit = 3000000 if symbol == "NQ" else 5000000

        target_notional = abs(target_size * price)
        notional_limit = es_notional_limit if symbol == "ES" else nq_notional_limit

        if target_notional > notional_limit:
            capacity_flags["notional_exceeded"] = True
            warnings.append(
                f"Target notional ${target_notional:.0f} exceeds limit ${notional_limit:.0f}"
            )
            allowed = False
        else:
            capacity_flags["notional_ok"] = True

        # Volume limits (if volume state provided)
        if volume_state:
            vol_1min = volume_state.get("volume_1min", 0)
            vol_5min = volume_state.get("volume_5min", 0)

            # Don't take more than 5% of 1-minute volume
            if vol_1min > 0 and abs(target_size) > 0.05 * vol_1min:
                capacity_flags["volume_1min_exceeded"] = True
                warnings.append(f"Size exceeds 5% of 1-min volume")
                allowed = False
            else:
                capacity_flags["volume_1min_ok"] = True

            # Don't take more than 3% of 5-minute volume
            if vol_5min > 0 and abs(target_size) > 0.03 * vol_5min:
                capacity_flags["volume_5min_exceeded"] = True
                warnings.append(f"Size exceeds 3% of 5-min volume")
                allowed = False
            else:
                capacity_flags["volume_5min_ok"] = True

        return allowed, capacity_flags, warnings

    def evaluate_risk_with_context(
        self,
        symbol: str,
        target_size: float,
        price: float,
        policy_decision: Optional[Dict[str, Any]] = None,
        volume_state: Optional[Dict[str, float]] = None,
        regime_label: str = "",
        regime_confidence: float = 0.0,
        eval_result: Optional[Dict[str, Any]] = None,
    ) -> RiskDecision:
        """
        Comprehensive risk evaluation incorporating session, flow, capacity, and regime (v2.1).

        Regime-aware risk scaling (v2.1):
        - TREND: Slightly larger size in trend direction, reduce against
        - RANGE: Overall size reduction, avoid large positions
        - REVERSAL: Significant size reduction until reversal confirmed

        Args:
            policy_decision: Dict with session_name, session_modifiers, flow_signals
            volume_state: Dict with volume_1min, volume_5min
            regime_label: Current regime (TREND, RANGE, REVERSAL)
            regime_confidence: [0, 1] confidence in regime classification

        Returns:
            RiskDecision with full context including regime adjustments
        """
        session_name = (
            policy_decision.get("session_name", "") if policy_decision else ""
        )
        session_modifiers = (
            policy_decision.get("session_modifiers", {}) if policy_decision else {}
        )
        flow_signals = (
            policy_decision.get("flow_signals", {}) if policy_decision else {}
        )

        warnings = []

        # Step 1: Check basic capacity limits (on unscaled size)
        capacity_ok, capacity_flags, cap_warnings = self._enforce_capacity_limits(
            symbol, target_size, price, volume_state
        )
        warnings.extend(cap_warnings)

        if not capacity_ok:
            self._log(
                f"Capacity check FAILED for {symbol}: {cap_warnings}", session_name
            )
            return RiskDecision(
                action="BLOCK",
                approved_size=0.0,
                confidence=1.0,
                reasoning="Capacity limits exceeded",
                session_name=session_name,
                session_modifiers=session_modifiers,
                flow_signals=flow_signals,
                capacity_flags=capacity_flags,
                risk_scaling_factors={},
            )

        # Step 2: Check daily loss limit (early exit)
        if self.daily_pnl < -self.max_daily_loss:
            self._log(
                f"Daily loss limit exceeded: {self.daily_pnl:.2f} / -{self.max_daily_loss:.2f}",
                session_name,
            )
            return RiskDecision(
                action="FORCE_EXIT",
                approved_size=0.0,
                confidence=1.0,
                reasoning="Daily loss limit exceeded",
                session_name=session_name,
                session_modifiers=session_modifiers,
                flow_signals=flow_signals,
                capacity_flags=capacity_flags,
                risk_scaling_factors={},
            )

        # Step 3: Check exposure limits (on unscaled target size)
        target_exposure = abs(target_size * price)
        current_symbol_exp = self.current_exposure_per_symbol.get(symbol, 0.0)
        projected_symbol_exp = current_symbol_exp + target_exposure

        # Check symbol exposure limit
        if projected_symbol_exp > self.max_symbol_exposure:
            # Reduce unscaled size to fit limit
            available = self.max_symbol_exposure - current_symbol_exp
            reduced_size = available / max(price, 0.01)
            self._log(
                f"Symbol exposure limit hit for {symbol}: reduced from {target_size:.0f} to {reduced_size:.0f}",
                session_name,
            )
            warnings.append(f"Size reduced due to symbol exposure limit")

            # Now apply scaling to the reduced size
            session_scaled_size, session_factors = self._apply_session_risk_scaling(
                session_name, reduced_size, session_modifiers
            )
            flow_adjusted_size, flow_adjustments, flow_warnings = (
                self._apply_flow_risk_adjustments(
                    flow_signals, session_scaled_size, session_name
                )
            )

            return RiskDecision(
                action="REDUCE_SIZE",
                approved_size=flow_adjusted_size / max(target_size, 1),
                confidence=0.7,
                reasoning="Reduced to fit symbol exposure limit",
                session_name=session_name,
                session_modifiers=session_modifiers,
                flow_signals=flow_signals,
                capacity_flags=capacity_flags,
                risk_scaling_factors={**session_factors, **flow_adjustments},
            )

        # Check total exposure limit
        projected_total = (
            self.current_total_exposure - current_symbol_exp + projected_symbol_exp
        )

        if projected_total > self.max_total_exposure:
            # Reduce unscaled size to fit portfolio limit
            available = self.max_total_exposure - (
                self.current_total_exposure - current_symbol_exp
            )
            reduced_size = available / max(price, 0.01)
            self._log(
                f"Portfolio total exposure limit hit: reduced from {target_size:.0f} to {reduced_size:.0f}",
                session_name,
            )
            warnings.append("Size reduced due to portfolio exposure limit")

            # Now apply scaling to the reduced size
            session_scaled_size, session_factors = self._apply_session_risk_scaling(
                session_name, reduced_size, session_modifiers
            )
            flow_adjusted_size, flow_adjustments, flow_warnings = (
                self._apply_flow_risk_adjustments(
                    flow_signals, session_scaled_size, session_name
                )
            )

            return RiskDecision(
                action="REDUCE_SIZE",
                approved_size=flow_adjusted_size / max(target_size, 1),
                confidence=0.6,
                reasoning="Reduced to fit portfolio exposure limit",
                session_name=session_name,
                session_modifiers=session_modifiers,
                flow_signals=flow_signals,
                capacity_flags=capacity_flags,
                risk_scaling_factors={**session_factors, **flow_adjustments},
            )

        # Step 4: Apply session-aware risk scaling (target_size fits limits)
        session_scaled_size, session_factors = self._apply_session_risk_scaling(
            session_name, target_size, session_modifiers
        )

        # Step 5: Apply flow-aware risk adjustments
        flow_adjusted_size, flow_adjustments, flow_warnings = (
            self._apply_flow_risk_adjustments(
                flow_signals, session_scaled_size, session_name
            )
        )
        warnings.extend(flow_warnings)

        # Step 6: Decision is ALLOW (exposure limits passed, scaling applied)
        confidence = min(1.0, 0.9 - len(flow_warnings) * 0.1)

        self._log(
            f"Risk decision ALLOW for {symbol}: size={flow_adjusted_size:.0f}, confidence={confidence:.0%}",
            session_name,
        )

        # Step 7: Apply scenario-aware risk scaling (v2.3)
        scenario_result = (
            eval_result.get("scenario_result", None) if eval_result else None
        )
        eval_score = eval_result.get("eval_score", 0.0) if eval_result else 0.0
        scenario_scaled_size = flow_adjusted_size
        scenario_risk_factor = 1.0
        scenario_alignment = 0.0

        if scenario_result:
            scenario_scaled_size, scenario_risk_factor = (
                self._apply_scenario_risk_scaling(
                    flow_adjusted_size, scenario_result, eval_score
                )
            )
            scenario_alignment = (
                scenario_result.regime_alignment
                if hasattr(scenario_result, "regime_alignment")
                else 0.0
            )
            self._log(
                f"Scenario risk scaled: {flow_adjusted_size:.0f} → {scenario_scaled_size:.0f} (factor={scenario_risk_factor:.2f})",
                session_name,
            )

        return RiskDecision(
            action="ALLOW",
            approved_size=scenario_scaled_size / max(target_size, 1),
            confidence=confidence,
            reasoning="; ".join(warnings) if warnings else "All checks passed",
            session_name=session_name,
            session_modifiers=session_modifiers,
            flow_signals=flow_signals,
            capacity_flags=capacity_flags,
            risk_scaling_factors={**session_factors, **flow_adjustments},
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            regime_adjustments=self._apply_regime_risk_scaling(
                scenario_scaled_size, regime_label, regime_confidence
            ),
            scenario_risk_factor=scenario_risk_factor,
            scenario_alignment=scenario_alignment,
        )

    def _apply_regime_risk_scaling(
        self, approved_size: float, regime_label: str, regime_confidence: float
    ) -> Dict[str, Any]:
        """
        Apply regime-conditioned risk scaling to approved position size.

        TREND (v2.1):
        - Slightly larger size in trend direction (no size reduction)
        - Reduce size against trend
        - Allow slightly more risk in aligned positions

        RANGE (v2.1):
        - Overall size reduction (reduce to 70%)
        - Avoid large positions
        - Tighten risk control

        REVERSAL (v2.1):
        - Significant size reduction (reduce to 50%)
        - Increase caution on both sides
        - Tight risk control until confirmed

        Args:
            approved_size: Size approved by capacity/session checks
            regime_label: Current regime
            regime_confidence: [0, 1] confidence in regime

        Returns:
            Dict with adjustment details (scaling_factor, reason, adjustment)
        """
        if not regime_label or regime_confidence < 0.3:
            return {}

        if regime_label == "TREND":
            # In TREND: Allow approved size, slight boost for trend-aligned
            return {
                "regime": "TREND",
                "scaling_factor": 1.0,
                "reason": "TREND regime - allow full approved size",
            }

        elif regime_label == "RANGE":
            # In RANGE: Reduce to 70% of approved size
            scaling = 0.70 * regime_confidence + 1.0 * (1 - regime_confidence)
            return {
                "regime": "RANGE",
                "scaling_factor": scaling,
                "reason": f"RANGE regime - reduce to {scaling:.0%} of approved size",
                "adjustment": approved_size * (scaling - 1.0),
            }

        elif regime_label == "REVERSAL":
            # In REVERSAL: Reduce to 50% of approved size
            scaling = 0.50 * regime_confidence + 1.0 * (1 - regime_confidence)
            return {
                "regime": "REVERSAL",
                "scaling_factor": scaling,
                "reason": f"REVERSAL regime - reduce to {scaling:.0%} of approved size",
                "adjustment": approved_size * (scaling - 1.0),
            }

        return {}

    def _apply_scenario_risk_scaling(
        self,
        approved_size: float,
        scenario_result: Any,  # ScenarioResult from v2.2
        eval_score: float,
    ) -> Tuple[float, float]:
        """
        Apply scenario-aware risk scaling to approved position size (v2.3).

        Scenario Risk Scaling Logic:
        - If scenario strongly aligned with position direction:
            * increase approved size by +10% (within limits)
        - If scenario strongly misaligned:
            * reduce approved size by -20%
            * block aggressive entries
        - If CHOP dominates (RANGE regime):
            * reduce size globally (0.75x)
            * tighten stops

        Args:
            approved_size: Size approved by capacity/session/regime checks
            scenario_result: ScenarioResult with probability distributions
            eval_score: Evaluation score [-1, +1]

        Returns:
            (scaled_size, scenario_risk_factor)
        """
        if not scenario_result:
            return approved_size, 1.0

        prob_up = (
            scenario_result.probability_up
            if hasattr(scenario_result, "probability_up")
            else 0.33
        )
        prob_down = (
            scenario_result.probability_down
            if hasattr(scenario_result, "probability_down")
            else 0.33
        )
        prob_chop = (
            scenario_result.probability_chop
            if hasattr(scenario_result, "probability_chop")
            else 0.34
        )
        scenario_alignment = (
            scenario_result.regime_alignment
            if hasattr(scenario_result, "regime_alignment")
            else 0.5
        )
        regime_label = (
            scenario_result.regime_label
            if hasattr(scenario_result, "regime_label")
            else ""
        )

        scenario_risk_factor = 1.0

        # Determine alignment: scenarios aligned with eval_score direction?
        if eval_score > 0:
            # Bullish eval
            alignment_prob = prob_up
            misalignment_prob = prob_down
        elif eval_score < 0:
            # Bearish eval
            alignment_prob = prob_down
            misalignment_prob = prob_up
        else:
            # Neutral eval
            alignment_prob = prob_chop
            misalignment_prob = max(prob_up, prob_down)

        # Strong alignment: +10% size boost (alignment prob > 50%)
        if alignment_prob > 0.50:
            scenario_risk_factor = 1.10
            self._log(f"Scenario aligned {alignment_prob:.0%}: +10% size boost")

        # Strong misalignment: -20% size reduction (misalignment > 50%)
        elif misalignment_prob > 0.50:
            scenario_risk_factor = 0.80
            self._log(
                f"Scenario misaligned (against prob {misalignment_prob:.0%}): -20% size reduction"
            )

        # CHOP dominates (RANGE): -25% size reduction
        elif prob_chop > 0.50 and regime_label == "RANGE":
            scenario_risk_factor = 0.75
            self._log(f"CHOP dominates {prob_chop:.0%} in RANGE: -25% size reduction")

        # Extreme CHOP (>65%): -35% size reduction
        elif prob_chop > 0.65:
            scenario_risk_factor = 0.65
            self._log(f"Extreme CHOP {prob_chop:.0%}: -35% size reduction")

        scaled_size = approved_size * scenario_risk_factor

        return scaled_size, scenario_risk_factor

    def _log(self, message: str, session_name: str = ""):
        """Log with optional session context."""
        if self.logger:
            prefix = f"[{session_name}] " if session_name else ""
            self.logger.info(f"{prefix}{message}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PortfolioRiskManager("
            f"capital=${self.total_capital:.2f}, "
            f"exposure=${self.current_total_exposure:.2f}/"
            f"${self.max_total_exposure:.2f}, "
            f"daily_pnl=${self.daily_pnl:.2f}, "
            f"utilization={self.get_capital_utilization_percent():.1f}%)"
        )
