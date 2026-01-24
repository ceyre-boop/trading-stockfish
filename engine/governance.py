"""
Governance System - Meta-level risk control and kill-switch logic.

Acts as a global risk officer that can halt all trading if:
- Daily loss exceeds maximum threshold
- External risk events trigger
- Capital preservation required

The kill switch is irreversible within a trading session.
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class GovernanceState:
    """Snapshot of governance state."""
    max_daily_loss: float
    kill_switch_triggered: bool
    trigger_time: Optional[datetime]
    trigger_reason: Optional[str]


class Governance:
    """
    Global governance and risk control system.
    
    Responsibilities:
    - Monitor portfolio daily loss
    - Trigger kill switch if thresholds exceeded
    - Enforce trading halt across all strategies
    - Provide audit trail of governance decisions
    
    Kill switch is irreversible once triggered.
    """
    
    def __init__(
        self,
        max_daily_loss: float,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize governance system.
        
        Args:
            max_daily_loss: Maximum daily loss before triggering kill switch
            logger: Optional logger for audit trail
        """
        self.max_daily_loss = max_daily_loss
        self.kill_switch_triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason: Optional[str] = None
        self.logger = logger
        self.decision_history = []
    
    def evaluate(self, daily_loss: float) -> None:
        """
        Evaluate risk metrics and trigger kill switch if needed.
        
        Args:
            daily_loss: Current daily loss (should be negative for actual losses)
        
        Side effects:
            - May set kill_switch_triggered = True (irreversible)
            - Logs governance decision
        """
        # Note: daily_loss is negative for losses, positive for gains
        # So -daily_loss gives us the absolute loss amount
        absolute_loss = -daily_loss if daily_loss < 0 else 0.0
        
        decision = {
            'timestamp': datetime.now(),
            'daily_loss': daily_loss,
            'absolute_loss': absolute_loss,
            'max_allowed': self.max_daily_loss,
            'kill_switch_triggered_before': self.kill_switch_triggered,
            'action': 'NONE'
        }
        
        # Check if loss threshold exceeded
        if absolute_loss > self.max_daily_loss and not self.kill_switch_triggered:
            self.kill_switch_triggered = True
            self.trigger_time = datetime.now()
            self.trigger_reason = f"Daily loss ${absolute_loss:.2f} exceeds ${self.max_daily_loss:.2f}"
            decision['action'] = 'KILL_SWITCH_ACTIVATED'
            
            if self.logger:
                self.logger.error(
                    f"[KillSwitch] ACTIVATED: {self.trigger_reason}"
                )
        
        decision['kill_switch_triggered_after'] = self.kill_switch_triggered
        self.decision_history.append(decision)
    
    def can_trade(self) -> bool:
        """
        Check if trading is permitted.
        
        Returns:
            False if kill switch triggered, True otherwise
        
        Side effects: None
        """
        return not self.kill_switch_triggered
    
    def force_flatten(self) -> bool:
        """
        Check if all positions should be forcefully closed.
        
        Returns:
            True if kill switch triggered (force all exits)
        
        Side effects: None
        """
        return self.kill_switch_triggered
    
    def get_state(self) -> GovernanceState:
        """
        Get immutable snapshot of governance state.
        
        Returns:
            GovernanceState with all governance metrics
        
        Side effects: None
        """
        return GovernanceState(
            max_daily_loss=self.max_daily_loss,
            kill_switch_triggered=self.kill_switch_triggered,
            trigger_time=self.trigger_time,
            trigger_reason=self.trigger_reason
        )
    
    def reset_session(self) -> None:
        """
        Reset governance for new session.
        
        Note: Kill switch in current session cannot be reset.
        This is for new trading sessions only.
        
        Side effects:
            - Clears decision history
            - Preserves max_daily_loss
            - Kill switch state persists if currently active
        """
        # Kill switch cannot be reset mid-session
        if self.kill_switch_triggered and self.trigger_time:
            session_duration = datetime.now() - self.trigger_time
            if self.logger:
                self.logger.info(
                    f"[GovernanceReset] Kill switch active for "
                    f"{session_duration.total_seconds():.1f}s, "
                    f"reason: {self.trigger_reason}"
                )
        
        self.decision_history.clear()
    
    def override_action(
        self,
        action: str,
        symbol: str,
        reason: str
    ) -> str:
        """
        Override trading action if governance constraints violated.
        
        Args:
            action: Requested action (ENTER, ADD, REDUCE, EXIT, REVERSE)
            symbol: Trading symbol
            reason: Reason for the action
        
        Returns:
            Original action if allowed, overridden action otherwise
        
        Side effects:
            - Logs governance override decision
        """
        if not self.can_trade():
            override = "EXIT" if action != "EXIT" else "DO_NOTHING"
            
            if self.logger:
                self.logger.warning(
                    f"[ActionOverride] {symbol}: {action} → {override} "
                    f"(kill switch active)"
                )
            
            return override
        
        return action
    
    def get_report(self) -> dict:
        """
        Get comprehensive governance report.
        
        Returns:
            Dict with governance metrics and decision history
        
        Side effects: None
        """
        return {
            'max_daily_loss': self.max_daily_loss,
            'kill_switch_triggered': self.kill_switch_triggered,
            'trigger_time': self.trigger_time,
            'trigger_reason': self.trigger_reason,
            'decisions_made': len(self.decision_history),
            'is_trading_halted': not self.can_trade(),
            'decision_history': self.decision_history[-10:]  # Last 10 decisions
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "HALTED" if self.kill_switch_triggered else "ACTIVE"
        return (
            f"Governance("
            f"status={status}, "
            f"max_loss=${self.max_daily_loss:.2f}, "
            f"triggered={self.kill_switch_triggered})"
        )
