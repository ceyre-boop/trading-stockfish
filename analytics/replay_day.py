"""
Replay Day Tool - Step-by-Step Candle Analysis for Trading Stockfish

Implements a ReplayEngine that steps through historical data candle-by-candle,
displaying all engine state, decision reasoning, and outcomes. Used for:
  - Deep understanding of engine decisions
  - Debugging trading logic
  - Educational walkthrough of market events
  - Validating causal logic

Features:
  - Step through single candles or run entire days/periods
  - Full state inspection at each candle
  - All causal factors displayed (8 factors + macro + sentiment)
  - Decision reasoning with confidence levels
  - Execution details (fills, slippage, costs)
  - Position state and P&L tracking
  - Health monitor status and risk multiplier
  - Governance status (kill switches)
  - Full logging to logs/replay/

Author: Trading-Stockfish Analytics
Version: 1.0.0
License: MIT
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES FOR REPLAY STATE
# =============================================================================

class ReplayStatus(Enum):
    """Replay engine status."""
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class ReplaySnapshot:
    """Complete state snapshot at a single candle."""
    candle_index: int
    timestamp: datetime
    
    # OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Market State (8 causal factors)
    market_state: Dict[str, Any] = field(default_factory=dict)
    
    # Causal Evaluator Output
    eval_score: float = 0.0
    eval_confidence: float = 0.0
    subsystem_scores: Dict[str, float] = field(default_factory=dict)
    
    # Policy Engine Decision
    policy_action: str = "DO_NOTHING"
    target_size: float = 0.0
    action_reasoning: str = ""
    
    # Execution Details
    fill_price: Optional[float] = None
    filled_size: Optional[float] = None
    transaction_cost: float = 0.0
    slippage: float = 0.0
    
    # Position State
    position_side: str = "FLAT"
    position_size: float = 0.0
    entry_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_adverse_excursion: float = 0.0
    
    # Health & Governance
    regime_label: str = "neutral"
    health_status: str = "HEALTHY"
    risk_multiplier: float = 1.0
    governance_kill_switch: bool = False
    
    # Daily Totals
    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0


@dataclass
class ReplaySession:
    """Complete replay session metadata."""
    symbol: str
    start_date: datetime
    end_date: datetime
    config_hash: str
    snapshots: List[ReplaySnapshot] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_snapshot(self, snapshot: ReplaySnapshot) -> None:
        """Add a snapshot to the session."""
        self.snapshots.append(snapshot)
    
    def compute_stats(self) -> None:
        """Compute session statistics from snapshots."""
        if not self.snapshots:
            self.stats = {"total_candles": 0}
            return
        
        realized_pnls = [s.realized_pnl for s in self.snapshots]
        daily_pnls = [s.daily_pnl for s in self.snapshots]
        
        self.stats = {
            "total_candles": len(self.snapshots),
            "start_price": self.snapshots[0].close,
            "end_price": self.snapshots[-1].close,
            "price_change": self.snapshots[-1].close - self.snapshots[0].close,
            "final_pnl": self.snapshots[-1].cumulative_pnl if self.snapshots else 0.0,
            "total_realized_pnl": sum(realized_pnls),
            "average_daily_pnl": np.mean(daily_pnls) if daily_pnls else 0.0,
            "max_drawdown": min([s.cumulative_pnl for s in self.snapshots]) if self.snapshots else 0.0,
        }


# =============================================================================
# REPLAY ENGINE
# =============================================================================

class ReplayEngine:
    """
    Step through historical data candle-by-candle with full state inspection.
    
    Provides:
      - step(): Advance one candle, return snapshot
      - run_full(): Run entire period, collect all snapshots
      - reset(): Reset internal state
      - export_log(): Write detailed log to file
      - export_json(): Write snapshots as JSON
    
    Attributes:
        symbol: Trading symbol (e.g., 'EURUSD')
        data: DataFrame with OHLCV data
        config: Engine configuration (evaluator, policy, execution, etc.)
        status: Current replay status
        current_index: Current candle index
        session: Replay session with all snapshots
    """
    
    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize ReplayEngine.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with OHLCV data (must have columns: open, high, low, close, volume)
            config: Optional configuration dict (evaluator weights, policy thresholds, etc.)
            verbose: Enable detailed logging
        """
        self.symbol = symbol
        self.data = data.copy()
        self.config = config or {}
        self.verbose = verbose
        
        # Validate data
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Reset index to ensure it's 0-based
        self.data = self.data.reset_index(drop=True)
        
        # State
        self.status = ReplayStatus.READY
        self.current_index = 0
        self.session = ReplaySession(
            symbol=symbol,
            start_date=datetime.now(),
            end_date=datetime.now(),
            config_hash=self._compute_config_hash()
        )
        
        # Position tracking
        self.position_side = "FLAT"
        self.position_size = 0.0
        self.entry_price = None
        self.cumulative_pnl = 0.0
        self.realized_pnl = 0.0
        self.daily_pnl = 0.0
        self.peak_cumulative_pnl = 0.0
        
        # Health and governance state
        self.health_monitor_state = {
            "health_status": "HEALTHY",
            "risk_multiplier": 1.0,
        }
        self.governance_state = {
            "kill_switch_active": False,
            "reason": "",
        }
        
        # Setup logging
        self._setup_logging()
        
        logger.info(
            f"ReplayEngine initialized: symbol={symbol}, "
            f"data_length={len(self.data)}, config_hash={self.session.config_hash}"
        )
    
    def _setup_logging(self) -> None:
        """Setup file logging for replay."""
        logs_dir = Path('logs/replay')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f'replay_{self.symbol}_{timestamp}.log'
        
        # Add file handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        self.log_file = log_file
        logger.info(f"Replay logging to: {log_file}")
    
    def _compute_config_hash(self) -> str:
        """Compute hash of config for session identification."""
        import hashlib
        config_str = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def step(self) -> Optional[ReplaySnapshot]:
        """
        Advance one candle and return state snapshot.
        
        Returns:
            ReplaySnapshot for current candle, or None if at end of data
        """
        if self.current_index >= len(self.data):
            self.status = ReplayStatus.STOPPED
            logger.info("Replay reached end of data")
            return None
        
        if self.status == ReplayStatus.READY:
            self.status = ReplayStatus.RUNNING
        
        # Get current row
        row = self.data.iloc[self.current_index]
        
        # Create snapshot
        snapshot = ReplaySnapshot(
            candle_index=self.current_index,
            timestamp=datetime.now(),  # Would be actual candle time in production
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume']),
        )
        
        # Add market state (simplified version - in production would call MarketStateBuilder)
        snapshot.market_state = self._build_market_state(self.current_index)
        
        # Add causal evaluator output (simplified version)
        eval_result = self._evaluate_causal(snapshot.market_state, self.current_index)
        snapshot.eval_score = eval_result['score']
        snapshot.eval_confidence = eval_result['confidence']
        snapshot.subsystem_scores = eval_result['subsystem_scores']
        
        # Add policy decision (simplified version)
        policy_result = self._apply_policy(eval_result, snapshot)
        snapshot.policy_action = policy_result['action']
        snapshot.target_size = policy_result['target_size']
        snapshot.action_reasoning = policy_result['reasoning']
        
        # Apply execution simulator (simplified version)
        execution_result = self._simulate_execution(policy_result, snapshot)
        snapshot.fill_price = execution_result['fill_price']
        snapshot.filled_size = execution_result['filled_size']
        snapshot.transaction_cost = execution_result['transaction_cost']
        snapshot.slippage = execution_result['slippage']
        
        # Update position state
        self._update_position(execution_result, snapshot)
        
        # Add position state to snapshot
        snapshot.position_side = self.position_side
        snapshot.position_size = self.position_size
        snapshot.entry_price = self.entry_price
        snapshot.unrealized_pnl = self._calculate_unrealized_pnl(snapshot)
        snapshot.realized_pnl = self.realized_pnl
        
        # Add health and governance
        snapshot.regime_label = self.health_monitor_state.get('regime', 'neutral')
        snapshot.health_status = self.health_monitor_state['health_status']
        snapshot.risk_multiplier = self.health_monitor_state['risk_multiplier']
        snapshot.governance_kill_switch = self.governance_state['kill_switch_active']
        
        # Add P&L
        snapshot.daily_pnl = self.daily_pnl
        snapshot.cumulative_pnl = self.cumulative_pnl
        
        # Add to session
        self.session.add_snapshot(snapshot)
        
        # Log if verbose
        if self.verbose:
            self._log_snapshot(snapshot)
        
        # Advance
        self.current_index += 1
        
        return snapshot
    
    def run_full(self) -> List[ReplaySnapshot]:
        """
        Run entire replay and return all snapshots.
        
        Returns:
            List of all ReplaySnapshot objects
        """
        logger.info(f"Starting full replay of {len(self.data)} candles")
        
        snapshots = []
        while self.current_index < len(self.data):
            snapshot = self.step()
            if snapshot:
                snapshots.append(snapshot)
        
        self.status = ReplayStatus.STOPPED
        self.session.compute_stats()
        
        logger.info(
            f"Full replay complete: {len(snapshots)} candles, "
            f"final PnL: {self.cumulative_pnl:.2f}"
        )
        
        return snapshots
    
    def reset(self) -> None:
        """Reset replay state to beginning."""
        self.status = ReplayStatus.READY
        self.current_index = 0
        self.session.snapshots = []
        
        # Reset positions
        self.position_side = "FLAT"
        self.position_size = 0.0
        self.entry_price = None
        self.cumulative_pnl = 0.0
        self.realized_pnl = 0.0
        self.daily_pnl = 0.0
        self.peak_cumulative_pnl = 0.0
        
        logger.info("Replay reset to beginning")
    
    # =========================================================================
    # INTERNAL SIMULATION METHODS (Simplified for base implementation)
    # =========================================================================
    
    def _build_market_state(self, idx: int) -> Dict[str, Any]:
        """Build market state for current candle (simplified)."""
        if idx < 20:
            return {
                "status": "insufficient_data",
                "factors": {}
            }
        
        # Get recent data
        lookback = self.data.iloc[max(0, idx-50):idx+1]
        
        # Calculate technical indicators
        sma_fast = lookback['close'].rolling(5).mean().iloc[-1]
        sma_slow = lookback['close'].rolling(20).mean().iloc[-1]
        rsi = self._calculate_rsi(lookback['close'].values, 14)
        volatility = lookback['close'].pct_change().std() * np.sqrt(252)
        
        return {
            "status": "valid",
            "factors": {
                "sma_fast": float(sma_fast) if not pd.isna(sma_fast) else 0.0,
                "sma_slow": float(sma_slow) if not pd.isna(sma_slow) else 0.0,
                "rsi": float(rsi) if not pd.isna(rsi) else 0.0,
                "volatility": float(volatility) if not pd.isna(volatility) else 0.0,
                "momentum": float((lookback['close'].iloc[-1] - lookback['close'].iloc[-5]) / lookback['close'].iloc[-5]) if len(lookback) >= 5 else 0.0,
            }
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 0.0
        
        deltas = np.diff(prices[-period-1:])
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        return 100.0 - (100.0 / (1.0 + rs)) if rs > 0 else 0.0
    
    def _evaluate_causal(self, market_state: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Evaluate market using causal logic (simplified)."""
        if market_state['status'] != 'valid':
            return {
                'score': 0.0,
                'confidence': 0.0,
                'subsystem_scores': {}
            }
        
        factors = market_state['factors']
        
        # Simple evaluation logic
        trend_score = 0.5 if factors.get('sma_fast', 0) > factors.get('sma_slow', 0) else -0.5
        rsi_score = (factors.get('rsi', 50) - 50) / 50
        momentum_score = factors.get('momentum', 0) * 10  # Scale momentum
        
        # Combine scores
        eval_score = (trend_score * 0.5 + rsi_score * 0.3 + momentum_score * 0.2)
        eval_score = max(-1.0, min(1.0, eval_score))  # Clamp to [-1, 1]
        
        # Confidence based on convergence
        confidence = 0.5 + abs(eval_score) * 0.5  # Higher confidence for extreme scores
        
        return {
            'score': float(eval_score),
            'confidence': float(confidence),
            'subsystem_scores': {
                'trend': float(trend_score),
                'rsi': float(rsi_score),
                'momentum': float(momentum_score),
            }
        }
    
    def _apply_policy(self, eval_result: Dict[str, Any], snapshot: ReplaySnapshot) -> Dict[str, Any]:
        """Apply policy engine logic (simplified)."""
        score = eval_result['score']
        confidence = eval_result['confidence']
        
        # Determine action based on score
        if abs(score) < 0.2:
            action = "DO_NOTHING"
            target_size = 0.0
        elif abs(score) < 0.5:
            action = "ENTER_SMALL" if score > 0 else "DO_NOTHING"
            target_size = 0.5 if score > 0 else 0.0
        else:
            action = "ENTER_FULL" if score > 0 else "EXIT"
            target_size = 1.0 if score > 0 else 0.0
        
        reasoning = f"Score={score:.3f}, Confidence={confidence:.3f}, Action={action}"
        
        return {
            'action': action,
            'target_size': float(target_size),
            'reasoning': reasoning
        }
    
    def _simulate_execution(self, policy_result: Dict[str, Any], snapshot: ReplaySnapshot) -> Dict[str, Any]:
        """Simulate trade execution (simplified)."""
        action = policy_result['action']
        mid_price = snapshot.close
        
        # Simple spread model
        spread = mid_price * 0.001  # 0.1% spread
        
        if action == "DO_NOTHING" or action == "HOLD":
            return {
                'fill_price': None,
                'filled_size': 0.0,
                'transaction_cost': 0.0,
                'slippage': 0.0,
            }
        
        # Fill price with slippage
        if action in ["ENTER_FULL", "ENTER_SMALL"]:
            fill_price = mid_price + spread / 2
            slippage = spread / 2
        else:  # EXIT
            fill_price = mid_price - spread / 2
            slippage = -spread / 2
        
        filled_size = policy_result['target_size']
        transaction_cost = abs(filled_size * mid_price) * 0.0005  # 0.05% commission
        
        return {
            'fill_price': float(fill_price),
            'filled_size': float(filled_size),
            'transaction_cost': float(transaction_cost),
            'slippage': float(slippage),
        }
    
    def _update_position(self, execution_result: Dict[str, Any], snapshot: ReplaySnapshot) -> None:
        """Update position state based on execution."""
        if execution_result['filled_size'] == 0.0:
            return
        
        fill_price = execution_result['fill_price']
        filled_size = execution_result['filled_size']
        
        if filled_size > 0 and self.position_side == "FLAT":
            self.position_side = "LONG"
            self.position_size = filled_size
            self.entry_price = fill_price
        elif filled_size == 0 and self.position_side != "FLAT":
            # Close position
            if self.position_size > 0:
                exit_pnl = self.position_size * (fill_price - self.entry_price)
                self.realized_pnl += exit_pnl
                self.cumulative_pnl += exit_pnl
            self.position_side = "FLAT"
            self.position_size = 0.0
            self.entry_price = None
    
    def _calculate_unrealized_pnl(self, snapshot: ReplaySnapshot) -> float:
        """Calculate unrealized P&L."""
        if self.position_size == 0 or self.entry_price is None:
            return 0.0
        
        return self.position_size * (snapshot.close - self.entry_price)
    
    def _log_snapshot(self, snapshot: ReplaySnapshot) -> None:
        """Log snapshot details."""
        log_msg = (
            f"[{snapshot.candle_index:4d}] "
            f"C={snapshot.close:10.4f} "
            f"Eval={snapshot.eval_score:6.3f} "
            f"Action={snapshot.policy_action:12s} "
            f"Pos={snapshot.position_side:5s}({snapshot.position_size:.1f}) "
            f"PnL={snapshot.cumulative_pnl:10.2f} "
            f"Health={snapshot.health_status:8s}"
        )
        logger.info(log_msg)
    
    def export_log(self) -> Path:
        """Export detailed human-readable log."""
        logs_dir = Path('logs/replay')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f'replay_detailed_{self.symbol}_{timestamp}.log'
        
        with open(log_file, 'w') as f:
            f.write(f"{'='*120}\n")
            f.write(f"REPLAY LOG: {self.symbol}\n")
            f.write(f"{'='*120}\n")
            f.write(f"Start: {self.session.start_date}\n")
            f.write(f"End: {self.session.end_date}\n")
            f.write(f"Total Candles: {len(self.session.snapshots)}\n")
            f.write(f"Config Hash: {self.session.config_hash}\n")
            f.write(f"\nSession Stats:\n{json.dumps(self.session.stats, indent=2)}\n")
            f.write(f"\n{'='*120}\n")
            f.write(f"CANDLE-BY-CANDLE DETAILS\n")
            f.write(f"{'='*120}\n\n")
            
            for snapshot in self.session.snapshots:
                f.write(self._format_snapshot(snapshot))
        
        logger.info(f"Exported detailed log to: {log_file}")
        return log_file
    
    def _format_snapshot(self, snapshot: ReplaySnapshot) -> str:
        """Format snapshot for logging."""
        output = f"""
Candle {snapshot.candle_index}: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
  Price: O={snapshot.open:.4f} H={snapshot.high:.4f} L={snapshot.low:.4f} C={snapshot.close:.4f} V={snapshot.volume}
  
  Market State: {json.dumps(snapshot.market_state, indent=4)}
  
  Evaluation:
    Score: {snapshot.eval_score:.3f} (Confidence: {snapshot.eval_confidence:.3f})
    Subsystems: {json.dumps(snapshot.subsystem_scores, indent=6)}
  
  Policy Decision:
    Action: {snapshot.policy_action}
    Target Size: {snapshot.target_size:.2f}
    Reasoning: {snapshot.action_reasoning}
  
  Execution:
    Fill Price: {snapshot.fill_price or 'N/A'}
    Filled Size: {snapshot.filled_size or 'N/A'}
    Transaction Cost: {snapshot.transaction_cost:.6f}
    Slippage: {snapshot.slippage:.6f}
  
  Position:
    Side: {snapshot.position_side}
    Size: {snapshot.position_size:.2f}
    Entry: {snapshot.entry_price or 'N/A'}
    Unrealized PnL: {snapshot.unrealized_pnl:.2f}
    Realized PnL: {snapshot.realized_pnl:.2f}
  
  Health & Governance:
    Regime: {snapshot.regime_label}
    Health: {snapshot.health_status}
    Risk Multiplier: {snapshot.risk_multiplier:.2f}
    Kill Switch: {snapshot.governance_kill_switch}
  
  P&L:
    Daily: {snapshot.daily_pnl:.2f}
    Cumulative: {snapshot.cumulative_pnl:.2f}

"""
        return output
    
    def export_json(self) -> Path:
        """Export snapshots as JSON."""
        logs_dir = Path('logs/replay')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = logs_dir / f'replay_snapshots_{self.symbol}_{timestamp}.json'
        
        # Convert snapshots to dictionaries
        snapshots_dict = [asdict(s) for s in self.session.snapshots]
        
        output = {
            "symbol": self.symbol,
            "config_hash": self.session.config_hash,
            "stats": self.session.stats,
            "snapshots": snapshots_dict
        }
        
        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Exported JSON snapshots to: {json_file}")
        return json_file
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        self.session.compute_stats()
        return self.session.stats
