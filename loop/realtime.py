"""
Realtime trading orchestration loop.

Coordinates all core modules (live_feed, state_builder, evaluator, orders)
to execute trading decisions in real-time with comprehensive logging and error handling.

Main entry point for the trading engine. Runs main loop that:
1. Fetches live market data
2. Builds complete market state
3. Evaluates trading opportunities
4. Executes trades (if not in demo mode)
5. Logs all decisions and results

Usage:
    python loop/realtime.py                 # Run in DEMO mode (default)
    python loop/realtime.py --live          # Run in LIVE trading mode
    python loop/realtime.py --symbol GBPUSD # Trade different symbol
    python loop/realtime.py --interval 0.5  # Faster loop (0.5s instead of 1s)

Configuration:
    SYMBOL: Trading pair (default: EURUSD)
    LOOP_INTERVAL: Seconds between iterations (default: 1.0)
    DEMO_MODE: True to log decisions without trading (default: True)
    MAX_POSITION_SIZE: Maximum lot size per trade (default: 1.0)
"""

import sys
import time
import logging
import signal
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import core modules
from state.state_builder import build_state
from engine.evaluator import evaluate, EvaluatorConfig, Decision
from mt5.live_feed import MT5LiveFeed
from mt5.orders import MT5Orders, OrderAction

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class Config:
    """Trading engine configuration"""
    SYMBOL: str = "EURUSD"
    LOOP_INTERVAL: float = 1.0  # seconds
    DEMO_MODE: bool = True
    MAX_POSITION_SIZE: float = 1.0
    LOG_LEVEL: str = "INFO"
    ENABLE_LOGGING_FILE: bool = True
    LOG_DIR: str = "logs"


# Default configuration (can be overridden by config/settings.py if it exists)
CONFIG = Config()

# Try to import configuration from config/settings.py if available
try:
    from config.settings import CONFIG as IMPORTED_CONFIG
    CONFIG = IMPORTED_CONFIG
except ImportError:
    logging.debug("config/settings.py not found, using default configuration")

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Configure logging with file rotation and console output"""
    logger = logging.getLogger('trading_engine')
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if config.ENABLE_LOGGING_FILE:
        log_dir = Path(config.LOG_DIR)
        log_dir.mkdir(exist_ok=True)
        
        # Daily rotating log file
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f'trading_engine_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


logger = setup_logging(CONFIG)


# ============================================================================
# POSITION TRACKING
# ============================================================================

@dataclass
class PositionState:
    """Track current open position"""
    position_id: Optional[int] = None
    symbol: Optional[str] = None
    direction: Optional[str] = None  # "buy" or "sell"
    entry_price: float = 0.0
    volume: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    entry_time: Optional[float] = None
    
    def is_open(self) -> bool:
        """Check if position is currently open"""
        return self.position_id is not None
    
    def close(self):
        """Close the position"""
        self.position_id = None
        self.symbol = None
        self.direction = None
        self.entry_time = None


# ============================================================================
# MAIN ORCHESTRATION ENGINE
# ============================================================================

class RealtimeEngine:
    """Main trading engine that coordinates all modules"""
    
    def __init__(self, config: Config, demo_mode: bool = True):
        """
        Initialize the trading engine.
        
        Args:
            config: Configuration object with trading parameters
            demo_mode: If True, log decisions without executing trades
        """
        self.config = config
        self.demo_mode = demo_mode
        self.running = False
        self.iterations = 0
        self.errors = 0
        
        logger.info(f"Initializing RealtimeEngine - DEMO_MODE={self.demo_mode}")
        logger.info(f"Symbol: {config.SYMBOL}, Interval: {config.LOOP_INTERVAL}s")
        
        # Initialize core modules
        self.feed = MT5LiveFeed()
        self.orders = MT5Orders() if not demo_mode else None
        
        # Position tracking
        self.position = PositionState()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C or SIGTERM"""
        logger.info("Shutdown signal received, closing engine...")
        self.running = False
    
    def connect(self) -> bool:
        """
        Establish connection to MT5 (or mock mode if terminal not available).
        
        Returns:
            True if connection successful, False if falling back to demo mode
        """
        logger.info("Attempting to connect to MetaTrader5...")
        
        try:
            if self.feed.connect():
                logger.info("✓ Connected to MetaTrader5 successfully")
                return True
            else:
                logger.warning("✗ Failed to connect to MetaTrader5")
                logger.info("Switching to DEMO mode (mock data)")
                self.demo_mode = True
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            logger.info("Switching to DEMO mode (mock data)")
            self.demo_mode = True
            return False
    
    def _fetch_market_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch current market data for the trading symbol.
        
        Returns:
            Dictionary with tick data, or None if fetch fails
        """
        try:
            tick = self.feed.get_tick(self.config.SYMBOL)
            if tick is None:
                logger.warning(f"Failed to fetch tick data for {self.config.SYMBOL}")
                return None
            
            return {
                'symbol': self.config.SYMBOL,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread_pips': (tick.ask - tick.bid) * 10000,  # Assuming 4 decimals
                'timestamp': tick.timestamp,
            }
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def _build_state(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build complete market state using state_builder.
        
        Args:
            market_data: Current market data from live_feed
            
        Returns:
            Complete state dictionary, or None if build fails
        """
        try:
            state = build_state(self.config.SYMBOL, use_demo=self.demo_mode)
            
            if state is None:
                logger.warning("Failed to build market state")
                return None
            
            # Add real-time market data to state
            state['market_data'] = market_data
            state['timestamp'] = time.time()
            
            return state
        except Exception as e:
            logger.error(f"Error building market state: {e}")
            return None
    
    def _evaluate_opportunity(self, state: Dict[str, Any]) -> Optional[Decision]:
        """
        Evaluate trading opportunity using the evaluator.
        
        Args:
            state: Complete market state
            
        Returns:
            Decision enum (BUY, SELL, CLOSE, HOLD) or None if evaluation fails
        """
        try:
            # Get open position for evaluation
            position_info = None
            if self.position.is_open():
                position_info = {
                    'id': self.position.position_id,
                    'direction': self.position.direction,
                    'entry_price': self.position.entry_price,
                    'volume': self.position.volume,
                }
            
            # Evaluate - evaluate() returns a dict with action, confidence, reasoning
            result = evaluate(state, position_info)
            
            if result is None:
                logger.debug("Evaluator returned None (no clear decision)")
                return Decision.HOLD
            
            decision = result.get('action', 'hold')
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', '')
            
            logger.info(
                f"Decision: {decision} | "
                f"Confidence: {confidence:.2f} | "
                f"Reasoning: {reasoning}"
            )
            
            # Convert string decision to Decision enum
            try:
                return Decision[decision.upper()]
            except KeyError:
                logger.warning(f"Unknown decision type: {decision}")
                return Decision.HOLD
        
        except Exception as e:
            logger.error(f"Error evaluating opportunity: {e}")
            return None
    
    def _execute_decision(self, decision: Decision, state: Dict[str, Any]) -> bool:
        """
        Execute trading decision.
        
        Args:
            decision: Trading decision (BUY, SELL, CLOSE, HOLD)
            state: Current market state (for entry/exit levels)
            
        Returns:
            True if execution successful (or logged in demo mode)
        """
        if decision == Decision.HOLD:
            logger.debug("Decision: HOLD - no action taken")
            return True
        
        if self.demo_mode:
            logger.info(f"[DEMO MODE] Decision {decision} logged but NOT executed")
            return True
        
        if self.orders is None:
            logger.warning("Orders module not initialized, skipping execution")
            return False
        
        try:
            if decision == Decision.BUY:
                return self._execute_buy(state)
            elif decision == Decision.SELL:
                return self._execute_sell(state)
            elif decision == Decision.CLOSE:
                return self._execute_close(state)
            else:
                logger.warning(f"Unknown decision type: {decision}")
                return False
        
        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            self.errors += 1
            return False
    
    def _execute_buy(self, state: Dict[str, Any]) -> bool:
        """Execute buy order"""
        try:
            # Calculate entry levels from state
            support = state.get('support_level', state['market_data']['bid'] - 0.005)
            resistance = state.get('resistance_level', state['market_data']['bid'] + 0.010)
            
            volume = self.config.MAX_POSITION_SIZE
            
            result = self.orders.buy(
                symbol=self.config.SYMBOL,
                volume=volume,
                stop_loss=support,
                take_profit=resistance,
            )
            
            if result.success:
                logger.info(f"✓ BUY order executed: Ticket={result.order_id}, Price={result.price}")
                self.position.position_id = result.order_id
                self.position.symbol = self.config.SYMBOL
                self.position.direction = "buy"
                self.position.entry_price = result.price
                self.position.volume = volume
                self.position.stop_loss = support
                self.position.take_profit = resistance
                self.position.entry_time = time.time()
                return True
            else:
                logger.error(f"✗ BUY order failed: {result.error_message}")
                return False
        
        except Exception as e:
            logger.error(f"Exception in buy execution: {e}")
            return False
    
    def _execute_sell(self, state: Dict[str, Any]) -> bool:
        """Execute sell order"""
        try:
            # Calculate entry levels from state
            support = state.get('support_level', state['market_data']['ask'] - 0.010)
            resistance = state.get('resistance_level', state['market_data']['ask'] + 0.005)
            
            volume = self.config.MAX_POSITION_SIZE
            
            result = self.orders.sell(
                symbol=self.config.SYMBOL,
                volume=volume,
                stop_loss=resistance,
                take_profit=support,
            )
            
            if result.success:
                logger.info(f"✓ SELL order executed: Ticket={result.order_id}, Price={result.price}")
                self.position.position_id = result.order_id
                self.position.symbol = self.config.SYMBOL
                self.position.direction = "sell"
                self.position.entry_price = result.price
                self.position.volume = volume
                self.position.stop_loss = resistance
                self.position.take_profit = support
                self.position.entry_time = time.time()
                return True
            else:
                logger.error(f"✗ SELL order failed: {result.error_message}")
                return False
        
        except Exception as e:
            logger.error(f"Exception in sell execution: {e}")
            return False
    
    def _execute_close(self, state: Dict[str, Any]) -> bool:
        """Close open position"""
        if not self.position.is_open():
            logger.debug("No open position to close")
            return True
        
        try:
            result = self.orders.close_position(self.position.position_id)
            
            if result.success:
                logger.info(f"✓ Position closed: Ticket={self.position.position_id}, Exit Price={result.price}")
                exit_price = result.price
                pnl = self._calculate_pnl(exit_price)
                logger.info(f"Trade P&L: {pnl:+.5f} ({self._calculate_pnl_percent(exit_price):+.2f}%)")
                self.position.close()
                return True
            else:
                logger.error(f"✗ Close position failed: {result.error_message}")
                return False
        
        except Exception as e:
            logger.error(f"Exception in close execution: {e}")
            return False
    
    def _calculate_pnl(self, exit_price: float) -> float:
        """Calculate profit/loss in pips"""
        if not self.position.is_open():
            return 0.0
        
        diff = exit_price - self.position.entry_price
        if self.position.direction == "sell":
            diff = -diff
        
        return diff * 10000  # Assuming 4 decimal places
    
    def _calculate_pnl_percent(self, exit_price: float) -> float:
        """Calculate profit/loss as percentage"""
        if not self.position.is_open() or self.position.entry_price == 0:
            return 0.0
        
        diff = exit_price - self.position.entry_price
        if self.position.direction == "sell":
            diff = -diff
        
        return (diff / self.position.entry_price) * 100
    
    def _log_iteration_summary(self, market_data: Dict[str, Any], decision: Decision):
        """Log summary of current iteration"""
        logger.debug(
            f"Iteration {self.iterations}: "
            f"Bid={market_data['bid']:.5f}, "
            f"Ask={market_data['ask']:.5f}, "
            f"Spread={market_data['spread_pips']:.1f}pips, "
            f"Decision={decision}, "
            f"Position={'OPEN' if self.position.is_open() else 'CLOSED'}"
        )
    
    def run_iteration(self) -> bool:
        """
        Run a single engine iteration.
        
        Returns:
            True if iteration completed successfully
        """
        self.iterations += 1
        iteration_start = time.time()
        
        try:
            # Step 1: Fetch market data
            market_data = self._fetch_market_data()
            if market_data is None:
                logger.warning("Skipping iteration - failed to fetch market data")
                return False
            
            # Step 2: Build market state
            state = self._build_state(market_data)
            if state is None:
                logger.warning("Skipping iteration - failed to build state")
                return False
            
            # Step 3: Evaluate opportunity
            decision = self._evaluate_opportunity(state)
            if decision is None:
                logger.warning("Skipping iteration - evaluation failed")
                return False
            
            # Step 4: Execute decision (if not in demo mode)
            self._execute_decision(decision, state)
            
            # Step 5: Log iteration summary
            self._log_iteration_summary(market_data, decision)
            
            iteration_time = time.time() - iteration_start
            logger.debug(f"Iteration completed in {iteration_time*1000:.1f}ms")
            
            return True
        
        except Exception as e:
            logger.error(f"Unexpected error in iteration {self.iterations}: {e}")
            self.errors += 1
            return False
    
    def run(self):
        """
        Main trading loop.
        
        Runs continuously, fetching data and executing trades at configured interval.
        """
        logger.info("=" * 70)
        logger.info("REALTIME TRADING ENGINE STARTED")
        logger.info("=" * 70)
        logger.info(f"Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
        logger.info(f"Symbol: {self.config.SYMBOL}")
        logger.info(f"Interval: {self.config.LOOP_INTERVAL}s")
        logger.info(f"Max Position Size: {self.config.MAX_POSITION_SIZE}L")
        logger.info("=" * 70)
        
        # Connect to MT5
        self.connect()
        
        # Main loop
        self.running = True
        last_iteration = time.time()
        
        try:
            while self.running:
                # Wait for next iteration
                elapsed = time.time() - last_iteration
                sleep_time = max(0, self.config.LOOP_INTERVAL - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_iteration = time.time()
                
                # Run iteration
                self.run_iteration()
        
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
        
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
        
        finally:
            self._shutdown()
    
    def _shutdown(self):
        """Clean shutdown of the engine"""
        logger.info("=" * 70)
        logger.info("REALTIME TRADING ENGINE SHUTDOWN")
        logger.info("=" * 70)
        
        # Close open position if any
        if self.position.is_open() and not self.demo_mode and self.orders:
            logger.info("Closing open position before shutdown...")
            try:
                result = self.orders.close_position(self.position.position_id)
                if result.success:
                    logger.info(f"Position closed successfully: {self.position.position_id}")
                else:
                    logger.warning(f"Failed to close position: {result.error_message}")
            except Exception as e:
                logger.error(f"Error closing position: {e}")
        
        # Disconnect from MT5
        try:
            self.feed.disconnect()
            logger.info("Disconnected from MetaTrader5")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
        
        # Log final statistics
        logger.info(f"Engine Statistics:")
        logger.info(f"  Iterations: {self.iterations}")
        logger.info(f"  Errors: {self.errors}")
        logger.info(f"  Error Rate: {(self.errors/self.iterations*100):.2f}%" if self.iterations > 0 else "  Error Rate: N/A")
        logger.info("=" * 70)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Parse arguments and start the trading engine"""
    parser = argparse.ArgumentParser(
        description="Realtime trading engine orchestration loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python loop/realtime.py                    # Run in DEMO mode (default)
  python loop/realtime.py --live             # Run in LIVE trading mode
  python loop/realtime.py --symbol GBPUSD    # Trade GBPUSD instead of EURUSD
  python loop/realtime.py --interval 0.5     # Faster loop (0.5s per iteration)
  python loop/realtime.py --live --symbol GBPUSD --interval 2.0
        """
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        default=False,
        help='Run in LIVE mode (execute real trades). Default is DEMO mode.',
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default=CONFIG.SYMBOL,
        help=f'Trading symbol. Default: {CONFIG.SYMBOL}',
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=CONFIG.LOOP_INTERVAL,
        help=f'Loop interval in seconds. Default: {CONFIG.LOOP_INTERVAL}',
    )
    
    parser.add_argument(
        '--max-size',
        type=float,
        default=CONFIG.MAX_POSITION_SIZE,
        help=f'Maximum position size in lots. Default: {CONFIG.MAX_POSITION_SIZE}',
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=CONFIG.LOG_LEVEL,
        help=f'Logging level. Default: {CONFIG.LOG_LEVEL}',
    )
    
    args = parser.parse_args()
    
    # Build configuration from arguments
    config = Config(
        SYMBOL=args.symbol,
        LOOP_INTERVAL=args.interval,
        DEMO_MODE=not args.live,
        MAX_POSITION_SIZE=args.max_size,
        LOG_LEVEL=args.log_level,
    )
    
    # Create and run engine
    engine = RealtimeEngine(config, demo_mode=config.DEMO_MODE)
    engine.run()


if __name__ == '__main__':
    main()
