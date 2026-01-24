"""
Real-Time Monitoring Dashboard - Phase RT-3

CLI-based real-time monitoring dashboard for live trading with:
  - Live price updates
  - Order book summary
  - Engine decisions
  - PnL tracking (realized + unrealized)
  - Exposure monitoring
  - Health status
  - Governance status
  - Connector health
  - Keyboard controls (pause/resume/flatten/shutdown)
"""

import curses
import threading
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any, Callable, List
from enum import Enum
import queue


class DashboardMode(Enum):
    """Dashboard display modes."""
    SUMMARY = "SUMMARY"
    DETAILED = "DETAILED"
    ALERTS = "ALERTS"


@dataclass
class DashboardUpdate:
    """Update for dashboard display."""
    timestamp: float
    price_data: Optional[Dict[str, Any]] = None
    orderbook_data: Optional[Dict[str, Any]] = None
    engine_stats: Optional[Dict[str, Any]] = None
    pnl_data: Optional[Dict[str, Any]] = None
    exposure_data: Optional[Dict[str, Any]] = None
    health_status: Optional[Dict[str, Any]] = None
    governance_status: Optional[Dict[str, Any]] = None
    connector_status: Optional[Dict[str, Any]] = None
    alerts: Optional[List[str]] = None


class MonitoringDashboard:
    """
    Real-time CLI monitoring dashboard for live trading.
    
    Features:
    - Live display of price updates, orderbook, engine decisions
    - PnL tracking and exposure monitoring
    - Health and governance status
    - Connector health monitoring
    - Keyboard controls for pause/resume/flatten/shutdown
    - Multiple display modes (summary, detailed, alerts)
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MonitoringDashboard.
        
        Args:
            logger: Optional logger instance
            config: Optional configuration overrides
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Display configuration
        self.refresh_interval = self.config.get('refresh_interval_s', 0.5)
        self.max_alerts = self.config.get('max_alerts', 10)
        self.mode = DashboardMode.SUMMARY
        
        # Data storage
        self._latest_update: Optional[DashboardUpdate] = None
        self._alerts: List[str] = []
        self._price_history: Dict[str, List[float]] = {}
        self._update_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # Display state
        self._is_running = False
        self._display_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Callbacks for keyboard input
        self._on_pause_callback: Optional[Callable] = None
        self._on_resume_callback: Optional[Callable] = None
        self._on_flatten_callback: Optional[Callable] = None
        self._on_shutdown_callback: Optional[Callable] = None
        
        # Color pairs (will be initialized in curses)
        self.color_pairs = {}
        
        self.logger.info("MonitoringDashboard initialized")
    
    # ===== Lifecycle =====
    
    def start(self) -> bool:
        """Start the monitoring dashboard."""
        if self._is_running:
            self.logger.warning("Dashboard already running")
            return False
        
        try:
            self._is_running = True
            self._stop_event.clear()
            
            self._display_thread = threading.Thread(
                target=self._run_display,
                daemon=False,
                name="MonitoringDashboardThread"
            )
            self._display_thread.start()
            
            self.logger.info("Monitoring dashboard started")
            return True
        
        except Exception as e:
            self.logger.error("Error starting dashboard: %s", e, exc_info=True)
            self._is_running = False
            return False
    
    def stop(self) -> bool:
        """Stop the monitoring dashboard."""
        if not self._is_running:
            self.logger.warning("Dashboard not running")
            return False
        
        try:
            self._stop_event.set()
            self._is_running = False
            
            if self._display_thread and self._display_thread.is_alive():
                self._display_thread.join(timeout=5.0)
            
            self.logger.info("Monitoring dashboard stopped")
            return True
        
        except Exception as e:
            self.logger.error("Error stopping dashboard: %s", e, exc_info=True)
            return False
    
    # ===== Updates =====
    
    def post_update(self, update: DashboardUpdate) -> None:
        """Post update to dashboard."""
        self._latest_update = update
        
        try:
            self._update_queue.put_nowait(update)
        except queue.Full:
            pass  # Drop oldest update
        
        # Add alerts to alert queue
        if update.alerts:
            for alert in update.alerts:
                self._add_alert(alert)
    
    def _add_alert(self, alert: str) -> None:
        """Add alert to alert queue."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_with_time = f"[{timestamp}] {alert}"
        self._alerts.insert(0, alert_with_time)
        
        # Keep only latest alerts
        if len(self._alerts) > self.max_alerts:
            self._alerts = self._alerts[:self.max_alerts]
    
    # ===== Callbacks =====
    
    def register_pause_callback(self, callback: Callable) -> None:
        """Register callback for pause command."""
        self._on_pause_callback = callback
    
    def register_resume_callback(self, callback: Callable) -> None:
        """Register callback for resume command."""
        self._on_resume_callback = callback
    
    def register_flatten_callback(self, callback: Callable) -> None:
        """Register callback for flatten positions command."""
        self._on_flatten_callback = callback
    
    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register callback for shutdown command."""
        self._on_shutdown_callback = callback
    
    # ===== Display Loop =====
    
    def _run_display(self) -> None:
        """Run the dashboard display loop."""
        try:
            curses.wrapper(self._display_loop)
        except KeyboardInterrupt:
            self.logger.info("Dashboard interrupted by user")
        except Exception as e:
            self.logger.error("Dashboard error: %s", e, exc_info=True)
        finally:
            self._is_running = False
    
    def _display_loop(self, stdscr: Any) -> None:
        """Main display loop (runs within curses context)."""
        # Initialize curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.timeout(int(self.refresh_interval * 1000))  # Timeout in ms
        
        # Initialize colors
        self._init_colors(stdscr)
        
        # Main loop
        while not self._stop_event.is_set():
            try:
                # Process keyboard input (non-blocking)
                key = stdscr.getch()
                if key != -1:
                    self._handle_input(key)
                
                # Clear screen
                stdscr.clear()
                
                # Draw based on mode
                if self.mode == DashboardMode.SUMMARY:
                    self._draw_summary(stdscr)
                elif self.mode == DashboardMode.DETAILED:
                    self._draw_detailed(stdscr)
                elif self.mode == DashboardMode.ALERTS:
                    self._draw_alerts(stdscr)
                
                # Draw footer with help
                self._draw_footer(stdscr)
                
                # Refresh display
                stdscr.refresh()
                
            except Exception as e:
                self.logger.error("Display loop error: %s", e, exc_info=True)
                time.sleep(0.1)
    
    def _init_colors(self, stdscr: Any) -> None:
        """Initialize color pairs."""
        try:
            # Define color pairs
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Healthy
            curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
            curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)     # Error
            curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Info
            curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)   # Normal
            
            self.color_pairs = {
                'healthy': curses.color_pair(1),
                'warning': curses.color_pair(2),
                'error': curses.color_pair(3),
                'info': curses.color_pair(4),
                'normal': curses.color_pair(5)
            }
        except Exception as e:
            self.logger.warning("Color initialization failed: %s", e)
    
    def _handle_input(self, key: int) -> None:
        """Handle keyboard input."""
        if key == ord('q'):  # Quit
            if self._on_shutdown_callback:
                try:
                    self._on_shutdown_callback()
                except Exception as e:
                    self.logger.error("Shutdown callback error: %s", e)
        
        elif key == ord('p'):  # Pause
            if self._on_pause_callback:
                try:
                    self._on_pause_callback()
                except Exception as e:
                    self.logger.error("Pause callback error: %s", e)
        
        elif key == ord('r'):  # Resume
            if self._on_resume_callback:
                try:
                    self._on_resume_callback()
                except Exception as e:
                    self.logger.error("Resume callback error: %s", e)
        
        elif key == ord('f'):  # Flatten
            if self._on_flatten_callback:
                try:
                    self._on_flatten_callback()
                except Exception as e:
                    self.logger.error("Flatten callback error: %s", e)
        
        elif key == ord('s'):  # Switch mode to Summary
            self.mode = DashboardMode.SUMMARY
        
        elif key == ord('d'):  # Switch mode to Detailed
            self.mode = DashboardMode.DETAILED
        
        elif key == ord('a'):  # Switch mode to Alerts
            self.mode = DashboardMode.ALERTS
    
    def _draw_summary(self, stdscr: Any) -> None:
        """Draw summary view."""
        try:
            row = 0
            
            # Header
            self._add_line(stdscr, row, "═" * 80, 'info')
            row += 1
            self._add_line(stdscr, row, "TRADING STOCKFISH v1.0 - LIVE TRADING DASHBOARD", 'info')
            row += 1
            self._add_line(stdscr, row, f"Mode: SUMMARY | Time: {datetime.now().strftime('%H:%M:%S')}", 'normal')
            row += 1
            self._add_line(stdscr, row, "═" * 80, 'info')
            row += 1
            
            # Price data
            if self._latest_update and self._latest_update.price_data:
                prices = self._latest_update.price_data
                self._add_line(stdscr, row, "PRICE DATA", 'info')
                row += 1
                for symbol, data in prices.items():
                    price_str = f"  {symbol:8s} | Bid: {data.get('bid', 0):10.2f} Ask: {data.get('ask', 0):10.2f} Last: {data.get('last', 0):10.2f}"
                    self._add_line(stdscr, row, price_str, 'normal')
                    row += 1
                row += 1
            
            # PnL data
            if self._latest_update and self._latest_update.pnl_data:
                pnl = self._latest_update.pnl_data
                self._add_line(stdscr, row, "PnL", 'info')
                row += 1
                pnl_str = f"  Realized: {pnl.get('realized_pnl', 0):12.2f} | Unrealized: {pnl.get('unrealized_pnl', 0):12.2f} | Total: {pnl.get('total_pnl', 0):12.2f}"
                self._add_line(stdscr, row, pnl_str, 'normal')
                row += 1
                row += 1
            
            # Health status
            if self._latest_update and self._latest_update.health_status:
                health = self._latest_update.health_status
                health_color = 'healthy' if health.get('status') == 'healthy' else 'warning'
                self._add_line(stdscr, row, f"Health: {health.get('status', 'unknown').upper()}", health_color)
                row += 1
                row += 1
            
            # Connector status
            if self._latest_update and self._latest_update.connector_status:
                connectors = self._latest_update.connector_status
                self._add_line(stdscr, row, "CONNECTORS", 'info')
                row += 1
                for name, status in connectors.items():
                    status_str = "✓" if status.get('connected') else "✗"
                    connector_line = f"  {name:12s} {status_str} | Latency: {status.get('latency', 0):6.1f}ms | Errors: {status.get('errors', 0):3d}"
                    color = 'healthy' if status.get('connected') else 'error'
                    self._add_line(stdscr, row, connector_line, color)
                    row += 1
                row += 1
            
            # Alerts
            if self._alerts:
                self._add_line(stdscr, row, "RECENT ALERTS", 'warning')
                row += 1
                for alert in self._alerts[:5]:
                    self._add_line(stdscr, row, f"  {alert}", 'warning')
                    row += 1
        
        except Exception as e:
            self.logger.error("Summary draw error: %s", e, exc_info=True)
    
    def _draw_detailed(self, stdscr: Any) -> None:
        """Draw detailed view."""
        try:
            row = 0
            
            # Header
            self._add_line(stdscr, row, "═" * 80, 'info')
            row += 1
            self._add_line(stdscr, row, "TRADING STOCKFISH v1.0 - DETAILED VIEW", 'info')
            row += 1
            self._add_line(stdscr, row, f"Mode: DETAILED | Time: {datetime.now().strftime('%H:%M:%S')}", 'normal')
            row += 1
            self._add_line(stdscr, row, "═" * 80, 'info')
            row += 1
            
            # Engine stats
            if self._latest_update and self._latest_update.engine_stats:
                stats = self._latest_update.engine_stats
                self._add_line(stdscr, row, "ENGINE STATISTICS", 'info')
                row += 1
                self._add_line(stdscr, row, f"  Decisions: {stats.get('decisions_made', 0)} | Orders: {stats.get('orders_submitted', 0)} | Filled: {stats.get('orders_filled', 0)} | Rejected: {stats.get('orders_rejected', 0)}", 'normal')
                row += 1
                self._add_line(stdscr, row, f"  Latency P50: {stats.get('latency_p50', 0):.1f}ms | P99: {stats.get('latency_p99', 0):.1f}ms | Errors: {stats.get('errors', 0)}", 'normal')
                row += 1
                row += 1
            
            # Exposure
            if self._latest_update and self._latest_update.exposure_data:
                exposure = self._latest_update.exposure_data
                self._add_line(stdscr, row, "EXPOSURE", 'info')
                row += 1
                for symbol, exp in exposure.items():
                    exp_str = f"  {symbol:8s} | Qty: {exp.get('quantity', 0):8.2f} | Value: {exp.get('value', 0):12.2f} | Risk: {exp.get('risk_pct', 0):6.2f}%"
                    self._add_line(stdscr, row, exp_str, 'normal')
                    row += 1
                row += 1
            
            # Order book
            if self._latest_update and self._latest_update.orderbook_data:
                ob = self._latest_update.orderbook_data
                self._add_line(stdscr, row, "ORDER BOOK", 'info')
                row += 1
                for symbol, levels in ob.items():
                    bid_str = f"  {symbol:8s} Bids: {levels.get('bid_levels', 0)} | Asks: {levels.get('ask_levels', 0)}"
                    self._add_line(stdscr, row, bid_str, 'normal')
                    row += 1
        
        except Exception as e:
            self.logger.error("Detailed draw error: %s", e, exc_info=True)
    
    def _draw_alerts(self, stdscr: Any) -> None:
        """Draw alerts view."""
        try:
            row = 0
            
            # Header
            self._add_line(stdscr, row, "═" * 80, 'info')
            row += 1
            self._add_line(stdscr, row, "TRADING STOCKFISH v1.0 - ALERTS VIEW", 'info')
            row += 1
            self._add_line(stdscr, row, f"Mode: ALERTS | Time: {datetime.now().strftime('%H:%M:%S')}", 'normal')
            row += 1
            self._add_line(stdscr, row, "═" * 80, 'info')
            row += 1
            
            # All alerts
            self._add_line(stdscr, row, f"Total Alerts: {len(self._alerts)}", 'info')
            row += 1
            
            for alert in self._alerts:
                self._add_line(stdscr, row, alert, 'warning')
                row += 1
        
        except Exception as e:
            self.logger.error("Alerts draw error: %s", e, exc_info=True)
    
    def _draw_footer(self, stdscr: Any) -> None:
        """Draw footer with help."""
        try:
            max_y = stdscr.getmaxyx()[0]
            
            footer_text = "Help: [S]ummary [D]etailed [A]lerts [P]ause [R]esume [F]latten [Q]uit"
            stdscr.addstr(max_y - 1, 0, footer_text, self.color_pairs.get('info', 0))
        
        except Exception as e:
            self.logger.debug("Footer draw error: %s", e)
    
    def _add_line(self, stdscr: Any, row: int, text: str, color_key: str = 'normal') -> None:
        """Add a line to the display."""
        try:
            max_y, max_x = stdscr.getmaxyx()
            
            if row >= max_y - 1:
                return  # Don't draw beyond screen
            
            # Truncate if necessary
            if len(text) > max_x - 1:
                text = text[:max_x - 1]
            
            color = self.color_pairs.get(color_key, 0)
            stdscr.addstr(row, 0, text, color)
        
        except Exception as e:
            self.logger.debug("Add line error: %s", e)
    
    def is_running(self) -> bool:
        """Check if dashboard is running."""
        return self._is_running
