# MONITORING DASHBOARD GUIDE

**Phase RT-3 Component**

## Overview

The `MonitoringDashboard` is a CLI-based real-time monitoring interface for live trading sessions. It provides:

- **Live Market Data Display** - Real-time price updates, order books
- **Trading Metrics** - Engine decisions, order status, PnL
- **System Health** - Connector status, memory usage, latency
- **Alerts and Warnings** - Recent anomalies and safety events
- **Keyboard Controls** - Interactive pause/resume/flatten/shutdown

## Starting the Dashboard

### Basic Usage

```python
from realtime import MonitoringDashboard, DashboardUpdate
from realtime.live_trading_orchestrator import LiveTradingOrchestrator

# Create dashboard
dashboard = MonitoringDashboard()

# Start dashboard (launches in separate thread)
dashboard.start()

# Post updates in your main loop
update = DashboardUpdate(
    timestamp=time.time(),
    price_data={'SPY': {'bid': 450.00, 'ask': 450.50, 'last': 450.25}},
    pnl_data={'realized_pnl': 250.50, 'unrealized_pnl': 125.75}
)
dashboard.post_update(update)

# Dashboard runs until stopped
# ... main trading loop ...

# Stop dashboard
dashboard.stop()
```

### With Orchestrator Integration

```python
orchestrator = LiveTradingOrchestrator(em, el, dfr, sl)
dashboard = MonitoringDashboard()

# Register callbacks
dashboard.register_pause_callback(orchestrator.engine_loop.pause)
dashboard.register_resume_callback(orchestrator.engine_loop.resume)
dashboard.register_flatten_callback(orchestrator.engine_loop.flatten_positions)
dashboard.register_shutdown_callback(orchestrator.stop)

# Start both
dashboard.start()
orchestrator.start()

# Main loop posts updates
while orchestrator.is_running():
    update = create_dashboard_update(orchestrator)
    dashboard.post_update(update)
    time.sleep(0.5)

dashboard.stop()
```

## Display Modes

### Summary Mode (Default)

Accessed via **[S]** key or default startup.

**Layout:**
```
════════════════════════════════════════════════════════════════════════════
TRADING STOCKFISH v1.0 - LIVE TRADING DASHBOARD
Mode: SUMMARY | Time: 14:30:45
════════════════════════════════════════════════════════════════════════════

PRICE DATA
  SPY      | Bid:       450.00 Ask:       450.50 Last:       450.25
  QQQ      | Bid:       380.00 Ask:       380.50 Last:       380.15
  IWM      | Bid:       190.00 Ask:       190.50 Last:       190.25

PnL
  Realized:      250.50 | Unrealized:      125.75 | Total:      376.25

Health: HEALTHY

CONNECTORS
  ibkr         ✓ | Latency:   45.2ms | Errors:   0
  fix          ✓ | Latency:   12.5ms | Errors:   0
  zmq          ✓ | Latency:    8.1ms | Errors:   0

RECENT ALERTS
  [14:30:30] Warning: High bid-ask spread detected on SPY
  [14:29:15] Info: New position opened SPY +100 shares

Help: [S]ummary [D]etailed [A]lerts [P]ause [R]esume [F]latten [Q]uit
```

**Information Displayed:**
- Current price (bid/ask/last) for tracked symbols
- Realized and unrealized PnL
- System health status
- Connector connectivity and latency
- Recent alerts and events

**Best For:**
- Quick overview during trading
- Monitoring key metrics
- Watching market conditions

### Detailed Mode

Accessed via **[D]** key.

**Layout:**
```
════════════════════════════════════════════════════════════════════════════
TRADING STOCKFISH v1.0 - DETAILED VIEW
Mode: DETAILED | Time: 14:30:45
════════════════════════════════════════════════════════════════════════════

ENGINE STATISTICS
  Decisions: 1250 | Orders: 50 | Filled: 48 | Rejected: 2
  Latency P50: 4.5ms | P99: 15.2ms | Errors: 0

EXPOSURE
  SPY      | Qty:      100.00 | Value:      45000.00 | Risk:   15.50%
  QQQ      | Qty:       75.00 | Value:      28500.00 | Risk:   11.75%
  IWM      | Qty:      150.00 | Value:      28500.00 | Risk:   10.25%

ORDER BOOK
  SPY      Bids: 10 | Asks: 10
  QQQ      Bids: 10 | Asks: 10
  IWM      Bids:  5 | Asks:  5

Help: [S]ummary [D]etailed [A]lerts [P]ause [R]esume [F]latten [Q]uit
```

**Information Displayed:**
- Engine decision count and latency percentiles
- Current position exposure by symbol
- Order book snapshot (depth levels)
- Execution statistics

**Best For:**
- Deep analysis of engine performance
- Position and exposure review
- Order book analysis
- Performance troubleshooting

### Alerts Mode

Accessed via **[A]** key.

**Layout:**
```
════════════════════════════════════════════════════════════════════════════
TRADING STOCKFISH v1.0 - ALERTS VIEW
Mode: ALERTS | Time: 14:30:45
════════════════════════════════════════════════════════════════════════════

Total Alerts: 15

[14:30:45] Critical: Connector ibkr disconnected
[14:30:40] Warning: Feed stale for QQQ (15.2s)
[14:30:35] Warning: High bid-ask spread on SPY (0.75%)
[14:30:30] Info: New position opened SPY +100
[14:30:25] Warning: Order rejected: OrderID 12345 - Insufficient buying power
...

Help: [S]ummary [D]etailed [A]lerts [P]ause [R]esume [F]latten [Q]uit
```

**Information Displayed:**
- All recent alerts in chronological order
- Alert severity (Critical/Warning/Info)
- Alert timestamp and message
- Up to 10 most recent alerts

**Best For:**
- Reviewing recent system events
- Tracking issues
- Debugging problems
- Post-session analysis

## Keyboard Controls

### Navigation

| Key | Action | Effect |
|-----|--------|--------|
| **S** | Summary Mode | Switch to summary view |
| **D** | Detailed Mode | Switch to detailed view |
| **A** | Alerts Mode | Switch to alerts view |

### Trading Control

| Key | Action | Effect |
|-----|--------|--------|
| **P** | Pause | Pause trading engine, hold positions |
| **R** | Resume | Resume trading from paused state |
| **F** | Flatten | Close all positions immediately |
| **Q** | Quit | Shutdown trading session gracefully |

### Examples

**Pausing trading during high volatility:**
```
1. Press [P]
2. Engine pauses accepting new trades
3. Existing positions held
4. Can review situation
5. Press [R] to resume
```

**Emergency position closure:**
```
1. Press [F]
2. All positions immediately flattened
3. Market orders sent for liquidation
4. Dashboard updates with new PnL
```

**Graceful shutdown:**
```
1. Press [Q]
2. Engine stops accepting new trades
3. Existing orders completed/cancelled
4. Session terminates
5. Dashboard closes
```

## Color Coding

### Status Colors

| Color | Meaning | Examples |
|-------|---------|----------|
| 🟢 **Green** | Healthy/Good | Connected connector, no errors, positive PnL |
| 🟡 **Yellow** | Warning | High latency, degraded state, stale data |
| 🔴 **Red** | Error/Critical | Disconnected connector, failsafe active |
| 🔵 **Cyan** | Information | Headers, section titles, status info |
| ⚪ **White** | Normal | Regular data display |

### Example Color Usage

```
Health: 🟢 HEALTHY                    (Green = good)
Health: 🟡 DEGRADED                   (Yellow = warning)
Health: 🔴 FAILSAFE                   (Red = critical)

Connector ibkr     🟢 ✓                (Green = connected)
Connector fix      🔴 ✗                (Red = disconnected)

PnL realized: 🟢 +250.50              (Green = profit)
PnL realized: 🔴 -250.50              (Red = loss)
```

## Data Fields Explained

### Price Data

```
SPY | Bid: 450.00 Ask: 450.50 Last: 450.25
    |  └─ Best bid price
    |  └─ Best ask price
    |  └─ Last traded price
```

### PnL Data

```
Realized: 250.50      # Profit from closed positions
Unrealized: 125.75    # Profit from open positions
Total: 376.25         # Sum of realized + unrealized
```

### Health Status

```
HEALTHY    # All systems normal, trading active
DEGRADED   # Anomaly detected, monitoring increased
FAILSAFE   # Critical issue, positions flattening, trading disabled
```

### Connector Metrics

```
Latency: 45.2ms       # Round-trip latency to exchange
Errors: 0             # Error count in last period
Connected: ✓ / ✗      # Connection status
```

### Engine Statistics

```
Decisions: 1250       # Total decisions made
Orders: 50            # Total orders submitted
Filled: 48            # Orders filled successfully
Rejected: 2           # Orders rejected by broker

Latency P50: 4.5ms    # 50th percentile latency (median)
Latency P99: 15.2ms   # 99th percentile latency (worst case)
```

### Exposure Metrics

```
Qty: 100.00          # Quantity of open position
Value: 45000.00      # Dollar value of position
Risk: 15.50%         # Percentage of portfolio at risk
```

## Configuration

### Dashboard Settings

```python
config = {
    'refresh_interval_s': 0.5,    # Update frequency (0.5 seconds)
    'max_alerts': 10,              # Maximum alerts to display
}

dashboard = MonitoringDashboard(config=config)
```

### Update Frequency

- **0.2s** - Very responsive, high CPU (not recommended)
- **0.5s** - Recommended for active trading
- **1.0s** - Lower CPU, acceptable responsiveness
- **2.0s** - Low CPU, laggy display

## Performance Impact

### CPU Usage

- Dashboard alone: ~1-3% CPU
- With color rendering: +0.5% CPU
- With high update frequency: can increase to 10%+

### Memory Usage

- Base: ~5 MB
- Per 100 symbols: +2 MB
- Per 1000 alerts in history: +1 MB

### Latency Impact

- Posting updates: <0.1ms
- Rendering display: 10-50ms (depends on terminal speed)
- Keyboard input: <1ms

## Troubleshooting

### Dashboard Won't Start

**Issue:** `start()` returns False
**Solution:**
1. Check logs in `logs/live/live_trading_*.log`
2. Verify terminal is large enough (minimum 80x24)
3. Check for curses library availability
4. Run: `python -c "import curses"`

### Display Corruption or Flickering

**Issue:** Dashboard text is garbled or flickers
**Solution:**
1. Increase refresh interval: `'refresh_interval_s': 1.0`
2. Check terminal compatibility
3. Update terminal emulator
4. Increase terminal size

### Keyboard Commands Don't Work

**Issue:** Pressing keys has no effect
**Solution:**
1. Dashboard must be running (check `is_running()`)
2. Dashboard window must be in focus
3. Register callbacks properly: `register_*_callback()`
4. Check that callbacks are not throwing exceptions

### High CPU Usage

**Issue:** Dashboard consuming excessive CPU
**Solution:**
1. Increase refresh interval: `'refresh_interval_s': 2.0`
2. Reduce displayed symbols
3. Disable color rendering if available
4. Update terminal emulator driver

### Data Not Updating

**Issue:** Dashboard shows stale data
**Solution:**
1. Check `post_update()` calls are happening
2. Verify `DashboardUpdate` objects have all fields
3. Check main loop is not blocked
4. Verify dashboard is in LIVE state

## Integration Examples

### Basic Integration

```python
from realtime import MonitoringDashboard, DashboardUpdate

dashboard = MonitoringDashboard()
dashboard.start()

while True:
    update = DashboardUpdate(
        timestamp=time.time(),
        price_data=get_current_prices(),
        pnl_data=get_pnl()
    )
    dashboard.post_update(update)
    time.sleep(0.5)

dashboard.stop()
```

### With Orchestrator

```python
orchestrator = LiveTradingOrchestrator(em, el, dfr, sl)
dashboard = MonitoringDashboard()

# Connect controls
dashboard.register_pause_callback(
    lambda: orchestrator.engine_loop.pause()
)
dashboard.register_flatten_callback(
    lambda: orchestrator.engine_loop.flatten_positions()
)

orchestrator.start()
dashboard.start()

while orchestrator.is_running():
    update = DashboardUpdate(
        timestamp=time.time(),
        price_data=orchestrator.data_feed_router.get_latest_prices(),
        engine_stats=orchestrator.engine_loop.get_stats(),
        pnl_data=orchestrator.engine_loop.get_pnl(),
        health_status=orchestrator.get_stats(),
        connector_status=orchestrator.get_connector_status(),
        alerts=get_recent_alerts()
    )
    dashboard.post_update(update)
    time.sleep(0.5)

dashboard.stop()
orchestrator.stop()
```

### With Safety Events

```python
def on_safety_event(event):
    alert = f"[{event.event_type.value}] {event.message}"
    dashboard._add_alert(alert)
    
    if event.severity == "critical":
        dashboard._add_alert("🔴 CRITICAL EVENT - Immediate action needed")

orchestrator.register_safety_event_callback(on_safety_event)
```

## Best Practices

1. **Always check `is_running()` before posting updates:**
   ```python
   if dashboard.is_running():
       dashboard.post_update(update)
   ```

2. **Post updates at reasonable intervals:**
   ```python
   # Not too fast (high CPU)
   # Not too slow (stale data)
   # Recommended: 0.5 - 1.0 seconds
   ```

3. **Include all relevant data:**
   ```python
   update = DashboardUpdate(
       price_data=prices,
       pnl_data=pnl,
       engine_stats=stats,
       health_status=health,
       connector_status=connectors,
       alerts=recent_alerts
   )
   ```

4. **Register callbacks before starting:**
   ```python
   dashboard.register_pause_callback(...)
   dashboard.register_flatten_callback(...)
   dashboard.start()  # After all callbacks
   ```

5. **Stop gracefully:**
   ```python
   if dashboard.is_running():
       dashboard.stop()  # Always call, never kill
   ```

## Terminal Requirements

### Minimum Size

- Width: 80 characters
- Height: 24 lines

### Recommended

- Width: 120+ characters
- Height: 40+ lines

### Terminal Emulators Tested

- ✅ Windows Terminal (10+)
- ✅ macOS Terminal
- ✅ Linux: xterm, gnome-terminal, konsole
- ✅ VS Code integrated terminal
- ✅ tmux
- ✅ screen

## Advanced Customization

### Adding Custom Display Sections

```python
def _draw_custom_section(self, stdscr, row):
    self._add_line(stdscr, row, "CUSTOM SECTION", 'info')
    row += 1
    for item in custom_items:
        self._add_line(stdscr, row, item, 'normal')
        row += 1
    return row
```

### Creating New Display Mode

```python
class DashboardMode(Enum):
    # ... existing modes ...
    CUSTOM = "CUSTOM"

# In _display_loop:
if self.mode == DashboardMode.CUSTOM:
    self._draw_custom(stdscr)
```

## Related Documentation

- [LIVE_TRADING_ORCHESTRATOR_SPEC.md](LIVE_TRADING_ORCHESTRATOR_SPEC.md) - Orchestrator integration
- [SAFETY_LAYER_SPEC.md](SAFETY_LAYER_SPEC.md) - Safety events and alerts
- [QUICK_START_LIVE_TRADING.md](QUICK_START_LIVE_TRADING.md) - Getting started guide
