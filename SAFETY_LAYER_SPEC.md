# SAFETY LAYER SPECIFICATION

**Phase RT-3 Component**

## Overview

The `SafetyLayer` is a real-time anomaly detection system that monitors live trading for data integrity issues, execution anomalies, and feed health problems. It automatically detects anomalies and triggers appropriate responses through the orchestrator.

## Safety Rules

### Data Sanity Checks

#### 1. Negative Prices

**Rule:** No price component can be negative

```
Condition: bid < 0 OR ask < 0 OR last < 0
Severity: CRITICAL
Action: Generate SafetyEvent, trigger assessment
```

**Example:**
```python
# Detected anomaly
bid = 450.00
ask = 452.00
last = -100.00  # ANOMALY - negative price

event = SafetyEvent(
    event_type=SafetyEventType.NEGATIVE_PRICE,
    severity="CRITICAL",
    symbol="SPY",
    message="Negative price detected: bid=450.00, ask=452.00, last=-100.00"
)
```

#### 2. Crossed Market

**Rule:** Best bid must not exceed best ask

```
Condition: bid > ask
Severity: WARNING (if isolated) or CRITICAL (if persistent)
Action: Generate SafetyEvent, investigate feed
```

**Example:**
```python
# Crossed market detected
bid = 455.00
ask = 452.00  # Bid > Ask = ANOMALY

event = SafetyEvent(
    event_type=SafetyEventType.ORDERBOOK_INVALID,
    severity="WARNING",
    symbol="QQQ",
    message="Crossed market: bid=455.00 > ask=452.00"
)
```

#### 3. Price Jumps

**Rule:** Price changes limited to configurable threshold (default: 20%)

```
Condition: |price_change| / previous_price > max_jump_pct
Severity: WARNING
Action: Log event, investigate market conditions
```

**Configuration:**
```python
config = {
    'max_price_jump_pct': 20.0  # Allow up to 20% moves
}
```

**Example:**
```python
# Price jump detection
previous = 450.00
current = 562.50
change_pct = abs((562.50 - 450.00) / 450.00) * 100 = 25%  # Exceeds 20%

event = SafetyEvent(
    event_type=SafetyEventType.PRICE_SPIKE,
    severity="WARNING",
    message="Price jump 25.00% from 450.00 to 562.50"
)
```

#### 4. Time Gaps

**Rule:** Max gap between price updates limited (default: 30 seconds)

```
Condition: time_gap > max_time_gap_s
Severity: WARNING
Action: Flag as potential connection issue
```

**Configuration:**
```python
config = {
    'max_time_gap_s': 30.0  # Max 30s between updates
}
```

**Example:**
```python
# Time gap detection
last_update_time = 14:30:00
current_update_time = 14:30:45
gap = 45 seconds  # Exceeds 30s threshold

event = SafetyEvent(
    event_type=SafetyEventType.TIME_GAP,
    severity="WARNING",
    message="Time gap 45.0s exceeds threshold 30.0s"
)
```

#### 5. Bid-Ask Spread Anomalies

**Rule:** Bid-ask spread limited to multiplier of historical norm (default: 10x)

```
Condition: current_spread / historical_spread > bid_ask_spread_multiplier
Severity: WARNING
Action: Monitor for feed quality issues
```

**Configuration:**
```python
config = {
    'bid_ask_spread_multiplier': 10.0  # Max 10x normal spread
}
```

**Example:**
```python
# Spread anomaly detection
historical_spread = 0.02  # Normal: 0.02 on SPY
historical_spread_pct = (0.02 / 450) * 100 = 0.0044%

current_bid = 450.00
current_ask = 450.50
current_spread = 0.50
current_spread_pct = (0.50 / 450.50) * 100 = 0.111%

multiplier = 0.111 / 0.0044 = 25.2x  # Exceeds 10x threshold

event = SafetyEvent(
    event_type=SafetyEventType.DATA_ANOMALY,
    severity="WARNING",
    message="Spread anomaly: 0.1111% (25.2x normal)"
)
```

### Execution Sanity Checks

#### 1. Fills Outside Expected Range

**Rule:** Execution fill price must be within reasonable slippage (default: 5%)

```
Condition: |fill_price - order_price| / order_price > slippage_threshold
Severity: WARNING (if minor) or CRITICAL (if major)
Action: Log execution, investigate broker
```

**Example:**
```python
# Execution slippage check
order_price = 450.00
fill_price = 468.75  # 4.17% worse than order
slippage = (468.75 - 450.00) / 450.00 * 100 = 4.17%  # Within 5% threshold

# OK - within threshold
```

#### 2. Fill Quantity Exceeds Order

**Rule:** Filled quantity cannot exceed ordered quantity

```
Condition: fill_quantity > order_quantity
Severity: CRITICAL
Action: Immediate investigation, potential system error
```

**Example:**
```python
# Fill quantity validation
order_quantity = 100
fill_quantity = 125  # CRITICAL - more than ordered

event = SafetyEvent(
    event_type=SafetyEventType.EXECUTION_ANOMALY,
    severity="CRITICAL",
    message="Fill quantity 125 exceeds order 100"
)
```

#### 3. Rejection Loop Detection

**Rule:** Repeated rejections within window indicate systemic issue (default: 5 in 60s)

```
Condition: rejection_count >= threshold within window
Severity: CRITICAL
Action: Disable order submission, trigger failsafe
```

**Configuration:**
```python
config = {
    'reject_loop_threshold': 5,      # 5 rejections
    'reject_window_s': 60.0           # in 60 seconds
}
```

**Example:**
```python
# Rejection loop detection
timestamps = [14:30:01, 14:30:02, 14:30:03, 14:30:04, 14:30:05]
# 5 rejections in 5 seconds → ANOMALY

event = SafetyEvent(
    event_type=SafetyEventType.ORDER_REJECT_LOOP,
    severity="CRITICAL",
    message="Rejection loop detected: 5 rejections in 60s"
)
```

### Feed Health Checks

#### 1. Stale Price Detection

**Rule:** If no price update for threshold duration, mark feed stale (default: 10s)

```
Condition: time_since_update > stale_threshold
Severity: WARNING
Action: Flag as stale, trigger monitoring increase
```

**Configuration:**
```python
config = {
    'stale_data_threshold_s': 10.0  # 10 seconds
}
```

**Example:**
```python
# Stale feed detection
last_update = 14:30:15
current_time = 14:30:27
time_since_update = 12 seconds  # Exceeds 10s threshold

event = SafetyEvent(
    event_type=SafetyEventType.FEED_STALE,
    severity="WARNING",
    symbol="SPY",
    message="Feed stale for 12.0s (threshold: 10.0s)"
)
```

#### 2. Stale Order Book Detection

**Rule:** Monitor for stale order book snapshots

```
Condition: last_orderbook_update > threshold
Severity: WARNING
Action: May indicate stale exchange data
```

#### 3. Stale News/Macro Data

**Rule:** Monitor for stale news and macro economic data

```
Condition: last_news_update > threshold OR last_macro_update > threshold
Severity: INFO/WARNING
Action: Background monitoring, non-critical
```

## Automatic Transitions

### Degradation Triggers

The SafetyLayer automatically triggers transitions:

```
LIVE → DEGRADED when:
  - Data anomaly detected (negative prices, crossed market)
  - Execution anomaly detected (fills outside range)
  - Feed becomes stale (no update > 10s)
  - Multiple safety events in short time
```

### Failsafe Triggers

The SafetyLayer triggers failsafe when:

```
DEGRADED → FAILSAFE when:
  - Critical severity events detected
  - Repeated degradations (>= 3)
  - Rejection loop detected
  - Data integrity compromised
```

**Failsafe Actions:**
```
1. Flatten all open positions
2. Cancel all pending orders
3. Disable new order submission
4. Enable read-only mode
5. Alert operators
6. Generate incident report
```

## API Reference

### Price Tick Checking

#### `check_price_tick(symbol, bid, ask, last, timestamp) -> Optional[SafetyEvent]`

Check single price tick for anomalies.

```python
event = safety_layer.check_price_tick(
    symbol="SPY",
    bid=450.00,
    ask=450.50,
    last=450.25,
    timestamp=time.time()
)

if event:
    print(f"Anomaly: {event.message}")
```

**Checks Performed:**
- Negative prices
- Crossed market
- Price jumps
- Time gaps
- Price updates tracked

### Order Book Checking

#### `check_orderbook(symbol, bid_levels, ask_levels, timestamp) -> Optional[SafetyEvent]`

Check order book snapshot for anomalies.

```python
bid_levels = [
    (450.00, 1000),
    (449.95, 2000),
    (449.90, 3000)
]
ask_levels = [
    (450.50, 1500),
    (450.55, 2500),
    (450.60, 3500)
]

event = safety_layer.check_orderbook(
    symbol="SPY",
    bid_levels=bid_levels,
    ask_levels=ask_levels,
    timestamp=time.time()
)
```

**Checks Performed:**
- Negative prices
- Crossed market
- Spread sanity
- Level validity

### Execution Checking

#### `check_execution(symbol, order_price, fill_price, fill_quantity, order_quantity) -> Optional[SafetyEvent]`

Check order execution for anomalies.

```python
event = safety_layer.check_execution(
    symbol="SPY",
    order_price=450.00,
    fill_price=450.50,
    fill_quantity=100,
    order_quantity=100
)
```

**Checks Performed:**
- Negative prices
- Slippage validation
- Fill quantity validation

### Rejection Tracking

#### `record_order_rejection(symbol, timestamp) -> Optional[SafetyEvent]`

Record order rejection and detect patterns.

```python
event = safety_layer.record_order_rejection(
    symbol="SPY",
    timestamp=time.time()
)

if event and event.event_type == SafetyEventType.ORDER_REJECT_LOOP:
    print("Rejection loop detected!")
```

### Feed Health Checking

#### `check_feed_staleness(symbol, current_time) -> Optional[SafetyEvent]`

Check if feed data is stale.

```python
event = safety_layer.check_feed_staleness(
    symbol="SPY",
    current_time=time.time()
)

if event:
    print(f"Feed stale: {event.message}")
```

### State Management

#### `reset() -> None`

Reset safety layer state (e.g., after recovery).

```python
safety_layer.reset()  # Clear all anomaly counters
```

#### `get_stats() -> Dict[str, Any]`

Get safety layer statistics.

```python
stats = safety_layer.get_stats()
print(f"Total anomalies: {stats['total_anomalies']}")
print(f"Price spikes: {stats['anomaly_counts']['PRICE_SPIKE']}")
```

#### `get_health_status() -> Dict[str, Any]`

Get current health status.

```python
status = safety_layer.get_health_status()
print(f"Tracked symbols: {status['tracked_symbols']}")
print(f"Stale symbols: {status['stale_symbols']}")
print(f"Overall health: {status['health_status']}")
```

## Configuration

```python
config = {
    # Price checks
    'max_price_jump_pct': 20.0,           # 20% max jump
    'max_time_gap_s': 30.0,               # 30s max gap
    'bid_ask_spread_multiplier': 10.0,    # 10x normal spread
    
    # Execution checks
    'max_execution_time_s': 5.0,          # 5s max fill time
    'reject_loop_threshold': 5,           # 5 rejects
    
    # Feed health
    'stale_data_threshold_s': 10.0,       # 10s stale threshold
}

safety_layer = SafetyLayer(config=config)
```

## Event Types

### SafetyEventType Enum

```python
class SafetyEventType(Enum):
    DATA_ANOMALY = "DATA_ANOMALY"
    EXECUTION_ANOMALY = "EXECUTION_ANOMALY"
    FEED_STALE = "FEED_STALE"
    PRICE_SPIKE = "PRICE_SPIKE"
    ORDER_REJECT_LOOP = "ORDER_REJECT_LOOP"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    TIME_GAP = "TIME_GAP"
    ORDERBOOK_INVALID = "ORDERBOOK_INVALID"
```

### Event Structure

```python
@dataclass
class SafetyEvent:
    timestamp: float              # Event timestamp
    event_type: SafetyEventType   # Type of anomaly
    severity: str                 # "info", "warning", "critical"
    symbol: str                   # Affected symbol
    message: str                  # Human-readable description
    data: Optional[Dict]          # Event-specific data
```

## Integration with Orchestrator

### Event Publishing

```python
# Safety layer detects anomaly
event = SafetyEvent(...)

# Publish to orchestrator
orchestrator.on_safety_event(event)

# Orchestrator processes:
if event.severity == "critical":
    orchestrator._trigger_failsafe()
elif event.event_type in [DATA_ANOMALY, EXECUTION_ANOMALY]:
    orchestrator._set_state(OrchestratorState.DEGRADED)
```

### Callback Registration

```python
def handle_safety_event(event):
    logger.warning(f"Safety event: {event.message}")
    if event.severity == "critical":
        send_alert(event.message)

orchestrator.register_safety_event_callback(handle_safety_event)
```

## Monitoring and Logging

### Logged Events

All anomalies logged to: `logs/live/safety_events_<timestamp>.log`

```
2026-01-19 14:30:45,123 - SafetyLayer - WARNING - Safety event: PRICE_SPIKE: Price jump 25.5% from 450.00 to 562.50
2026-01-19 14:30:50,456 - SafetyLayer - CRITICAL - Safety event: NEGATIVE_PRICE: Negative price detected: bid=450.00, ask=452.00, last=-100.00
```

### Statistics Available

```python
stats = safety_layer.get_stats()
# {
#     'total_anomalies': 5,
#     'anomaly_counts': {
#         'PRICE_SPIKE': 2,
#         'FEED_STALE': 2,
#         'DATA_ANOMALY': 1,
#         ...
#     },
#     'tracked_symbols': 10,
#     'symbols_with_rejects': 2
# }
```

## Performance Considerations

### Latency Impact

- Price tick check: <1ms
- Order book check: <1ms
- Execution check: <0.5ms
- Feed staleness check: <0.1ms

### Memory Usage

- Per symbol tracking: ~500 bytes
- With 100 symbols: ~50 KB
- Rejection history (per symbol): ~1-10 KB

### CPU Impact

- Background monitoring: negligible
- Per-event checking: <1% CPU per 1000 events/s

## Tuning Guidelines

### Aggressive Safety (Low Risk Tolerance)

```python
config = {
    'max_price_jump_pct': 10.0,           # Stricter
    'max_time_gap_s': 5.0,                # Stricter
    'bid_ask_spread_multiplier': 5.0,     # Stricter
    'reject_loop_threshold': 3,           # Stricter
    'stale_data_threshold_s': 5.0,        # Stricter
}
```

### Conservative Safety (High Risk Tolerance)

```python
config = {
    'max_price_jump_pct': 50.0,           # More lenient
    'max_time_gap_s': 60.0,               # More lenient
    'bid_ask_spread_multiplier': 20.0,    # More lenient
    'reject_loop_threshold': 10,          # More lenient
    'stale_data_threshold_s': 30.0,       # More lenient
}
```

### Balanced Safety (Recommended)

```python
config = {
    'max_price_jump_pct': 20.0,           # Default
    'max_time_gap_s': 30.0,               # Default
    'bid_ask_spread_multiplier': 10.0,    # Default
    'reject_loop_threshold': 5,           # Default
    'stale_data_threshold_s': 10.0,       # Default
}
```

## Best Practices

1. **Monitor safety statistics regularly:**
   ```python
   stats = safety_layer.get_stats()
   if stats['total_anomalies'] > 10:
       alert("High anomaly count")
   ```

2. **Reset on recovery:**
   ```python
   if orchestrator.recover_from_degraded():
       safety_layer.reset()
   ```

3. **Log all critical events:**
   ```python
   if event.severity == "critical":
       logger.critical(f"Safety critical: {event.to_dict()}")
   ```

4. **Tune based on historical data:**
   ```python
   # Analyze normal market conditions
   # Adjust thresholds accordingly
   # Re-tune quarterly or after market changes
   ```

5. **Test anomaly detection:**
   ```python
   # Inject test anomalies
   # Verify detection works
   # Verify failsafe triggers
   ```

## Related Documentation

- [LIVE_TRADING_ORCHESTRATOR_SPEC.md](LIVE_TRADING_ORCHESTRATOR_SPEC.md) - Orchestrator integration
- [MONITORING_DASHBOARD_GUIDE.md](MONITORING_DASHBOARD_GUIDE.md) - Alert display
- [QUICK_START_LIVE_TRADING.md](QUICK_START_LIVE_TRADING.md) - Getting started
