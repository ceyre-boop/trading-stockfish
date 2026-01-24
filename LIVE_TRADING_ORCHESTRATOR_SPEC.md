# LIVE TRADING ORCHESTRATOR SPECIFICATION

**Phase RT-3 Component**

## Overview

The `LiveTradingOrchestrator` is the central orchestration engine for live trading sessions. It manages the complete trading lifecycle, from initialization through shutdown, including state transitions, component coordination, health monitoring, and failsafe mechanisms.

## State Machine

```
STARTUP
   ↓
READY (all components healthy, waiting for market)
   ↓
LIVE (active trading)
   ↓
DEGRADED (anomaly detected, monitoring increased)
   ↓
FAILSAFE (critical anomaly, trading disabled, positions flattened)
   ↓
SHUTDOWN (graceful termination)

State Transitions:
- STARTUP → READY, SHUTDOWN
- READY → LIVE, SHUTDOWN
- LIVE → DEGRADED, SHUTDOWN
- DEGRADED → LIVE, FAILSAFE, SHUTDOWN
- FAILSAFE → READY, SHUTDOWN
- SHUTDOWN → (none)
```

## Architecture

### Component Hierarchy

```
LiveTradingOrchestrator
├── ExchangeManager (multi-connector orchestration)
├── RealTimeEngineLoop (trading decisions)
├── DataFeedRouter (data distribution)
├── SafetyLayer (anomaly detection)
├── EngineHealthMonitor (component health)
└── Governance (position/loss limits)
```

### Main Threads

| Thread | Purpose | Interval |
|--------|---------|----------|
| Main Loop | Event processing, state transitions | Continuous (10ms tick) |
| Health Monitor | Connector/engine health checks | 5 seconds (configurable) |
| Safety Monitor | Anomaly detection and response | Event-driven |

## Core Responsibilities

### 1. Component Initialization

```python
orchestrator = LiveTradingOrchestrator(
    exchange_manager=em,
    engine_loop=el,
    data_feed_router=dfr,
    safety_layer=sl
)

# Start all components
orchestrator.start()
```

**Initialization Sequence:**
1. Verify all components are created
2. Start ExchangeManager (connect to brokers)
3. Start DataFeedRouter (begin data distribution)
4. Start RealTimeEngineLoop (begin decision-making)
5. Initialize SafetyLayer
6. Transition to READY state

### 2. Session Management

#### Pre-Market (Before Market Open)

```python
# READY state
- All connectors connected and healthy
- No trades executed
- Waiting for market open signal
```

#### Market Open

```python
# Transition: READY → LIVE
orchestrator._set_state(OrchestratorState.LIVE)
- Begin accepting trading signals
- Start position tracking
- Enable risk management
```

#### Continuous Trading

```python
# LIVE state
- Process market data continuously
- Execute trades per engine decisions
- Monitor PnL and exposure
- Track all fills and rejections
```

#### Market Close

```python
# Post-market processing
- Close or flatten remaining positions (optional)
- Generate end-of-day reports
- Archive trading logs
```

### 3. State Transition Management

```python
def _set_state(self, new_state: OrchestratorState) -> bool:
    # Validate transition is allowed
    # Update internal state
    # Log transition
    # Trigger all callbacks
    # Return success status
```

**Callbacks on State Change:**
- Custom handlers registered via `register_state_change_callback()`
- Used for UI updates, alerting, position management

### 4. Health Monitoring

#### Connector Health

```python
def _check_connector_health(self):
    for connector in self.exchange_manager:
        is_connected = connector.is_connected_check()
        latency = connector.get_stats()['latency_ms']
        error_count = connector.get_stats()['error_count']
        
        if not is_connected:
            # Trigger degradation
        if error_count > threshold:
            # Record failure event
```

**Health Metrics:**
- Is connected (boolean)
- Latency (milliseconds)
- Messages received/dropped
- Error count

#### Engine Health

```python
def _check_engine_health(self):
    stats = engine_loop.get_stats()
    
    decisions_made = stats['decisions_made']
    latency_p50 = stats['latency_p50_ms']
    latency_p99 = stats['latency_p99_ms']
    error_count = stats['error_count']
    
    if error_count > threshold:
        # Record health event
```

#### Overall Assessment

```python
def _assess_overall_health(self):
    # Check if all connectors down → DEGRADED
    # Check if critical failures → DEGRADED
    # Check if repeated degradations → FAILSAFE
    # Check if recovered → reset degradation counter
```

### 5. Safety Integration

```python
def on_safety_event(self, event: SafetyEvent):
    if event.event_type == SafetyEventType.DATA_ANOMALY:
        # Transition to DEGRADED
        self._set_state(OrchestratorState.DEGRADED)
    
    if event.severity == "critical":
        # Trigger failsafe
        self._trigger_failsafe()
```

**Safety Event Handling:**
- Data anomalies → DEGRADED
- Execution anomalies → DEGRADED
- Stale feeds → DEGRADED
- Critical events → FAILSAFE

### 6. Failsafe Activation

```python
def _trigger_failsafe(self):
    # Flatten all positions
    self.engine_loop.flatten_positions()
    
    # Disable trading
    self.engine_loop.disable_trading()
    
    # Transition to FAILSAFE state
    self._set_state(OrchestratorState.FAILSAFE)
```

**Failsafe Actions:**
- Immediately flatten all open positions
- Cancel all pending orders
- Disable new order submission
- Enable read-only mode (data collection only)
- Alert operators
- Generate incident report

### 7. Recovery from Degradation

```python
def recover_from_degraded(self) -> bool:
    # Verify connectors are healthy
    # Reset safety layer
    # Return to LIVE state
    # Reset degradation counter
```

**Recovery Conditions:**
- Must be in DEGRADED state
- At least one connector healthy
- Safety layer reset succeeds

## API Reference

### Lifecycle Methods

#### `start() -> bool`
Start live trading session.
```python
success = orchestrator.start()
if success:
    print("Session started")
```

#### `stop() -> bool`
Stop live trading session gracefully.
```python
success = orchestrator.stop()
if success:
    print("Session stopped")
```

#### `is_running() -> bool`
Check if session is running.
```python
if orchestrator.is_running():
    print("Session active")
```

### State Management

#### `get_state() -> OrchestratorState`
Get current orchestrator state.
```python
state = orchestrator.get_state()
if state == OrchestratorState.LIVE:
    print("Trading live")
```

### Event Handling

#### `on_connector_event(event: ConnectorHealthEvent)`
Handle connector health events.
```python
def handle_connector(event):
    print(f"Connector {event.connector_name}: {event.latency_ms}ms")

orchestrator.register_connector_callback(handle_connector)
```

#### `on_health_event(event: EngineHealthEvent)`
Handle engine health events.
```python
def handle_health(event):
    print(f"Engine latency P99: {event.latency_p99_ms}ms")

orchestrator.register_health_callback(handle_health)
```

#### `on_safety_event(event: SafetyEvent)`
Handle safety layer events.
```python
def handle_safety(event):
    print(f"Safety: {event.event_type.value} - {event.message}")

orchestrator.register_safety_callback(handle_safety)
```

### Recovery

#### `recover_from_degraded() -> bool`
Attempt recovery from DEGRADED state.
```python
if orchestrator.get_state() == OrchestratorState.DEGRADED:
    if orchestrator.recover_from_degraded():
        print("Recovered to LIVE")
```

### Statistics

#### `get_stats() -> Dict[str, Any]`
Get session statistics.
```python
stats = orchestrator.get_stats()
print(f"Session duration: {stats['session_duration']:.1f}s")
print(f"Safety events: {stats['safety_events']}")
print(f"PnL: {stats['realized_pnl']:.2f}")
```

#### `get_connector_status() -> Dict[str, Dict]`
Get all connector status.
```python
status = orchestrator.get_connector_status()
for name, info in status.items():
    print(f"{name}: {'Connected' if info['is_connected'] else 'Down'}")
```

## Configuration

Default configuration can be overridden:

```python
config = {
    'health_check_interval_s': 5.0,
    'failsafe_degradation_count': 3,
}

orchestrator = LiveTradingOrchestrator(
    exchange_manager=em,
    engine_loop=el,
    data_feed_router=dfr,
    safety_layer=sl,
    config=config
)
```

### Configuration Options

| Option | Default | Purpose |
|--------|---------|---------|
| `health_check_interval_s` | 5.0 | Seconds between health checks |
| `failsafe_degradation_count` | 3 | Degradations before failsafe |

## Error Handling

### Component Initialization Failure

```python
if not orchestrator.start():
    print("Error: Failed to initialize components")
    # Check logs for specific component failure
```

### Connector Disconnection

```python
# Automatic handling:
# 1. HealthMonitor detects disconnection
# 2. Orchestrator transitions to DEGRADED
# 3. Retries connection with exponential backoff
# 4. Transitions to FAILSAFE if persistent
```

### Safety Event Handling

```python
# Automatic handling:
# 1. SafetyLayer detects anomaly
# 2. Publishes SafetyEvent
# 3. Orchestrator processes event
# 4. Transitions state if needed
# 5. Triggers failsafe if critical
```

## Logging

All state transitions and events logged to:
- `logs/live/live_trading_<timestamp>.log` - Main orchestrator log
- `logs/live/safety_events_<timestamp>.log` - Safety events
- `logs/live/connector_health_<timestamp>.log` - Connector health
- `logs/live/governance_events_<timestamp>.log` - Governance events

Sample log entry:
```
2026-01-19 14:30:45,123 - LiveTradingOrchestrator - INFO - State transition: LIVE → DEGRADED
2026-01-19 14:30:45,456 - SafetyLayer - WARNING - Safety event: DATA_ANOMALY: Price jump 25.5% from 450.00 to 562.50
2026-01-19 14:30:50,789 - ExchangeManager - WARNING - Connector ibkr disconnected
```

## Usage Examples

### Basic Session

```python
from realtime import (
    LiveTradingOrchestrator,
    ExchangeManager,
    RealTimeEngineLoop,
    DataFeedRouter,
    SafetyLayer
)

# Create components
em = ExchangeManager()
el = RealTimeEngineLoop()
dfr = DataFeedRouter()
sl = SafetyLayer()

# Create orchestrator
orchestrator = LiveTradingOrchestrator(em, el, dfr, sl)

# Start session
orchestrator.start()

# Monitor state
while orchestrator.is_running():
    state = orchestrator.get_state()
    stats = orchestrator.get_stats()
    print(f"State: {state.value}, PnL: {stats['realized_pnl']:.2f}")
    time.sleep(1)

# Stop session
orchestrator.stop()
```

### With Recovery Handling

```python
orchestrator.start()

while orchestrator.is_running():
    state = orchestrator.get_state()
    
    if state == OrchestratorState.DEGRADED:
        # Try to recover
        if orchestrator.recover_from_degraded():
            print("Recovered to LIVE")
        else:
            print("Recovery failed, waiting...")
            time.sleep(5)
    
    time.sleep(1)

orchestrator.stop()
```

### With Custom Callbacks

```python
def on_state_change(old_state, new_state):
    print(f"State: {old_state.value} → {new_state.value}")
    if new_state == OrchestratorState.FAILSAFE:
        # Send alert
        send_alert("FAILSAFE ACTIVATED")

def on_safety_event(event):
    if event.severity == "critical":
        # Log incident
        log_incident(event)

orchestrator.register_state_change_callback(on_state_change)
orchestrator.register_safety_event_callback(on_safety_event)

orchestrator.start()
# ... rest of trading session
```

## Troubleshooting

### Session Won't Start

**Issue:** `start()` returns False
**Solution:**
1. Check logs in `logs/live/live_trading_*.log`
2. Verify all components are properly initialized
3. Check connector connectivity
4. Verify configuration is valid

### Repeated Transitions to DEGRADED

**Issue:** Session keeps transitioning LIVE → DEGRADED → LIVE
**Solution:**
1. Check SafetyLayer for recurring anomalies
2. Review `safety_events_*.log`
3. Verify feed quality and latency
4. Increase health check interval if too aggressive

### Failsafe Triggered Unexpectedly

**Issue:** Session enters FAILSAFE state without warning
**Solution:**
1. Review detailed logs in `logs/live/`
2. Check for repeated degradations (threshold=3)
3. Verify connector stability
4. Review safety event thresholds

## Performance Considerations

### Memory Usage
- Typical: 50-100 MB
- With large position/order history: up to 500 MB

### CPU Usage
- Idle (READY): <1%
- LIVE (moderate trading): 2-5%
- High-frequency trading: 10-20%

### Latency Impact
- Orchestrator overhead: <1ms per decision
- State transition latency: <5ms
- Health check impact: negligible

## Thread Safety

All public methods are thread-safe:
- State changes use locks
- Event queues are thread-safe
- Statistics access is synchronized

## Best Practices

1. **Always use context manager:**
   ```python
   with LiveTradingOrchestrator(...) as orch:
       # Session automatically starts and stops
   ```

2. **Monitor state transitions:**
   ```python
   orch.register_state_change_callback(on_state_change)
   ```

3. **Handle safety events:**
   ```python
   orch.register_safety_event_callback(on_safety_event)
   ```

4. **Regular health checks:**
   ```python
   stats = orch.get_stats()
   if stats['failsafe_activations'] > 0:
       investigate()
   ```

5. **Graceful shutdown:**
   ```python
   orch.stop()  # Always call, never kill process
   ```

## Related Documentation

- [SAFETY_LAYER_SPEC.md](SAFETY_LAYER_SPEC.md) - Safety checks and anomaly detection
- [MONITORING_DASHBOARD_GUIDE.md](MONITORING_DASHBOARD_GUIDE.md) - Real-time monitoring
- [QUICK_START_LIVE_TRADING.md](QUICK_START_LIVE_TRADING.md) - Getting started guide
- [PHASE_RT2_IMPLEMENTATION.md](PHASE_RT2_IMPLEMENTATION.md) - Exchange connectors
