# Phase 4 Quick Start Guide: EngineHealthMonitor

**For:** Monitoring engine performance and scaling position sizing based on health  
**Time:** 5 minutes to understand  
**Experience Level:** Beginner to Intermediate

---

## What is EngineHealthMonitor?

A self-aware performance tracking system that:
- Monitors rolling engine metrics (Sharpe ratio, drawdown)
- Compares performance to regime-specific thresholds
- Outputs a **risk_multiplier** (1.0, 0.5, or 0.0) to scale position size
- Automatically disables new entries if performance degrades critically

---

## TL;DR - 30 Seconds

```python
from engine.health_monitor import EngineHealthMonitor

# Create monitor
monitor = EngineHealthMonitor(window_size=500)

# After each bar
monitor.update(pnl=daily_pnl, regime_label="low_vol")

# Get multiplier for position sizing
multiplier = monitor.get_risk_multiplier()  # 1.0, 0.5, or 0.0

# Scale position
new_size = desired_size * multiplier

# Check status
if monitor.get_health_status() == "CRITICAL":
    block_new_entries()
```

---

## Key Concepts

### Risk Multiplier

The multiplier scales position size based on engine health:

| Status | Multiplier | Meaning |
|--------|-----------|---------|
| HEALTHY | 1.0 | Full position size allowed |
| DEGRADED | 0.5 | Half position size |
| CRITICAL | 0.0 | No new entries, exits only |

### Health Evaluation

Health is determined by comparing rolling metrics to regime thresholds:

```
HEALTHY:
  ✓ Rolling Sharpe >= regime minimum
  ✓ Rolling Drawdown <= regime maximum
  
DEGRADED:
  ✓ One metric passes, one fails
  
CRITICAL:
  ✗ Both metrics fail
```

### Regime-Aware Thresholds

Different market conditions have different performance expectations:

| Regime | Min Sharpe | Max Drawdown |
|--------|-----------|--------------|
| high_vol | 0.20 | 15% |
| low_vol | 0.10 | 10% |
| risk_on | 0.15 | 12% |
| risk_off | 0.05 | 20% |

---

## 3-Step Integration

### Step 1: Initialize

```python
from engine.health_monitor import EngineHealthMonitor

# At tournament start
monitor = EngineHealthMonitor(window_size=500)
```

### Step 2: Update Each Bar

```python
# After each bar closes
monitor.update(
    pnl=bar_pnl,
    regime_label=market_regime,
    realized_pnl=cumulative_realized,
    unrealized_pnl=current_unrealized
)
```

### Step 3: Use Risk Multiplier

```python
# Get multiplier for position sizing
multiplier = monitor.get_risk_multiplier()

# Scale position
scaled_position = desired_position * multiplier

# Execute trade
execute_trade(action, symbol, scaled_position, price)

# If CRITICAL, block new entries
if monitor.get_health_status() == "CRITICAL":
    if action != "EXIT":
        action = "DO_NOTHING"
```

---

## Common Scenarios

### Scenario 1: Normal Day (HEALTHY)

```
Morning:  +500 P&L, low volatility
          sharpe=0.8, drawdown=0.05
          → HEALTHY (1.0)

Noon:     +200 P&L
          sharpe=0.9, drawdown=0.04
          → HEALTHY (1.0)

Afternoon: +300 P&L
          sharpe=1.1, drawdown=0.03
          → HEALTHY (1.0)

Result: Full position size all day
```

### Scenario 2: Degradation (HEALTHY → DEGRADED → CRITICAL)

```
Morning:  +500 P&L
          sharpe=0.8, drawdown=0.05
          → HEALTHY (1.0)

Mid-day:  Sudden loss: -800 P&L
          sharpe=0.4, drawdown=0.18
          → DEGRADED (0.5) ⚠️

Afternoon: More losses: -300 P&L
          sharpe=-0.1, drawdown=0.25
          → CRITICAL (0.0) 🛑

Result: Position size halved at degradation,
        exits only when critical
```

### Scenario 3: Recovery (CRITICAL → DEGRADED → HEALTHY)

```
Morning:  Bad start, losses accumulate
          → CRITICAL (0.0)

Mid-day:  Recovery trades (exits)
          sharpe=0.2, drawdown=0.18
          → DEGRADED (0.5) ✓

Afternoon: Continued recovery
          sharpe=0.6, drawdown=0.10
          → HEALTHY (1.0) ✅

Result: Position size restored as recovery progresses
```

---

## Customization

### Adjust Window Size

```python
# Slower, smoother health tracking
monitor = EngineHealthMonitor(window_size=1000)

# Faster, more responsive
monitor = EngineHealthMonitor(window_size=250)
```

### Customize Regime Thresholds

```python
# Make low_vol stricter
monitor.set_regime_thresholds("low_vol", min_sharpe=0.15, max_drawdown=0.08)

# Make risk_off more lenient
monitor.set_regime_thresholds("risk_off", min_sharpe=0.02, max_drawdown=0.30)
```

---

## Public Methods

### Core Methods

```python
# Update with new P&L
monitor.update(pnl=100.0, regime_label="low_vol")

# Get current multiplier for position sizing
multiplier = monitor.get_risk_multiplier()  # 1.0, 0.5, 0.0

# Get current health status
status = monitor.get_health_status()  # "HEALTHY", "DEGRADED", "CRITICAL"

# Get current Sharpe ratio
sharpe = monitor.compute_sharpe()  # e.g., 0.82

# Get current drawdown
drawdown = monitor.compute_drawdown()  # e.g., 0.12 (12%)
```

### Reporting

```python
# Get complete snapshot
snapshot = monitor.get_state_snapshot(realized_pnl=100, unrealized_pnl=50)
print(f"Status: {snapshot.health_status}")
print(f"Multiplier: {snapshot.risk_multiplier}")

# Get full report
report = monitor.get_report()
print(f"Transitions: {report['status_transitions']}")
print(f"Bars processed: {report['bars_processed']}")
```

### Session Management

```python
# Reset for new session (day/week/month)
monitor.reset_for_session()
```

---

## Integration with RealDataTournament

### In TradingEngineSimulator

```python
# Create simulator with health monitoring enabled
simulator = TradingEngineSimulator(
    symbol='EURUSD',
    price_data=price_df,
    track_health=True  # Enable health monitor
)

# Run simulation (health monitor tracks performance automatically)
trades = simulator.run_simulation()
```

### In Custom Trading Loop

```python
monitor = EngineHealthMonitor(window_size=500)

for bar in bars:
    # Get risk multiplier for this bar
    multiplier = monitor.get_risk_multiplier()
    
    # Block new entries if CRITICAL
    if monitor.get_health_status() == "CRITICAL":
        action = "EXIT" if position_open else "DO_NOTHING"
    
    # Scale position size
    position_size = desired_size * multiplier
    
    # Execute trade
    execute_trade(action, symbol, position_size, price)
    
    # Calculate bar P&L
    bar_pnl = calculate_pnl()
    
    # Detect regime
    regime = detect_regime(bar)
    
    # Update health monitor
    monitor.update(pnl=bar_pnl, regime_label=regime)
```

---

## Logging

Health monitor automatically logs to:

```
logs/health/health_<YYYYMMDD_HHMMSS>.log
```

Log events include:
- Initialization
- Each bar update (DEBUG level)
- Status transitions (WARNING level)
- Configuration changes (INFO level)

---

## Debugging

### Check Current State

```python
# Get full report
report = monitor.get_report()
print(json.dumps(report, indent=2))

# Output:
# {
#   "health_status": "HEALTHY",
#   "risk_multiplier": 1.0,
#   "rolling_sharpe": 0.82,
#   "rolling_drawdown": 0.05,
#   "current_regime": "low_vol",
#   "bars_in_window": 250,
#   "status_transitions": 2,
#   ...
# }
```

### Monitor Transitions

```python
# Get last status
status = monitor.get_health_status()

# Get number of transitions
report = monitor.get_report()
transitions = report['status_transitions']

print(f"Status: {status}, Transitions: {transitions}")
```

### Verify Metrics

```python
# Check if metrics meet regime thresholds
regime = monitor.get_regime()
regime_config = monitor.expected_bands[regime]

sharpe = monitor.compute_sharpe()
drawdown = monitor.compute_drawdown()

print(f"Sharpe: {sharpe:.3f} (min: {regime_config.min_sharpe})")
print(f"Drawdown: {drawdown:.3f} (max: {regime_config.max_drawdown})")
```

---

## Performance

| Operation | Time |
|-----------|------|
| Update | <1ms |
| Get multiplier | <1ms |
| Compute Sharpe | <1ms |
| Compute drawdown | <1ms |
| Get report | <1ms |

Memory-efficient with bounded deques (max 500-1000 bars).

---

## Best Practices

✅ **DO:**
- Call update() after every bar
- Check multiplier before scaling position size
- Use regime labels accurately
- Review logs for status transitions
- Test with different window sizes

❌ **DON'T:**
- Skip updates (gaps cause stale metrics)
- Ignore CRITICAL status (it's a safety mechanism)
- Change thresholds too frequently
- Use future P&L data (only historical)

---

## Troubleshooting

### Q: Why is status stuck at HEALTHY?

**A:** Window not full yet. Monitor needs ~250 bars before meaningful evaluation.

### Q: Can I change regimes mid-bar?

**A:** Yes, set regime in update() call. Next evaluation uses new regime thresholds.

### Q: What if I have multiple symbols?

**A:** Create separate EngineHealthMonitor for each symbol.

### Q: How do I reset for a new day?

**A:** Call `monitor.reset_for_session()` at start of new trading session.

---

## Examples

### Example 1: Simple Usage

```python
monitor = EngineHealthMonitor(window_size=500)

# Simulate trading day
for bar in bars:
    bar_pnl = calculate_pnl(bar)
    regime = get_market_regime(bar)
    
    monitor.update(pnl=bar_pnl, regime_label=regime)
    multiplier = monitor.get_risk_multiplier()
    
    print(f"Bar {bar.time}: PnL={bar_pnl}, Mult={multiplier}")
```

### Example 2: With Action Override

```python
def execute_with_health_check(action, symbol, size, price):
    status = monitor.get_health_status()
    multiplier = monitor.get_risk_multiplier()
    
    # Override action if critical
    if status == "CRITICAL" and action != "EXIT":
        action = "DO_NOTHING"
    
    # Scale size
    size = size * multiplier
    
    # Execute
    execute_trade(action, symbol, size, price)
```

### Example 3: Monitoring Health Progression

```python
history = []

for bar in bars:
    monitor.update(pnl=bar_pnl, regime_label=regime)
    
    snapshot = monitor.get_state_snapshot()
    history.append({
        'time': bar.time,
        'status': snapshot.health_status,
        'multiplier': snapshot.risk_multiplier,
        'sharpe': snapshot.rolling_sharpe,
        'drawdown': snapshot.rolling_drawdown
    })

# Analyze progression
df = pd.DataFrame(history)
print(df[df['status'] != df['status'].shift()])  # Show transitions
```

---

**For Complete Details:** See [ENGINE_HEALTH_MONITOR_SPEC.md](docs/ENGINE_HEALTH_MONITOR_SPEC.md)
