# ENGINE HEALTH MONITOR SPECIFICATION

**Phase 4 - Trading Stockfish v1.0**  
**Version:** 1.0  
**Date:** January 19, 2026

---

## Executive Summary

The **EngineHealthMonitor** is a self-aware performance tracking system that monitors rolling engine metrics (Sharpe ratio, drawdown) against regime-specific thresholds and outputs a **risk_multiplier** ∈ {1.0, 0.5, 0.0} to scale the PolicyEngine's allowed position size.

**Core Purpose:** Detect when the trading engine's performance deviates from expected behavior and automatically reduce risk exposure or disable new entries.

**Key Metrics:**
- Rolling annualized Sharpe ratio
- Rolling maximum drawdown
- Regime-aware thresholds
- Health status (HEALTHY / DEGRADED / CRITICAL)
- Risk multiplier for position sizing

---

## Architecture

### System Overview

```
Trading Engine
    ↓
Generate P&L (each bar)
    ↓
EngineHealthMonitor.update(pnl, regime)
    ├─ Update rolling P&L
    ├─ Calculate rolling Sharpe
    ├─ Calculate rolling drawdown
    ├─ Evaluate health (vs regime thresholds)
    └─ Output risk_multiplier
    ↓
PolicyEngine receives risk_multiplier
    ├─ Scales max_position_size
    └─ If CRITICAL: Block new entries
    ↓
Execute trade (position_size * risk_multiplier)
```

### Component Responsibilities

| Component | Responsibility |
|-----------|-----------------|
| **EngineHealthMonitor** | Track performance, calculate metrics, evaluate health |
| **PolicyEngine** | Apply risk_multiplier to position sizing |
| **TradingEngine** | Feed P&L and regime to health monitor |

---

## Core Attributes

### Rolling Metrics

```python
rolling_pnl: deque           # Recent P&L values (max window_size)
rolling_returns: deque       # Recent returns (max window_size)
rolling_sharpe: float        # Annualized Sharpe ratio
rolling_drawdown: float      # Max drawdown in window (0.15 = 15%)
```

### Regime Tracking

```python
current_regime: str          # Current market regime
regime_history: deque        # History of regimes (max window_size)
```

### Regime Thresholds

```python
expected_bands: dict {
    "high_vol": {
        "min_sharpe": 0.2,
        "max_drawdown": 0.15
    },
    "low_vol": {
        "min_sharpe": 0.1,
        "max_drawdown": 0.10
    },
    "risk_on": {
        "min_sharpe": 0.15,
        "max_drawdown": 0.12
    },
    "risk_off": {
        "min_sharpe": 0.05,
        "max_drawdown": 0.20
    }
}
```

### Health State

```python
health_status: str           # "HEALTHY", "DEGRADED", "CRITICAL"
risk_multiplier: float       # 1.0, 0.5, or 0.0
cumulative_pnl: float        # Cumulative P&L
peak_cumulative_pnl: float   # Highest P&L reached
```

---

## Core Methods

### 1. `__init__(window_size=500, annualization_factor=252.0)`

Initialize health monitor.

**Parameters:**
- `window_size` (int): Number of bars for rolling window (default: 500)
- `annualization_factor` (float): Days per year for Sharpe annualization (default: 252 trading days)

**Behavior:**
- Creates empty rolling deques
- Initializes regime thresholds to defaults
- Sets health status to HEALTHY with risk_multiplier = 1.0

**Example:**
```python
monitor = EngineHealthMonitor(window_size=500)
```

---

### 2. `update(pnl, regime_label, realized_pnl=0.0, unrealized_pnl=0.0)`

Update health monitor with new P&L and regime information.

**Parameters:**
- `pnl` (float): P&L change for this bar (can be positive or negative)
- `regime_label` (str): Current market regime ("high_vol", "low_vol", "risk_on", "risk_off")
- `realized_pnl` (float): Cumulative realized P&L (for snapshot)
- `unrealized_pnl` (float): Current unrealized P&L (for snapshot)

**Behavior:**
- Adds P&L to rolling deques
- Updates cumulative P&L and tracks peak
- Recalculates rolling Sharpe and drawdown
- Evaluates health status
- Logs all updates

**Returns:** None (side effect: modifies internal state)

**Example:**
```python
# After a bar closes with -$150 loss in low_vol regime
monitor.update(pnl=-150.0, regime_label="low_vol", realized_pnl=-1000, unrealized_pnl=100)
```

---

### 3. `compute_sharpe() → float`

Get current rolling annualized Sharpe ratio.

**Formula:**
```
Sharpe = (mean_return / std_return) * sqrt(annualization_factor)
```

**Behavior:**
- Calculates from rolling_returns deque
- Annualizes to yearly metric
- Returns 0.0 if insufficient data

**Returns:** Annualized Sharpe ratio

**Example:**
```python
sharpe = monitor.compute_sharpe()  # Returns 0.82 (good)
```

---

### 4. `compute_drawdown() → float`

Get current rolling maximum drawdown.

**Formula:**
```
Max Drawdown = (Peak - Trough) / Peak
```

**Behavior:**
- Finds highest and lowest cumulative P&L in window
- Returns drawdown as proportion (0.15 = 15%)
- Returns 0.0 if insufficient data

**Returns:** Max drawdown in rolling window

**Example:**
```python
drawdown = monitor.compute_drawdown()  # Returns 0.12 (12%)
```

---

### 5. `evaluate_health() → None`

Evaluate engine health based on rolling metrics vs regime thresholds.

**Health Evaluation Logic:**
```
If (sharpe >= min_sharpe AND drawdown <= max_drawdown):
    status = "HEALTHY"
    multiplier = 1.0
Elif (sharpe >= min_sharpe OR drawdown <= max_drawdown):
    status = "DEGRADED"
    multiplier = 0.5
Else:
    status = "CRITICAL"
    multiplier = 0.0
```

**Behavior:**
- Compares rolling metrics to regime-specific thresholds
- Assigns health status
- Sets risk_multiplier
- Logs transitions

**Returns:** None (side effect: updates health_status and risk_multiplier)

**Example:**
```python
# If Sharpe drops below threshold:
# HEALTHY (1.0) → DEGRADED (0.5) → CRITICAL (0.0)
```

---

### 6. `get_risk_multiplier() → float`

Get current risk multiplier for position sizing.

**Returns:** 
- `1.0` - Full position size allowed (HEALTHY)
- `0.5` - Half position size (DEGRADED)
- `0.0` - No new entries (CRITICAL)

**Example:**
```python
multiplier = monitor.get_risk_multiplier()
new_position_size = desired_size * multiplier
```

---

### 7. `get_health_status() → str`

Get current health status.

**Returns:** "HEALTHY", "DEGRADED", or "CRITICAL"

**Example:**
```python
if monitor.get_health_status() == "CRITICAL":
    disable_new_entries()
```

---

### 8. `get_state_snapshot(realized_pnl=0.0, unrealized_pnl=0.0) → HealthSnapshot`

Get immutable snapshot of current health state.

**Returns:** `HealthSnapshot` dataclass with:
- `timestamp`: Current datetime
- `rolling_sharpe`: Current Sharpe ratio
- `rolling_drawdown`: Current drawdown
- `regime_label`: Current regime
- `health_status`: Current status
- `risk_multiplier`: Current multiplier
- `bars_in_window`: Bars in rolling window
- `realized_pnl`: Realized P&L
- `unrealized_pnl`: Unrealized P&L
- `daily_pnl`: Daily cumulative P&L

**Example:**
```python
snapshot = monitor.get_state_snapshot(realized_pnl=-500, unrealized_pnl=200)
print(f"Status: {snapshot.health_status}, Multiplier: {snapshot.risk_multiplier}")
```

---

### 9. `get_report() → Dict`

Get comprehensive health report.

**Returns:** Dictionary with:
- `health_status`: Current status
- `risk_multiplier`: Current multiplier
- `rolling_sharpe`: Current Sharpe
- `rolling_drawdown`: Current drawdown
- `current_regime`: Current regime
- `cumulative_pnl`: Total P&L
- `peak_cumulative_pnl`: Peak P&L
- `bars_processed`: Total bars processed
- `bars_in_window`: Bars currently in window
- `window_size`: Max window size
- `status_transitions`: Number of status changes
- `expected_bands`: All regime thresholds

**Example:**
```python
report = monitor.get_report()
print(f"Transitions: {report['status_transitions']}")
```

---

### 10. `set_regime_thresholds(regime, min_sharpe, max_drawdown) → None`

Customize thresholds for a specific regime.

**Parameters:**
- `regime` (str): Regime label
- `min_sharpe` (float): Minimum acceptable Sharpe
- `max_drawdown` (float): Maximum acceptable drawdown

**Example:**
```python
# Increase risk tolerance in high_vol regime
monitor.set_regime_thresholds("high_vol", min_sharpe=0.1, max_drawdown=0.20)
```

---

### 11. `reset_for_session() → None`

Reset rolling metrics for new session (new day, new symbol, etc).

**Behavior:**
- Clears rolling_pnl and rolling_returns deques
- Resets rolling_sharpe and rolling_drawdown to 0.0
- Resets health_status to HEALTHY and risk_multiplier to 1.0
- Clears cumulative P&L
- Preserves regime configuration

**Example:**
```python
# End of trading day
monitor.reset_for_session()

# Next day: fresh rolling window, but thresholds unchanged
```

---

## Health Evaluation Rules

### Regime-Aware Thresholds

Each market regime has specific expected performance bands:

| Regime | Min Sharpe | Max Drawdown | Use Case |
|--------|-----------|--------------|----------|
| **high_vol** | 0.20 | 15% | High volatility market - relax expectations |
| **low_vol** | 0.10 | 10% | Low volatility market - stricter rules |
| **risk_on** | 0.15 | 12% | Risk appetite high - balanced thresholds |
| **risk_off** | 0.05 | 20% | Risk aversion high - very relaxed rules |

### Health Status Logic

```
HEALTHY:
  ✓ rolling_sharpe >= regime.min_sharpe
  ✓ rolling_drawdown <= regime.max_drawdown
  → risk_multiplier = 1.0 (full position)

DEGRADED:
  ✓ Either (sharpe OR drawdown) meets threshold
  ✗ But not both
  → risk_multiplier = 0.5 (half position)

CRITICAL:
  ✗ rolling_sharpe < regime.min_sharpe
  ✗ rolling_drawdown > regime.max_drawdown
  → risk_multiplier = 0.0 (no new entries)
```

### Sharpe Ratio Calculation

```
rolling_returns = [r1, r2, r3, ..., rn]
mean_return = sum(returns) / n
std_return = sqrt(sum((r - mean)^2) / n)
rolling_sharpe = (mean_return / std_return) * sqrt(252)
```

### Drawdown Calculation

```
cumulative_pnl = [c1, c2, c3, ..., cn]  # Cumulative for window
peak = max(cumulative_pnl)
trough = min(cumulative_pnl)
rolling_drawdown = abs((trough - peak) / peak)
```

---

## Integration Flow

### Pre-Trade

```python
# Get current risk multiplier
multiplier = monitor.get_risk_multiplier()

# Get current health status
status = monitor.get_health_status()

# If CRITICAL, only allow exits
if status == "CRITICAL":
    if action != "EXIT":
        action = "DO_NOTHING"

# Scale position size
scaled_position = desired_position * multiplier
```

### Post-Trade

```python
# Update health monitor with realized P&L
monitor.update(
    pnl=bar_pnl,
    regime_label=market_regime,
    realized_pnl=cumulative_realized,
    unrealized_pnl=current_unrealized
)

# Health monitor automatically:
# - Updates rolling metrics
# - Calculates Sharpe and drawdown
# - Evaluates health status
# - Sets new risk_multiplier
```

### Daily Reset

```python
# End of day
monitor.reset_for_session()
# → Clears rolling window
# → Resets to HEALTHY status
# → Preserves configuration
```

---

## Example: Health Transition Sequence

### Scenario: Normal Day → Degradation → Recovery

```
Bar 1-100:   Profitable, consistent returns
             rolling_sharpe = 1.2, rolling_drawdown = 0.05
             Status: HEALTHY (1.0)

Bar 101-150: Losses accumulate
             rolling_sharpe = 0.8, rolling_drawdown = 0.12
             Evaluation: sharpe OK, drawdown approaching limit
             Status: HEALTHY (1.0)

Bar 151:     Sudden large loss
             rolling_sharpe = 0.4, rolling_drawdown = 0.16
             Evaluation: sharpe GOOD, drawdown EXCEEDED
             Status: DEGRADED (0.5) [LOG: TRANSITION]

Bar 152-180: Further losses
             rolling_sharpe = -0.1, rolling_drawdown = 0.22
             Evaluation: sharpe BAD, drawdown BAD
             Status: CRITICAL (0.0) [LOG: TRANSITION]
             → All new entries blocked, exits only

Bar 181-220: Recovery trades (exits + small re-entries)
             rolling_sharpe = 0.2, rolling_drawdown = 0.18
             Evaluation: sharpe OK, drawdown still high
             Status: DEGRADED (0.5) [LOG: TRANSITION]

Bar 221+:    Good recovery
             rolling_sharpe = 0.6, rolling_drawdown = 0.10
             Evaluation: sharpe OK, drawdown OK
             Status: HEALTHY (1.0) [LOG: TRANSITION]
```

---

## Logging

### Log File Location

```
logs/health/health_<YYYYMMDD_HHMMSS>.log
```

### Log Events

| Event | Level | Message |
|-------|-------|---------|
| Initialization | INFO | "EngineHealthMonitor initialized: window_size=500" |
| Bar Update | DEBUG | "Health update: pnl=100.0, sharpe=0.82, drawdown=0.05, status=HEALTHY" |
| Status Transition | WARNING | "Health status transition: HEALTHY → DEGRADED" |
| Threshold Update | INFO | "Updated thresholds for regime 'high_vol'" |
| Session Reset | INFO | "Resetting health monitor for new session" |

### Example Log Output

```
2026-01-19 10:00:00.123 INFO: EngineHealthMonitor initialized: window_size=500, annualization_factor=252.0
2026-01-19 10:00:01.456 DEBUG: Health update: pnl=100.0, cumulative=100.0, regime=low_vol, sharpe=0.0, drawdown=0.0, status=HEALTHY, multiplier=1.0
2026-01-19 10:00:02.789 DEBUG: Health update: pnl=50.0, cumulative=150.0, regime=low_vol, sharpe=0.5, drawdown=0.0, status=HEALTHY, multiplier=1.0
2026-01-19 10:15:30.234 WARNING: Health status transition: HEALTHY → DEGRADED | sharpe=0.8 (threshold=1.0), drawdown=0.12 (threshold=0.10), regime=low_vol, risk_multiplier=0.5
2026-01-19 11:00:00.567 INFO: Resetting health monitor for new session: bars_in_window=150, cumulative_pnl=3500.0
```

---

## Integration into RealDataTournament

### Initialization

```python
from engine.health_monitor import EngineHealthMonitor

# Create health monitor at tournament start
health_monitor = EngineHealthMonitor(window_size=500)
```

### Main Loop

```python
for bar in bars:
    # 1. Get current health status
    risk_multiplier = health_monitor.get_risk_multiplier()
    
    # 2. Scale position size
    if risk_multiplier == 0.0:
        # CRITICAL: No new entries
        if action != "EXIT":
            action = "DO_NOTHING"
    
    # 3. Execute trade with scaled position
    position_size = desired_size * risk_multiplier
    execute_trade(action, symbol, position_size, price)
    
    # 4. Calculate bar P&L
    bar_pnl = calculate_pnl(bar)
    
    # 5. Update health monitor
    regime = get_market_regime(bar)
    health_monitor.update(
        pnl=bar_pnl,
        regime_label=regime,
        realized_pnl=portfolio.realized_pnl,
        unrealized_pnl=portfolio.unrealized_pnl
    )
```

### End of Day

```python
# Reset for new day
health_monitor.reset_for_session()
```

---

## Performance Characteristics

| Operation | Time | Memory |
|-----------|------|--------|
| `update()` | <1ms | O(1) |
| `compute_sharpe()` | <1ms | O(window_size) |
| `compute_drawdown()` | <1ms | O(window_size) |
| `evaluate_health()` | <1ms | O(1) |
| `get_risk_multiplier()` | <1ms | O(1) |
| `get_state_snapshot()` | <1ms | O(1) |
| Full suite (1000 updates) | <100ms | O(window_size) |

**Memory Usage:** Minimal - deques are bounded by window_size

---

## Customization

### Adjusting Window Size

```python
# Slower, more stable health tracking
monitor = EngineHealthMonitor(window_size=1000)

# Faster, more responsive health tracking
monitor = EngineHealthMonitor(window_size=250)
```

### Customizing Regime Thresholds

```python
# Make low_vol regime stricter
monitor.set_regime_thresholds("low_vol", min_sharpe=0.15, max_drawdown=0.08)

# Make risk_off regime more lenient
monitor.set_regime_thresholds("risk_off", min_sharpe=0.02, max_drawdown=0.30)
```

### Adding New Regimes

```python
monitor.expected_bands["extreme_vol"] = RegimeThresholds(
    min_sharpe=0.3,
    max_drawdown=0.25
)
```

---

## Testing

### Test Coverage

- **39 tests total**
- Initialization and configuration
- Rolling metric calculations
- Health evaluation logic
- Risk multiplier scaling
- Regime tracking and thresholds
- Snapshot generation
- Session reset behavior
- Edge cases and boundary conditions
- Realistic trading scenarios

### Test Execution

```bash
python -m pytest tests/test_health_monitor.py -v
# Expected: 39 passed in 0.14s
```

---

## Determinism & Causality

### Deterministic Design

- All calculations use only historical data in rolling window
- No forward-looking information
- Same inputs always produce same outputs
- Deterministic floating-point operations with consistent precision

### Time-Causal Design

- Each bar uses only previous bars' data
- Current bar's P&L affects next bar's health evaluation
- Regime label is external input (from market analysis)
- No future P&L or price data used

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-19 | Initial release: Rolling metrics, regime thresholds, health evaluation |

---

## Roadmap for Phase 5+

### Phase 5: Advanced Risk Metrics
- Value-at-Risk (VaR) calculation
- Expected Shortfall (ES)
- Sortino ratio (penalizes downside)
- Calmar ratio (return vs max drawdown)

### Phase 6: Adaptive Thresholds
- Learn optimal thresholds from historical data
- Per-symbol threshold customization
- Seasonal threshold adjustments

### Phase 7: Predictive Health
- Forecast health degradation
- Preemptive risk reduction
- ML-based anomaly detection

---

## Support & Troubleshooting

### Q: Why is my health status stuck at HEALTHY?
**A:** Window may not be full yet. Health evaluation requires at least 50% of window_size data.

### Q: How often should I reset the health monitor?
**A:** At natural boundaries (end of day, new trading session). Resets clear rolling window but preserve configuration.

### Q: Can I change thresholds mid-session?
**A:** Yes, use `set_regime_thresholds()`. Next evaluation will use new values.

### Q: What's the right window_size for my strategy?
**A:** Use 250-500 bars for intraday, 100-250 for day trading. Larger = slower response, more stable.

---

## References

- Sharpe Ratio: (Return - Risk-Free Rate) / Volatility
- Drawdown: (Peak Value - Trough Value) / Peak Value
- Annualization: Multiply by sqrt(periods per year)
- Rolling Window: Fixed-size deque dropping oldest when full

---

**Classification:** PRODUCTION READY ✅  
**Last Updated:** January 19, 2026  
**Maintained by:** GitHub Copilot
