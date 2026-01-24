# PortfolioRiskManager Specification

**Version:** 1.0  
**Date:** 2026-01-19  
**Status:** Production  
**Component:** `engine/portfolio_risk_manager.py`

---

## Overview

The `PortfolioRiskManager` is a portfolio-level risk control system that enforces capital allocation limits, position sizing constraints, and daily loss thresholds. It prevents over-leverage, concentration risk, and catastrophic capital drawdowns.

### Key Features

✅ **Per-Symbol Exposure Limits** - Maximum $ exposure per trading symbol  
✅ **Total Portfolio Limits** - Maximum $ total exposure across all positions  
✅ **Daily Loss Limits** - Force exit if daily loss exceeds threshold  
✅ **Dynamic Position Sizing** - Reject positions exceeding available capital  
✅ **Real-time P&L Tracking** - Separate realized and unrealized P&L  
✅ **Capital Utilization** - Monitor portfolio leverage and available capital  
✅ **Deterministic State** - All operations side-effect-free except explicit updates

---

## Architecture

### Attributes

```python
class PortfolioRiskManager:
    total_capital: float                           # Total capital available
    max_symbol_exposure: float                     # Max per symbol (e.g., $25,000)
    max_total_exposure: float                      # Max total (e.g., $50,000)
    max_daily_loss: float                          # Max daily loss (e.g., $5,000)
    
    # State tracking
    current_exposure_per_symbol: Dict[str, float]  # Symbol → exposure mapping
    current_total_exposure: float                  # Sum of all exposures
    daily_pnl: float                               # Daily P&L (realized + unrealized)
    unrealized_pnl: float                          # Open position P&L
    realized_pnl: float                            # Closed position P&L
```

### Core Methods

#### `update_exposure(symbol, position_size, price)`

Update exposure tracking for a symbol.

```python
portfolio.update_exposure('ES', 5, 4500)  # 5 contracts @ $4,500 = $22,500 exposure

# Side effects:
# - Updates current_exposure_per_symbol['ES'] = 22,500
# - Updates current_total_exposure (sum of all exposures)
```

**Rules:**
- If `position_size == 0`: Close position, exposure → 0
- If `position_size > 0`: Update exposure to `abs(size * price)`
- Exposure = absolute value (long and short both count)

#### `update_pnl(realized, unrealized)`

Track P&L from trading activity.

```python
portfolio.update_pnl(realized=1000, unrealized=-500)
# daily_pnl = 1000 + (-500) = 500

portfolio.update_pnl(realized=-3000, unrealized=-2000)
# daily_pnl = -3000 + (-2000) = -5000
```

**Rules:**
- `realized`: P&L from closed positions
- `unrealized`: P&L from open positions
- `daily_pnl = realized + unrealized`

#### `can_open_position(symbol, target_size, price)` → bool

Check if position can be opened given current limits.

```python
can_open = portfolio.can_open_position('ES', 5, 4500)

# Returns True if ALL conditions met:
# 1. Position exposure < max_symbol_exposure
# 2. Total exposure (with new position) < max_total_exposure
# 3. Daily loss > -max_daily_loss (i.e., loss hasn't reached limit)

# Returns False if ANY condition violated
```

**Example:**
```python
portfolio = PortfolioRiskManager(
    total_capital=100000,
    max_symbol_exposure=25000,     # Max $25k per symbol
    max_total_exposure=50000,      # Max $50k total
    max_daily_loss=5000            # Max $5k daily loss
)

# ES position: 5 @ $4,500 = $22,500 (OK - under $25k limit)
portfolio.can_open_position('ES', 5, 4500)  # → True

# ES position: 10 @ $3,000 = $30,000 (BLOCKED - exceeds $25k limit)
portfolio.can_open_position('ES', 10, 3000)  # → False
```

#### `should_force_exit()` → bool

Check if daily loss threshold exceeded (force exit all positions).

```python
portfolio.update_pnl(realized=-5500, unrealized=0)
portfolio.should_force_exit()  # → True (exceeds $5,000 limit)

portfolio.update_pnl(realized=-3000, unrealized=-1500)
portfolio.should_force_exit()  # → False (-4,500 still within limit)
```

#### `reset_daily_limits()`

Reset daily P&L tracking at session boundaries.

```python
portfolio.update_pnl(realized=-2000, unrealized=-500)
# daily_pnl = -2,500

portfolio.reset_daily_limits()
# daily_pnl = 0
# realized_pnl = 0
# unrealized_pnl = 0
# Exposure tracking PRESERVED
```

#### `flatten_all_positions()`

Clear all position tracking (for force exit scenarios).

```python
portfolio.update_exposure('ES', 5, 4500)
portfolio.update_exposure('NQ', 2, 15000)
# current_total_exposure = $52,500

portfolio.flatten_all_positions()
# current_exposure_per_symbol = {}
# current_total_exposure = 0
```

#### `get_available_capital()` → float

Calculate remaining capital for new positions.

```python
portfolio = PortfolioRiskManager(..., max_total_exposure=50000)
portfolio.update_exposure('ES', 5, 4500)  # $22,500 used

available = portfolio.get_available_capital()
# → 50,000 - 22,500 = 27,500
```

#### `get_capital_utilization_percent()` → float

Get portfolio leverage as percentage.

```python
utilization = portfolio.get_capital_utilization_percent()
# → (current_total_exposure / max_total_exposure) * 100
# → (22,500 / 50,000) * 100 = 45%
```

#### `get_state_snapshot()` → PortfolioState

Get immutable snapshot of portfolio state.

```python
snapshot = portfolio.get_state_snapshot()
# PortfolioState(
#     total_capital=100000,
#     current_exposure_per_symbol={'ES': 22500},
#     current_total_exposure=22500,
#     daily_pnl=500,
#     realized_pnl=1000,
#     unrealized_pnl=-500,
#     timestamp=datetime(...)
# )
```

---

## Exposure Rules

### Exposure Calculation

$$\text{Exposure} = |\text{Position Size}| \times \text{Price}$$

Both long and short positions count toward exposure limits.

```python
# Long position: 5 ES @ $4,500 = $22,500
portfolio.update_exposure('ES', 5, 4500)   # Exposure: $22,500

# Short position: -3 ES @ $4,500 = $13,500
portfolio.update_exposure('ES', -3, 4500)  # Exposure: $13,500

# Closed position: 0 ES
portfolio.update_exposure('ES', 0, 4500)   # Exposure: $0
```

### Position Sizing Constraints

**Check 1: Symbol Limit**
```
position_exposure <= max_symbol_exposure
```

**Check 2: Total Limit**
```
(current_total_exposure - old_symbol_exposure + new_exposure) <= max_total_exposure
```

**Check 3: Daily Loss Limit**
```
daily_pnl >= -max_daily_loss
```

If ANY check fails, position is rejected.

### Portfolio Balance Example

```python
portfolio = PortfolioRiskManager(
    total_capital=100000,
    max_symbol_exposure=30000,
    max_total_exposure=60000,
    max_daily_loss=6000
)

# Portfolio progression
portfolio.update_exposure('ES', 5, 4500)      # ES: $22.5k → Total: $22.5k
portfolio.update_exposure('NQ', 2, 15000)    # NQ: $30k → Total: $52.5k
portfolio.update_exposure('GC', 10, 2000)    # GC: $20k → Total: $72.5k (BLOCKED)

# Available capital
portfolio.get_available_capital()             # 60k - 52.5k = $7.5k
portfolio.get_capital_utilization_percent()  # 52.5k / 60k = 87.5%
```

---

## P&L Tracking

### Daily P&L Components

```
Daily P&L = Realized P&L + Unrealized P&L

Realized P&L
  └─ P&L from positions closed today
  └─ Includes entry/exit fills and commissions

Unrealized P&L
  └─ P&L from open positions
  └─ Mark-to-market based on current prices
```

### P&L Update Flow

```python
# Trade 1: Buy 5 ES @ 4500
portfolio.update_exposure('ES', 5, 4500)
portfolio.update_pnl(realized=0, unrealized=0)

# Price moves to 4520
portfolio.update_pnl(realized=0, unrealized=100)  # 5 * 20 = $100 gain

# Exit half (2.5 contracts) at 4520
portfolio.update_pnl(realized=100, unrealized=50)  # 2.5 * 20 = $50 gain

# Close remaining at 4530
portfolio.update_pnl(realized=175, unrealized=0)  # 2.5 * 10 = $25 gain
```

---

## Daily Loss Enforcement

### Force Exit Triggers

Daily loss force exit when:

$$\text{Daily Loss} < -\text{max\_daily\_loss}$$

```python
portfolio.max_daily_loss = 5000
portfolio.daily_pnl = -5500  # Exceeds -$5,000

portfolio.should_force_exit()  # → True
```

### Integration with Risk Controls

When `should_force_exit()` returns True:

1. ✅ Immediately exit all positions
2. ✅ Stop accepting new entries
3. ✅ Allow only exit/reduce actions
4. ✅ Clear all exposure tracking
5. ✅ Log force exit event

### Reset Daily Limits

Daily limits reset at:
- Session start
- Market open
- Manually via `reset_daily_limits()`

Exposure tracking is NOT cleared (positions continue).

---

## Integration with Trading Engine

### Pre-Trade Check

```python
def check_trade_allowed(symbol, size, price):
    # Check portfolio limits
    if not portfolio.can_open_position(symbol, size, price):
        return False, "Exposure limit exceeded"
    
    # Check force exit
    if portfolio.should_force_exit():
        return False, "Daily loss limit exceeded"
    
    return True, "OK"
```

### Post-Trade Update

```python
def execute_trade(symbol, size, price, pnl):
    # Update exposure
    portfolio.update_exposure(symbol, size, price)
    
    # Update P&L
    portfolio.update_pnl(realized=pnl.realized, unrealized=pnl.unrealized)
    
    # Check for force exit
    if portfolio.should_force_exit():
        trigger_force_exit_all()
```

### Session Reset

```python
def start_trading_session():
    portfolio.reset_daily_limits()
    portfolio.current_exposure_per_symbol.clear()
    portfolio.current_total_exposure = 0
    portfolio.session_start_time = datetime.now()
```

---

## Example Usage

### Scenario 1: Normal Day with Multiple Trades

```python
portfolio = PortfolioRiskManager(
    total_capital=100000,
    max_symbol_exposure=25000,
    max_total_exposure=50000,
    max_daily_loss=5000
)

# 09:30 - Market open
portfolio.reset_daily_limits()

# 10:00 - Open ES position
if portfolio.can_open_position('ES', 5, 4500):
    portfolio.update_exposure('ES', 5, 4500)
    portfolio.update_pnl(realized=0, unrealized=0)

# 11:00 - Price moves
portfolio.update_pnl(realized=0, unrealized=250)

# 14:00 - Open NQ position
if portfolio.can_open_position('NQ', 2, 15000):
    portfolio.update_exposure('NQ', 2, 15000)
    portfolio.update_pnl(realized=250, unrealized=100)

# 15:00 - Close ES
portfolio.update_exposure('ES', 0, 4505)
portfolio.update_pnl(realized=275, unrealized=100)

# 16:00 - Session end
print(f"Daily P&L: ${portfolio.daily_pnl:.2f}")
print(f"Positions: {portfolio.current_exposure_per_symbol}")
```

### Scenario 2: Force Exit Trigger

```python
portfolio = PortfolioRiskManager(..., max_daily_loss=5000)

# Mount losses
portfolio.update_pnl(realized=-2000, unrealized=-1000)
print(f"Loss: ${portfolio.daily_pnl:.2f}")  # -$3,000

portfolio.update_pnl(realized=-5500, unrealized=-500)
print(f"Loss: ${portfolio.daily_pnl:.2f}")  # -$6,000

# Force exit triggered
if portfolio.should_force_exit():
    # Exit all positions
    portfolio.flatten_all_positions()
    print("Force exit executed")
```

---

## Logging Integration

### Portfolio Events Logged

```
[ExposureUpdate] Symbol: ES, Size: 5, Price: 4500.00, Exposure: $22500.00, Total: $22500.00
[PnLUpdate] Realized: $0.00, Unrealized: $250.00, Daily: $250.00
[PositionBlocked] ES: Exposure $30000.00 exceeds symbol limit $25000.00
[ForceExit] Daily loss $6000.00 exceeds threshold $5000.00
[FlattenAll] All positions forcefully closed
```

---

## Performance Characteristics

| Operation | Time | Side Effects |
|-----------|------|--------------|
| `can_open_position()` | < 1ms | None |
| `update_exposure()` | < 1ms | Updates state |
| `update_pnl()` | < 1ms | Updates state |
| `should_force_exit()` | < 1ms | None |
| `get_state_snapshot()` | < 1ms | None |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-19 | Initial release: Core exposure tracking, daily loss limits, position blocking |

---

## Related Documentation

- [Governance Specification](GOVERNANCE_SPEC.md)
- [RealDataTournament Integration](../analytics/README.md)
- [Testing Guide](../tests/README.md)
