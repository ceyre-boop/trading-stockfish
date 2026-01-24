# Phase 3 Quick Start Guide: PortfolioRiskManager + Governance

**For:** Integrating portfolio-level risk controls into RealDataTournament  
**Time:** 5 minutes to integrate  
**Experience Level:** Intermediate Python

---

## TL;DR - The Three Critical Checks

```python
# 1. Can we trade?
if not governance.can_trade():
    action = 'EXIT'  # Force exit

# 2. Can we open this position?
if not portfolio.can_open_position(symbol, size, price):
    return False  # Block position

# 3. Did we lose too much?
governance.evaluate(portfolio.daily_pnl)
if governance.force_flatten():
    exit_all_positions()
```

---

## 1. Basic Setup (2 minutes)

### Import

```python
from engine.portfolio_risk_manager import PortfolioRiskManager
from engine.governance import Governance
```

### Initialize at Tournament Start

```python
def tournament_start():
    # Create portfolio manager
    portfolio = PortfolioRiskManager(
        total_capital=100000,              # Account size
        max_symbol_exposure=25000,         # Max per symbol
        max_total_exposure=50000,          # Max total
        max_daily_loss=5000                # Daily loss limit
    )
    
    # Create governance
    governance = Governance(
        max_daily_loss=5000
    )
    
    return portfolio, governance
```

---

## 2. Pre-Trade Check (1 minute)

### Check Both Limits Before Opening Position

```python
def can_execute_trade(action, symbol, size, price, 
                      portfolio, governance):
    
    # Governance check first (higher priority)
    if not governance.can_trade() and action != 'EXIT':
        action = governance.override_action(action, symbol, reason='Halted')
        # action = 'EXIT' or 'DO_NOTHING'
    
    # Portfolio check (if action still permits entry)
    if action in ['ENTER', 'ADD']:
        if not portfolio.can_open_position(symbol, size, price):
            return False, None  # Position blocked
    
    return True, action
```

### Example

```python
# Try to enter ES long
can_trade, action = can_execute_trade(
    action='ENTER', 
    symbol='ES', 
    size=5, 
    price=4500,
    portfolio=portfolio,
    governance=governance
)

if can_trade:
    execute_trade(action, symbol, size, price)
else:
    print("Position blocked by risk limits")
```

---

## 3. Post-Trade Update (1 minute)

### Update After Every Trade

```python
def update_after_trade(symbol, size, price, 
                       realized_pnl, unrealized_pnl,
                       portfolio, governance):
    
    # Update exposure
    portfolio.update_exposure(symbol, size, price)
    
    # Update P&L
    portfolio.update_pnl(
        realized=realized_pnl,
        unrealized=unrealized_pnl
    )
    
    # Evaluate governance (check daily loss)
    governance.evaluate(portfolio.daily_pnl)
    
    # Check for force exit
    if governance.force_flatten():
        exit_all_positions()
        print(f"KILL SWITCH ACTIVATED: {governance.get_report()}")
```

### Example

```python
# After closing ES position with $250 profit
update_after_trade(
    symbol='ES',
    size=5,
    price=4505,
    realized_pnl=250,
    unrealized_pnl=0,
    portfolio=portfolio,
    governance=governance
)
```

---

## 4. Common Scenarios

### Scenario 1: Normal Trade

```python
# Step 1: Pre-trade check
ok, action = can_execute_trade('ENTER', 'ES', 5, 4500, portfolio, governance)
if not ok: return

# Step 2: Execute trade
execute_trade(action, 'ES', 5, 4500)

# Step 3: Post-trade update
update_after_trade('ES', 5, 4500, 
                   realized_pnl=0, unrealized_pnl=100,
                   portfolio=portfolio, governance=governance)

# Result: Position open, P&L tracked, all limits checked ✓
```

### Scenario 2: Position Blocked by Exposure

```python
# ES already has $22.5k exposure (5 contracts @ $4500)
# Try to add NQ with $26k exposure
ok, action = can_execute_trade('ENTER', 'NQ', 1, 13000, portfolio, governance)
# Result: ok=False (would exceed max_total_exposure of $50k)
# Trade blocked, position never entered ✓
```

### Scenario 3: Kill Switch Activated

```python
# Daily loss accumulates to -$5500
portfolio.update_pnl(realized=-5500, unrealized=0)
governance.evaluate(portfolio.daily_pnl)
# governance.kill_switch_triggered = True

# Try to enter new position
ok, action = can_execute_trade('ENTER', 'GC', 2, 2000, portfolio, governance)
# Result: action='EXIT' (governance override)
# Remaining positions will be exited ✓
```

### Scenario 4: End of Day Reset

```python
# End of trading day
governance.reset_session()
portfolio.reset_daily_limits()

# Tomorrow: fresh daily P&L, but kill switch persists
# (If triggered today, it prevents trading tomorrow too)
```

---

## 5. State Inspection

### Check Portfolio State

```python
snapshot = portfolio.get_state_snapshot()
print(f"Total Exposure: ${snapshot.current_total_exposure:,.0f}")
print(f"Available Capital: ${snapshot.available_capital:,.0f}")
print(f"Daily P&L: ${snapshot.daily_pnl:,.0f}")
print(f"Leverage: {snapshot.capital_utilization_percent:.1f}%")
```

### Check Governance State

```python
report = governance.get_report()
print(f"Kill Switch Active: {report['kill_switch_triggered']}")
print(f"Trading Allowed: {report['can_trade']}")
print(f"Total Decisions: {len(report['decision_history'])}")
```

---

## 6. Logging Setup (Optional)

### Create Logs Directories

```python
import os
from datetime import datetime

# Portfolio logs
os.makedirs('logs/portfolio', exist_ok=True)

# Governance logs
os.makedirs('logs/governance', exist_ok=True)

# Log files created automatically with timestamps
portfolio_log = f"logs/portfolio/portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
governance_log = f"logs/governance/governance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
```

---

## 7. Integration Points

### In `analytics/run_elo_evaluation.py`

```python
# At tournament initialization
portfolio, governance = tournament_start()

# In trade loop
ok, action = can_execute_trade(action, symbol, size, price, 
                               portfolio, governance)

# After trade execution
update_after_trade(symbol, size, price, 
                   realized_pnl, unrealized_pnl,
                   portfolio, governance)

# At tournament end
final_report = governance.get_report()
```

---

## 8. Important Rules

### ✅ DO

- ✅ Check `governance.can_trade()` BEFORE position entry
- ✅ Check `portfolio.can_open_position()` BEFORE position entry
- ✅ Call `update_exposure()` AFTER every trade
- ✅ Call `update_pnl()` AFTER every trade or P&L update
- ✅ Call `governance.evaluate()` AFTER updating P&L
- ✅ Call `reset_daily_limits()` at session start
- ✅ Preserve kill switch across session resets

### ❌ DON'T

- ❌ Ignore `can_execute_trade()` result
- ❌ Open positions exceeding limits
- ❌ Forget to update P&L
- ❌ Skip governance evaluation
- ❌ Try to reverse kill switch (irreversible)
- ❌ Reset kill switch state manually

---

## 9. Performance Tips

### Caching

```python
# Cache portfolio state if checking multiple times
state = portfolio.get_state_snapshot()

# Check multiple conditions against same state
if state.available_capital > 10000:
    # ...
```

### Batch Updates

```python
# Combine multiple P&L updates
realized_total = 500 + 250 - 100  # Sum all trades
unrealized_total = 75 + -50       # Sum all unrealized

portfolio.update_pnl(realized=realized_total, 
                    unrealized=unrealized_total)
```

---

## 10. Debugging Checklist

### Position Blocked? Check This:

```python
# 1. Check exposure per symbol
symbol_exposure = portfolio.current_exposure_per_symbol.get('ES', 0)
print(f"ES Exposure: ${symbol_exposure:,.0f} / {portfolio.max_symbol_exposure:,.0f}")

# 2. Check total exposure
print(f"Total: ${portfolio.current_total_exposure:,.0f} / {portfolio.max_total_exposure:,.0f}")

# 3. Check daily loss
print(f"Daily P&L: ${portfolio.daily_pnl:,.0f} vs Limit: {portfolio.max_daily_loss:,.0f}")

# 4. Check governance
print(f"Kill Switch: {governance.kill_switch_triggered}")
print(f"Can Trade: {governance.can_trade()}")
```

### Kill Switch Triggered? Check This:

```python
report = governance.get_report()

# When triggered
print(f"Trigger Time: {report['trigger_time']}")
print(f"Trigger Reason: {report['trigger_reason']}")

# Decision history
for decision in report['decision_history'][-3:]:
    print(f"{decision['timestamp']}: {decision['reason']}")
```

---

## 11. Example: Complete Trading Day

```python
from engine.portfolio_risk_manager import PortfolioRiskManager
from engine.governance import Governance

# 1. Start of day
portfolio = PortfolioRiskManager(
    total_capital=100000,
    max_symbol_exposure=25000,
    max_total_exposure=50000,
    max_daily_loss=5000
)
governance = Governance(max_daily_loss=5000)

# 2. Morning: Trade ES
ok, _ = can_execute_trade('ENTER', 'ES', 5, 4500, portfolio, governance)
if ok:
    portfolio.update_exposure('ES', 5, 4500)
    portfolio.update_pnl(realized=0, unrealized=100)
    governance.evaluate(portfolio.daily_pnl)

# 3. Midday: Add NQ
ok, _ = can_execute_trade('ENTER', 'NQ', 1, 13000, portfolio, governance)
if ok:
    portfolio.update_exposure('NQ', 1, 13000)
    portfolio.update_pnl(realized=0, unrealized=50)
    governance.evaluate(portfolio.daily_pnl)

# 4. Afternoon: Close ES (loss)
ok, _ = can_execute_trade('EXIT', 'ES', 5, 4490, portfolio, governance)
if ok:
    portfolio.update_exposure('ES', -5, 4490)  # Close position
    portfolio.update_pnl(realized=-50, unrealized=0)
    governance.evaluate(portfolio.daily_pnl)

# 5. Late afternoon: Markets tank, accumulate losses
# Daily P&L hits -$5500
portfolio.update_pnl(realized=-4800, unrealized=-700)
governance.evaluate(portfolio.daily_pnl)

# Kill switch triggered!
if governance.force_flatten():
    print("EMERGENCY: Exiting all positions")
    # Close NQ position
    portfolio.update_exposure('NQ', -1, 12900)

# 6. End of day
portfolio.reset_daily_limits()
governance.reset_session()

print(f"Day Summary: ${portfolio.daily_pnl:,.0f}")
print(f"Kill Switch Active: {governance.kill_switch_triggered}")
```

---

## Quick Reference

| Method | When | Purpose |
|--------|------|---------|
| `portfolio.can_open_position()` | Before ENTER | Check if position allowed |
| `governance.can_trade()` | Before ANY action | Check if trading halted |
| `portfolio.update_exposure()` | After trade | Track new exposure |
| `portfolio.update_pnl()` | After P&L change | Track daily loss |
| `governance.evaluate()` | After P&L update | Check daily loss limit |
| `governance.force_flatten()` | When halted | Check if must exit |
| `portfolio.reset_daily_limits()` | Session start | Reset daily P&L |
| `governance.reset_session()` | Session end | Reset history |

---

**For Complete Details:** See `PORTFOLIO_RISK_MANAGER_SPEC.md` and `GOVERNANCE_SPEC.md`
