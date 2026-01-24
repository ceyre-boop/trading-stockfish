# Governance Specification

**Version:** 1.0  
**Date:** 2026-01-19  
**Status:** Production  
**Component:** `engine/governance.py`

---

## Overview

The `Governance` system is a meta-level risk officer that enforces global trading halts based on daily loss thresholds. Once activated, the kill switch halts all new trading for the remainder of the session and forces exit of all open positions.

### Key Features

✅ **Kill Switch Logic** - Irreversible trading halt when thresholds exceeded  
✅ **Global Risk Control** - Single source of truth for trading permission  
✅ **Audit Trail** - Complete decision history with timestamps  
✅ **Action Override** - Automatic conversion of trading actions  
✅ **Session Persistence** - Kill switch state preserved across resets  
✅ **Deterministic State** - All operations side-effect-free except explicit updates

---

## Architecture

### Attributes

```python
class Governance:
    max_daily_loss: float              # Maximum daily loss before kill switch
    kill_switch_triggered: bool        # Irreversible halt flag
    trigger_time: Optional[datetime]   # When kill switch was activated
    trigger_reason: Optional[str]      # Why it was triggered
    decision_history: List[dict]       # Audit trail of all decisions
```

### Core Methods

#### `evaluate(daily_loss)` 

Evaluate daily loss and trigger kill switch if needed.

```python
governance.evaluate(daily_loss=-3000)
# No action - within limit

governance.evaluate(daily_loss=-5500)
# Kill switch activated (exceeds -$5,000 limit)
```

**Rules:**
- If `|daily_loss| > max_daily_loss`: Trigger kill switch
- If kill switch already active: No change (irreversible)
- All evaluations logged to decision history

#### `can_trade()` → bool

Check if trading is permitted.

```python
if governance.can_trade():
    # OK to trade
else:
    # Kill switch active - halt all trading
```

**Rules:**
- Returns `True` if kill switch NOT triggered
- Returns `False` if kill switch IS triggered

#### `force_flatten()` → bool

Check if all positions should be forcefully closed.

```python
if governance.force_flatten():
    # Exit all positions immediately
```

**Rules:**
- Returns `True` if kill switch triggered
- Returns `False` if kill switch not triggered

#### `override_action(action, symbol, reason)` → str

Override trading action if governance constraints violated.

```python
action = 'ENTER'
overridden = governance.override_action(action, 'ES', 'Open position')

if governance.kill_switch_triggered:
    overridden = 'EXIT'  # Convert to exit
```

**Rules:**
- If trading allowed: Return action unchanged
- If kill switch active: Return 'EXIT' or 'DO_NOTHING'
- All overrides logged

#### `reset_session()`

Reset governance for new trading session.

```python
governance.reset_session()
# Clears decision history
# Preserves kill switch state (cannot be reset mid-session)
```

**Rules:**
- Decision history cleared
- max_daily_loss preserved
- Kill switch state persists if currently active
- Used for session transitions only

#### `get_state()` → GovernanceState

Get immutable snapshot of governance state.

```python
state = governance.get_state()
# GovernanceState(
#     max_daily_loss=5000,
#     kill_switch_triggered=True,
#     trigger_time=datetime(...),
#     trigger_reason="Daily loss $5500 exceeds $5000"
# )
```

#### `get_report()` → dict

Get comprehensive governance report.

```python
report = governance.get_report()
# {
#     'max_daily_loss': 5000,
#     'kill_switch_triggered': True,
#     'trigger_time': datetime(...),
#     'trigger_reason': "...",
#     'decisions_made': 45,
#     'is_trading_halted': True,
#     'decision_history': [...]
# }
```

---

## Kill Switch Logic

### Activation Criteria

Kill switch activates when:

$$\text{Daily Loss} > \text{max\_daily\_loss}$$

```python
governance.max_daily_loss = 5000

governance.evaluate(daily_loss=-3000)    # OK
governance.evaluate(daily_loss=-5000)    # OK (at limit)
governance.evaluate(daily_loss=-5001)    # TRIGGER kill switch
governance.evaluate(daily_loss=-10000)   # Still triggered (irreversible)
```

### Irreversibility

Once activated, kill switch cannot be reversed:

```python
governance.evaluate(daily_loss=-5500)  # Trigger
assert governance.kill_switch_triggered == True

governance.evaluate(daily_loss=-2000)  # Better, but...
assert governance.kill_switch_triggered == True  # Still active!
```

### Trading Halt Effects

When kill switch activates:

1. ✅ `can_trade()` returns False
2. ✅ `force_flatten()` returns True
3. ✅ All actions converted to EXIT/DO_NOTHING
4. ✅ New positions blocked
5. ✅ Existing positions allowed to be closed
6. ✅ Marked in decision history with timestamp

---

## Action Override Rules

### Override Scenarios

**Scenario 1: Trading Active (kill switch off)**
```python
governance.evaluate(daily_loss=-2000)  # Within limits

action = governance.override_action('ENTER', 'ES', 'Open')
# → 'ENTER' (unchanged)
```

**Scenario 2: Kill Switch Active**
```python
governance.evaluate(daily_loss=-5500)  # Trigger kill switch

# Entry attempt
action = governance.override_action('ENTER', 'ES', 'Open')
# → 'EXIT' or 'DO_NOTHING' (blocked)

# Add attempt
action = governance.override_action('ADD', 'ES', 'Add')
# → 'EXIT' or 'DO_NOTHING' (blocked)

# Exit attempt
action = governance.override_action('EXIT', 'ES', 'Close')
# → 'EXIT' (allowed - or DO_NOTHING for safety)
```

### Action Override Mapping

| Action | Kill Switch Off | Kill Switch On |
|--------|-----------------|----------------|
| ENTER | ENTER | EXIT/DO_NOTHING |
| ADD | ADD | EXIT/DO_NOTHING |
| REDUCE | REDUCE | REDUCE (allowed) |
| EXIT | EXIT | EXIT (allowed) |
| REVERSE | REVERSE | EXIT/DO_NOTHING |

---

## Decision History & Audit Trail

### Decision Record Structure

```python
{
    'timestamp': datetime(2026, 1, 19, 14, 32, 15),
    'daily_loss': -5500,
    'absolute_loss': 5500,
    'max_allowed': 5000,
    'kill_switch_triggered_before': False,
    'kill_switch_triggered_after': True,
    'action': 'KILL_SWITCH_ACTIVATED'
}
```

### Decision History Examples

```python
# Normal day - no triggers
governance.evaluate(daily_loss=-1000)
governance.evaluate(daily_loss=-2000)
governance.evaluate(daily_loss=-3000)
# decision_history: 3 records, no activation

# Bad day - triggers kill switch
governance.evaluate(daily_loss=-1000)
governance.evaluate(daily_loss=-3000)
governance.evaluate(daily_loss=-5500)
# decision_history: 3 records, third has action='KILL_SWITCH_ACTIVATED'
```

### Retrieving History

```python
# Last 10 decisions
history = governance.decision_history[-10:]

# Find trigger event
trigger_decision = next(
    d for d in governance.decision_history 
    if d['action'] == 'KILL_SWITCH_ACTIVATED'
)

print(f"Kill switch activated at {trigger_decision['timestamp']}")
print(f"Daily loss was ${trigger_decision['daily_loss']}")
```

---

## Integration with Portfolio Manager

### Trading Flow with Governance

```
1. Request trade (ENTER, ADD, REDUCE, EXIT)
   │
   ├─→ Check governance.can_trade()
   │   ├─ True: Proceed to step 2
   │   └─ False: Override to EXIT/DO_NOTHING
   │
   ├─→ Check portfolio.can_open_position()
   │   ├─ True: Execute trade
   │   └─ False: Block trade
   │
   ├─→ Execute trade
   │
   ├─→ Update portfolio exposure
   │
   ├─→ Update portfolio P&L
   │
   └─→ Governance.evaluate(portfolio.daily_pnl)
       ├─ If trigger: Kill switch activated
       └─ Portfolio blocks all future entries
```

### Example Integration

```python
portfolio = PortfolioRiskManager(...)
governance = Governance(max_daily_loss=5000)

def execute_trade(action, symbol, size, price):
    # Step 1: Check governance
    if not governance.can_trade() and action != 'EXIT':
        action = 'EXIT'  # Override to exit
    
    # Step 2: Check portfolio
    if not portfolio.can_open_position(symbol, size, price):
        return False, "Position blocked by portfolio limits"
    
    # Step 3: Execute trade
    # ... trade execution logic ...
    
    # Step 4: Update portfolio
    portfolio.update_exposure(symbol, size, price)
    portfolio.update_pnl(realized=trade_pnl, unrealized=0)
    
    # Step 5: Evaluate governance
    governance.evaluate(portfolio.daily_pnl)
    
    # Step 6: Check for force exit
    if governance.force_flatten():
        execute_force_exit_all()
    
    return True, "Trade executed"
```

---

## Session Management

### New Session Startup

```python
governance = Governance(max_daily_loss=5000)
# Kill switch off, ready for trading
```

### Session Reset

```python
# End of trading session
governance.reset_session()
# → Clears decision_history
# → Preserves max_daily_loss
# → Kill switch state PERSISTS (cannot reset mid-session)
```

### Multi-Day Tracking

```python
# Day 1
governance.evaluate(daily_loss=-5500)  # Kill switch activated
print(governance.can_trade())  # False

# Day 2 (next session)
governance = Governance(max_daily_loss=5000)  # New instance
print(governance.can_trade())  # True (fresh start)
```

---

## Logging & Reporting

### Governance Events Logged

```
[KillSwitch] ACTIVATED: Daily loss $5500.00 exceeds $5000.00
[ActionOverride] ES: ENTER → EXIT (kill switch active)
[GovernanceReset] Kill switch active for 3600.5s, reason: ...
```

### Report Examples

**Normal Day Report**
```python
report = governance.get_report()
# {
#     'max_daily_loss': 5000,
#     'kill_switch_triggered': False,
#     'trigger_time': None,
#     'trigger_reason': None,
#     'decisions_made': 15,
#     'is_trading_halted': False,
#     'decision_history': [...]
# }
```

**Triggered Kill Switch Report**
```python
report = governance.get_report()
# {
#     'max_daily_loss': 5000,
#     'kill_switch_triggered': True,
#     'trigger_time': datetime(2026, 1, 19, 14, 32, 15),
#     'trigger_reason': "Daily loss $5500.00 exceeds $5000.00",
#     'decisions_made': 42,
#     'is_trading_halted': True,
#     'decision_history': [...]
# }
```

---

## Example Usage

### Scenario 1: Normal Trading Day

```python
governance = Governance(max_daily_loss=5000)

# 10:00 - First trade
governance.evaluate(daily_loss=-500)
assert governance.can_trade() == True

# 11:00 - More trades
governance.evaluate(daily_loss=-1500)
assert governance.can_trade() == True

# 14:00 - Still OK
governance.evaluate(daily_loss=-3000)
assert governance.can_trade() == True

# 15:30 - End of day
print(f"Trading: {governance.can_trade()}")  # True
print(f"Kill switch: {governance.kill_switch_triggered}")  # False
```

### Scenario 2: Catastrophic Loss Day

```python
governance = Governance(max_daily_loss=5000)

# 11:00 - Some losses
governance.evaluate(daily_loss=-2000)
print(f"Status: {governance.can_trade()}")  # True

# 13:00 - Mounting losses
governance.evaluate(daily_loss=-4000)
print(f"Status: {governance.can_trade()}")  # True

# 14:00 - THRESHOLD EXCEEDED
governance.evaluate(daily_loss=-5500)
print(f"Status: {governance.can_trade()}")  # False
print(f"Kill switch: {governance.kill_switch_triggered}")  # True

# 14:01 - Try to trade
action = governance.override_action('ENTER', 'ES', 'Open')
print(f"Action: {action}")  # 'EXIT' or 'DO_NOTHING'

# 14:05 - Better P&L
governance.evaluate(daily_loss=-4500)
print(f"Status: {governance.can_trade()}")  # Still False (irreversible!)
```

---

## Performance Characteristics

| Operation | Time | Side Effects |
|-----------|------|--------------|
| `can_trade()` | < 1µs | None |
| `force_flatten()` | < 1µs | None |
| `evaluate()` | < 1ms | Updates state + history |
| `override_action()` | < 1ms | Logs override |
| `get_report()` | < 1ms | None |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-19 | Initial release: Kill switch logic, irreversibility, action override |

---

## Related Documentation

- [Portfolio Risk Manager](PORTFOLIO_RISK_MANAGER_SPEC.md)
- [RealDataTournament Integration](../analytics/README.md)
- [Testing Guide](../tests/README.md)
