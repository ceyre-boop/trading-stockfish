# ExecutionSimulator v1 - Realistic Trade Execution Model

## Overview

ExecutionSimulator v1 is a **minimal, pessimistic execution model** that prevents fantasy PnL and ensures honest ELO ratings for the trading engine. It routes all trades through realistic spread, slippage, and cost models based on market conditions.

**Key Principle:** Trades fill worse than the trader hopes, never better. This ensures conservative, realistic performance measurement.

---

## Architecture

### Input Model

```
Trade Execution Request:
  - action (ENTER/ADD/REDUCE/EXIT/REVERSE)
  - target_size (desired quantity)
  - mid_price (current market price)
  - liquidity_state (volume, depth, ATR)
  - volatility_state (current ATR, percentile, regime)
  - symbol (ES/NQ/EURUSD)
  - current_position (optional, for position state tracking)
```

### Processing Pipeline

```
1. Spread Calculation
   - Base spread (per symbol, fixed)
   - Volatility adjustment (spread widens in high vol)
   - Result: bid_price, ask_price

2. Slippage Calculation
   - Formula: k * ATR * (trade_size / liquidity_scale) * pessimism_factor
   - Pessimistic: always against trader
   - Scales with: trade size, volatility, low liquidity

3. Fill Price Determination
   - BUY: ask + slippage (worse than ask)
   - SELL: bid - slippage (worse than bid)
   - Rounded to tick size

4. Partial Fill Check
   - If liquidity is low: only fill fraction of target
   - fill_ratio = 0.8 (80% when constrained)
   - Return: actual_filled_size, constrained_flag

5. Cost Calculation
   - Commission: per-contract * trade_size * 2 (round-trip)
   - Covers entry + exit execution costs

6. Position Update
   - Update position state with actual fill
   - Track: side, quantity, entry_price, entry_cost
   - Calculate: unrealized P&L, realized P&L
```

### Output Model

```
ExecutionResult:
  - action: trade action taken
  - target_size: originally requested quantity
  - actual_filled_size: what actually filled
  - fill_price: actual fill price (pessimistic)
  - spread: bid-ask spread at execution
  - slippage: execution slippage vs mid
  - transaction_cost: commission + fees
  - total_cost: slippage + transaction_cost
  - liquidity_constraint_applied: was size reduced?
  - filled_percentage: (actual / target) * 100%
  - updated_position: new PositionState
```

---

## Configuration (execution_config.yaml)

### Symbol Configuration

Per-symbol execution parameters. Example for ES (E-mini S&P 500):

```yaml
symbols:
  ES:
    description: "E-mini S&P 500 Futures"
    tick_size: 0.25              # Minimum price movement
    fixed_spread: 1.0            # Fixed bid-ask in ticks
    spread_volatility_scale: 0.5 # Spread = fixed + (scale * vol%)
    slippage_coefficient: 0.15   # k in slippage formula
    commission_per_contract: 1.5 # USD per contract
    liquidity_scale: 1000.0      # Scale for partial fills
```

### Slippage Parameters

Global slippage configuration:

```yaml
slippage:
  pessimism_factor: 1.0         # Multiplier (>1.0 = more pessimistic)
  min_slippage: 0.0             # Minimum slippage magnitude
```

### Partial Fill Logic

Low-liquidity handling:

```yaml
partial_fills:
  enabled: true                  # Enable partial fills
  low_liquidity_threshold: 0.3  # Trigger when vol/ATR/scale < 0.3
  fill_ratio_base: 0.8          # Fill 80% of target when constrained
```

---

## Spread Model

### Simple, Volatility-Adjusted

```python
spread = fixed_spread + (spread_volatility_scale * volatility_percentile / 100)

# Example:
# ES: fixed_spread = 1.0 tick, volatility_scale = 0.5
# At 50th percentile: spread = 1.0 + 0.5 * 0.5 = 1.25 ticks
# At 100th percentile: spread = 1.0 + 0.5 * 1.0 = 1.5 ticks
```

**Why:** In volatile markets, spreads widen (more risk for market makers). In quiet markets, spreads tighten.

---

## Slippage Model

### Pessimistic, Size-Dependent

```python
slippage = k * ATR * (trade_size / liquidity_scale) * pessimism_factor

# Example (ES):
# k = 0.15, ATR = 20 points, trade_size = 10, liquidity_scale = 1000
# slippage = 0.15 * 20 * (10 / 1000) * 1.0 = 0.03 points = 0.12 ticks
```

**Why:** 
- Larger trades hit worse slippage (move the market more)
- High volatility (high ATR) = worse slippage
- Low liquidity (small scale) = worse slippage

---

## Cost Model

### Commission Only (Simple)

```python
transaction_cost = contract_commission * trade_size * 2

# Assumes round-trip: entry + exit
# Example (ES): 1.5 USD per contract, 10 contracts
# Total cost = 1.5 * 10 * 2 = $30
```

**Why:** Covers exchange fees and clearing costs at entry and exit.

---

## Partial Fills

### Low Liquidity Constraint

If market liquidity is too low (volume/ATR < threshold), fills are reduced:

```python
if liquidity_metric < low_liquidity_threshold:
    actual_size = target_size * fill_ratio_base
    liquidity_constrained = True
else:
    actual_size = target_size
    liquidity_constrained = False
```

**Benefit:** Prevents unrealistic large fills in thin markets.

---

## Position State Tracking

### ENTER Action
Opens new position with actual fill price and costs.

### ADD Action
Increases position, averaging entry price:
```python
new_entry_price = (old_price * old_qty + fill_price * new_qty) / (old_qty + new_qty)
```

### REDUCE Action
Decreases position, realizes P&L:
```python
realized_pnl = (mark_price - entry_price) * filled_size * direction - cost
```

### EXIT Action
Closes entire position, realizes full P&L.

### REVERSE Action
Closes position + opens opposite:
```python
exit_pnl = (mark_price - entry_price) * qty * direction
entry_cost_for_new = commission for new position
new_position = opposite_side with fill_price
```

---

## Execution Logging

Every trade logs:

```
[Symbol] Action=ENTER | Target=10 -> Filled=10 (100%) | 
FillPrice=4500.75 | Mid=4500.0 | Spread=1.25 | Slippage=0.75 | 
Cost=30.00 | TotalCost=30.75 | Liquidity=Vol1000/min ATR20 | 
Volatility=50pct Regime=moderate | ConstrainedFill=False
```

**Logged to:** `logs/execution_<timestamp>.log`

---

## Usage Example

### Basic Initialization

```python
from engine.execution_simulator import (
    ExecutionSimulator, LiquidityState, VolatilityState
)

executor = ExecutionSimulator(config_path="execution_config.yaml")
```

### Simulate a Trade

```python
liquidity = LiquidityState(
    volume_per_minute=1000,
    bid_size=500,
    ask_size=500,
    typical_atr=20.0
)

volatility = VolatilityState(
    current_atr=20.0,
    volatility_percentile=50,
    regime='moderate'
)

result = executor.simulate_execution(
    action='enter',
    target_size=10.0,
    mid_price=4500.0,
    liquidity_state=liquidity,
    volatility_state=volatility,
    symbol='ES'
)

print(f"Filled at: {result.fill_price}")
print(f"Cost: ${result.total_cost:.2f}")
print(f"Position: {result.updated_position.quantity} contracts")
```

### Get Trade Summary

```python
summary = executor.get_trade_log_summary()
print(f"Total trades: {summary['total_trades']}")
print(f"Total slippage: {summary['total_slippage']}")
print(f"Total costs: ${summary['total_costs']:.2f}")
print(f"Avg fill %: {summary['average_fill_percentage']:.1f}%")
```

---

## Integration with RealDataTournament

### Automatic Initialization

RealDataTournament automatically initializes ExecutionSimulator:

```python
tournament = RealDataTournament(
    data_path="data/ES_2024.csv",
    symbol="ES",
    timeframe="1H",
    official_mode=True
)
# ExecutionSimulator is initialized internally
```

### Automatic Routing

All trades in the simulator route through ExecutionSimulator:

1. `RealDataTradingSimulator._close_position()` calls `executor.simulate_execution()`
2. Real fill prices are used instead of mid prices
3. Position P&L reflects slippage and costs
4. Trade log tracks all execution details

### Execution Logging

Detailed execution logs saved to: `logs/execution_<timestamp>.log`

---

## Pessimism Philosophy

ExecutionSimulator is intentionally pessimistic:

| Scenario | Pessimism |
|----------|-----------|
| Buy 100 contracts ES | Fill worse than best ask |
| Volatile market | Wider spreads, more slippage |
| Thin liquidity | Partial fill, reduced size |
| Large position | High slippage vs ATR |
| Round-trip | Commission charged at entry AND exit |

**Result:** PnL is conservative, realistic, never inflated by fantasy fills.

---

## Performance Impact

ExecutionSimulator adds realistic costs to PnL:

```
Mid-price P&L:      $100
Less: Spread cost:   -$5
Less: Slippage:      -$8
Less: Commission:    -$10
= Realistic P&L:    $77
```

This ensures:
- ELO ratings reflect true engine profitability
- Engine improvements are REAL, not due to better mid prices
- Risk management is honest (costs matter)

---

## Extensibility

ExecutionSimulator v1 is minimal but extensible:

- Modify `execution_config.yaml` to change spreads/slippage
- Extend `_calculate_spread()` for dynamic spread models
- Extend `_calculate_slippage()` for order flow toxicity
- Add partial fill strategies (e.g., VWAP, TWAP)

**Principle:** Keep it simple, deterministic, and pessimistic.

---

## Files

- **engine/execution_simulator.py** - ExecutionSimulator class (600+ lines)
- **execution_config.yaml** - Configuration per symbol
- **logs/execution_<timestamp>.log** - Trade execution logs

---

## Status

✓ Implemented: v1.0
✓ Tested: All core functionality
✓ Integrated: RealDataTournament routing complete
✓ Logging: Full trace of all executions
✓ Production-ready: Deterministic, time-causal, honest

---

## Next Steps

- Run `python analytics/run_elo_evaluation.py --brutal-tournament` with ExecutionSimulator
- Monitor logs to verify realistic fill prices
- Compare PnL with/without ExecutionSimulator to quantify impact
- Adjust `execution_config.yaml` based on real market feedback
