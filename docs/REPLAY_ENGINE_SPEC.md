# REPLAY ENGINE SPECIFICATION - Phase 5

## Overview

The **ReplayEngine** is a research tool that steps through historical market data candle-by-candle, displaying the complete internal state of the Trading Stockfish engine at each candle. It provides:

- **Candle-by-candle inspection**: Step through data one candle at a time
- **Full state visibility**: See all 8 causal factors, evaluations, decisions, execution, and P&L
- **Deterministic replays**: Same data + config always produces same results
- **Comprehensive logging**: Detailed logs and JSON exports for analysis
- **Educational walkthrough**: Understand exactly how the engine thinks

## Architecture

### Core Components

```
ReplayEngine
├── step()              # Advance one candle
├── run_full()          # Run entire dataset
├── reset()             # Reset to beginning
├── export_json()       # Export snapshots as JSON
└── export_log()        # Export detailed human-readable log
```

### Data Structures

#### ReplaySnapshot
Complete state at a single candle:

```python
@dataclass
class ReplaySnapshot:
    # Candle OHLCV
    candle_index: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Market State (8 causal factors)
    market_state: Dict[str, Any]
    
    # Causal Evaluator Output
    eval_score: float              # [-1, +1] direction and strength
    eval_confidence: float         # [0, 1] conviction level
    subsystem_scores: Dict[str, float]  # Individual factor scores
    
    # Policy Engine Decision
    policy_action: str             # ENTER_FULL, ENTER_SMALL, EXIT, etc.
    target_size: float             # 0-1 normalized position size
    action_reasoning: str          # Why this action was chosen
    
    # Execution Details
    fill_price: float              # Actual fill price
    filled_size: float             # Actual filled size
    transaction_cost: float        # Commission + fees
    slippage: float                # Bid-ask spread cost
    
    # Position State
    position_side: str             # FLAT, LONG, SHORT
    position_size: float           # Current size
    entry_price: float             # Entry price
    unrealized_pnl: float          # Current P&L
    realized_pnl: float            # Closed P&L
    
    # Health & Governance
    regime_label: str              # Market regime
    health_status: str             # HEALTHY, DEGRADED, CRITICAL
    risk_multiplier: float         # 1.0, 0.5, or 0.0
    governance_kill_switch: bool   # Emergency stop active?
    
    # Daily Totals
    daily_pnl: float               # Today's P&L
    cumulative_pnl: float          # Total session P&L
```

#### ReplaySession
Metadata for entire replay run:

```python
@dataclass
class ReplaySession:
    symbol: str
    start_date: datetime
    end_date: datetime
    config_hash: str               # Hash of config
    snapshots: List[ReplaySnapshot]
    stats: Dict[str, Any]          # Computed statistics
```

## Usage Examples

### Basic Replay

```python
import pandas as pd
from analytics.replay_day import ReplayEngine

# Load market data
data = pd.read_csv('data/EURUSD_1h.csv', index_col='time', parse_dates=True)

# Create replay engine
engine = ReplayEngine(
    symbol='EURUSD',
    data=data,
    config={'evaluator_weights': {'macro': 0.3, 'liquidity': 0.3, 'volatility': 0.4}},
    verbose=True
)

# Step through candle-by-candle
for i in range(50):
    snapshot = engine.step()
    print(f"Candle {i}: {snapshot.policy_action} @ {snapshot.close:.4f}")
```

### Full Replay

```python
# Run entire dataset
snapshots = engine.run_full()

# Export results
json_file = engine.export_json()    # For machine reading
log_file = engine.export_log()      # For human reading

print(f"Processed {len(snapshots)} candles")
print(f"Final PnL: {engine.cumulative_pnl:.2f}")
```

### Inspect Specific Candle

```python
# Get all snapshots
snapshots = engine.run_full()

# Examine candle 42
snap = snapshots[42]
print(f"Timestamp: {snap.timestamp}")
print(f"Price: {snap.close:.4f}")
print(f"Eval: {snap.eval_score:.3f} (confidence: {snap.eval_confidence:.1%})")
print(f"Decision: {snap.policy_action} ({snap.target_size:.1%})")
print(f"Fill: {snap.fill_price:.4f} (cost: {snap.transaction_cost:.6f})")
print(f"Position: {snap.position_side} {snap.position_size:.1%}")
print(f"PnL: {snap.cumulative_pnl:.2f}")
```

### Find Critical Decisions

```python
snapshots = engine.run_full()

# Find entries with high confidence
high_confidence_entries = [
    s for s in snapshots 
    if s.policy_action.startswith('ENTER') and s.eval_confidence > 0.8
]

print(f"High-confidence entries: {len(high_confidence_entries)}")

# Find positions that went negative
losers = [
    s for s in snapshots 
    if s.realized_pnl < 0
]

print(f"Losing trades: {len(losers)}")
```

## Interpreting Replay Output

### Evaluation Score Ranges

| Range | Meaning | Action |
|-------|---------|--------|
| -1.0 to -0.8 | Strong bearish | ENTER_FULL (short) or EXIT (long) |
| -0.8 to -0.5 | Bearish | ENTER_SMALL (short) or REDUCE (long) |
| -0.5 to +0.5 | Neutral | HOLD or DO_NOTHING |
| +0.5 to +0.8 | Bullish | ENTER_SMALL (long) or REDUCE (short) |
| +0.8 to +1.0 | Strong bullish | ENTER_FULL (long) or EXIT (short) |

### Health Status Meanings

| Status | Risk Multiplier | Meaning | Action |
|--------|-----------------|---------|--------|
| HEALTHY | 1.0 | Engine performing well | Normal position sizing |
| DEGRADED | 0.5 | Performance declining | Reduce position size 50% |
| CRITICAL | 0.0 | Engine failing | Stop new entries, close positions |

### Regime Labels

- **high_vol**: High volatility environment, expect larger swings
- **low_vol**: Low volatility, tighter ranges
- **risk_on**: Risk appetite up, flow positive
- **risk_off**: Risk aversion, defensive positioning

## Export Formats

### JSON Export

Snapshot export with complete data suitable for machine analysis:

```json
{
  "symbol": "EURUSD",
  "config_hash": "a1b2c3d4",
  "stats": {
    "total_candles": 100,
    "start_price": 1.0500,
    "end_price": 1.0650,
    "final_pnl": 1500.00
  },
  "snapshots": [
    {
      "candle_index": 0,
      "timestamp": "2024-01-01T00:00:00",
      "open": 1.0500,
      "high": 1.0510,
      "low": 1.0490,
      "close": 1.0505,
      "volume": 15000,
      "eval_score": 0.25,
      "eval_confidence": 0.65,
      "policy_action": "ENTER_SMALL",
      "position_side": "LONG",
      "cumulative_pnl": 0.00
    },
    ...
  ]
}
```

### Log Export

Human-readable detailed log with full reasoning:

```
============================================================
REPLAY LOG: EURUSD
============================================================
Start: 2024-01-01
End: 2024-12-31
Total Candles: 252
Config Hash: a1b2c3d4

============================================================
CANDLE-BY-CANDLE DETAILS
============================================================

Candle 0: 2024-01-01 00:00:00
  Price: O=1.0500 H=1.0510 L=1.0490 C=1.0505 V=15000
  
  Evaluation:
    Score: 0.250 (Confidence: 0.650)
    Subsystems:
      - Trend: 0.300
      - RSI: 0.200
      - Momentum: 0.150
  
  Policy Decision:
    Action: ENTER_SMALL
    Target Size: 0.50
    Reasoning: Score=0.250, Confidence=0.650, Action=ENTER_SMALL
  
  Execution:
    Fill Price: 1.05075
    Filled Size: 0.50
    Transaction Cost: 0.000075
    Slippage: 0.00075
  
  Position:
    Side: LONG
    Size: 0.50
    Entry: 1.05075
    Unrealized PnL: 0.00
    Realized PnL: 0.00
  
  Health & Governance:
    Regime: risk_on
    Health: HEALTHY
    Risk Multiplier: 1.00
    Kill Switch: false
  
  P&L:
    Daily: 0.00
    Cumulative: 0.00
```

## Performance Characteristics

### Speed

- **Per-candle time**: ~1-2ms on modern hardware
- **1000 candles**: ~1-2 seconds
- **Full year (252 trading days)**: ~0.25-0.5 seconds

### Memory

- **Base overhead**: ~5 MB
- **Per snapshot**: ~2-3 KB
- **1000 candles**: ~2-3 MB
- **Full year**: ~0.5-0.75 MB

## Integration with Trading Engine

ReplayEngine is designed to work with:

1. **MarketStateBuilder**: Provides 8 causal factors
2. **CausalEvaluator**: Produces eval_score and confidence
3. **PolicyEngine**: Generates trading actions
4. **ExecutionSimulator**: Fills trades with realistic costs
5. **PortfolioRiskManager**: Monitors aggregate exposure
6. **Governance**: Enforces kill switches
7. **EngineHealthMonitor**: Tracks performance and health

All components are deterministic and time-causal.

## Logging

### Log Levels

- **DEBUG**: Individual candle processing steps
- **INFO**: Replay start/end, exports, milestones
- **WARNING**: Data quality issues, extreme values
- **ERROR**: Execution failures, data corruption

### Log Locations

```
logs/
├── replay/
│   ├── replay_EURUSD_20240101_120000.log        # Main log
│   ├── replay_detailed_EURUSD_20240101_120000.log  # Detailed log
│   └── replay_snapshots_EURUSD_20240101_120000.json  # JSON export
└── experiments/
    └── experiment_config_hash.log
```

## Advanced Usage

### Custom Market State

For integrating with live MarketStateBuilder:

```python
# Override market state building
engine.build_market_state = custom_build_function

# Then run replay
snapshots = engine.run_full()
```

### Replay Comparison

Compare two different configs:

```python
engine1 = ReplayEngine(symbol='EURUSD', data=data, config=config1)
engine2 = ReplayEngine(symbol='EURUSD', data=data, config=config2)

snapshots1 = engine1.run_full()
snapshots2 = engine2.run_full()

# Find differences
for s1, s2 in zip(snapshots1, snapshots2):
    if s1.policy_action != s2.policy_action:
        print(f"Different action at {s1.candle_index}: {s1.policy_action} vs {s2.policy_action}")
```

### Walk-Forward Analysis

Replay sequential time periods:

```python
dates = pd.date_range('2023-01-01', '2024-12-31', freq='M')

results = []
for start_date in dates:
    end_date = start_date + pd.DateOffset(months=1)
    period_data = data[start_date:end_date]
    
    engine = ReplayEngine(symbol='EURUSD', data=period_data)
    snapshots = engine.run_full()
    
    results.append({
        'period': start_date.strftime('%Y-%m'),
        'pnl': engine.cumulative_pnl,
        'trades': sum(1 for s in snapshots if s.policy_action.startswith('ENTER')),
    })

# Analyze stability
for r in results:
    print(f"{r['period']}: {r['pnl']:.2f} ({r['trades']} trades)")
```

## Determinism Guarantee

ReplayEngine is 100% deterministic:

- **Same data** + **same config** = **same snapshots**
- No random number generation in core replay
- All decisions are rule-based
- Timestamps are repeatable

This allows for:
- Reproducible research
- Regression testing
- Version control compatibility
- Cross-team validation

## Troubleshooting

### "Insufficient Data" Market State

**Problem**: Early candles show `status: "insufficient_data"`

**Solution**: This is expected. The engine needs 20+ historical candles to build valid market state. Trade decisions are deferred until enough data is available.

### All Positions Flat

**Problem**: Engine never enters any trades

**Solution**: 
1. Check evaluation scores are non-zero (issue with market state building)
2. Check policy thresholds (may be too strict)
3. Lower entry confidence threshold

### High Transaction Costs

**Problem**: PnL severely impacted by slippage and commissions

**Solution**:
1. Check ExecutionSimulator spread configuration
2. Reduce position size to reduce notional cost
3. Verify commission model is accurate

## See Also

- [EXPERIMENT_RUNNER_SPEC.md](EXPERIMENT_RUNNER_SPEC.md) - Parameter sweeps and comparison
- [QUICK_START_PHASE5.md](QUICK_START_PHASE5.md) - Quick integration guide
- [engine/health_monitor.py](engine/health_monitor.py) - Health tracking system
- [analytics/run_elo_evaluation.py](analytics/run_elo_evaluation.py) - Full tournament system
