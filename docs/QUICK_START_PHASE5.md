# QUICK START - Phase 5 Research Cockpit

## 30-Second Overview

Phase 5 gives Trading Stockfish two powerful research tools:

1. **ReplayEngine**: Step through historical data candle-by-candle, inspect every decision
2. **ExperimentRunner**: Systematically test parameter variations, rank by performance

Together they form the **Research Cockpit** for understanding and tuning the engine.

## Installation

All components are already integrated. No additional dependencies beyond:

```bash
pip install pandas numpy pyyaml pytest
```

## Quick Start - Replay a Day

### 1. Load Data

```python
import pandas as pd
from analytics.replay_day import ReplayEngine

# Load your market data
data = pd.read_csv('data/EURUSD_1h.csv')
# Must have columns: open, high, low, close, volume
```

### 2. Create Replay Engine

```python
engine = ReplayEngine(
    symbol='EURUSD',
    data=data,
    config={
        'evaluator_weights': {'macro': 0.3, 'liquidity': 0.3, 'volatility': 0.4}
    },
    verbose=True  # Print each decision
)
```

### 3. Run Full Replay

```python
snapshots = engine.run_full()
print(f"Processed {len(snapshots)} candles")
print(f"Final PnL: {engine.cumulative_pnl:.2f}")
```

### 4. Export Results

```python
json_file = engine.export_json()    # For analysis
log_file = engine.export_log()      # For reading

print(f"JSON export: {json_file}")
print(f"Log export: {log_file}")
```

## Quick Start - Run Experiment

### 1. Create Configuration

Option A: Minimal (Python)

```python
from analytics.experiment_runner import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    name="test_sweep",
    macro_weight_range=(0.0, 1.0),
    macro_weight_steps=3,
    volatility_threshold_steps=2,
    reversal_threshold_steps=2,
    cooldown_steps=2,
    fomc_trade_bans=[False, True],
    symbols=['EURUSD'],
    start_date="2023-01-01",
    end_date="2023-12-31",
    output_dir="experiments",
    experiment_name="test_sweep",
)
```

Option B: YAML (Recommended)

Create `config/experiment_config.yaml`:

```yaml
name: "macro_weight_test"
macro_weight_range: [0.0, 1.0]
macro_weight_steps: 5
volatility_threshold_range: [0.01, 0.05]
volatility_threshold_steps: 3
reversal_threshold_range: [0.3, 0.7]
reversal_threshold_steps: 3
cooldown_range: [1, 10]
cooldown_steps: 3
fomc_trade_bans: [false, true]
symbols: [EURUSD, GBPUSD]
start_date: "2023-01-01"
end_date: "2024-12-31"
experiment_name: "macro_weight_test"
```

### 2. Run Experiment

```python
from pathlib import Path
from analytics.experiment_runner import ExperimentConfig, ExperimentRunner

# Load config
config = ExperimentConfig.from_yaml(Path("config/experiment_config.yaml"))

# Create runner
runner = ExperimentRunner(config)

# Generate parameters (will create 5*3*3*3*2 = 270 configurations)
param_sets = runner.generate_parameter_sets()
print(f"Testing {len(param_sets)} configurations...")

# Run all experiments
results = runner.run_experiments()
print(f"Completed {len(results)} tests")

# Compare and rank
report = runner.compare_results()

# Export results
exported = runner.export_results()
print(f"Results saved: {exported['summary']}")
```

### 3. View Results

```bash
# Read the summary
cat experiments/macro_weight_test/SUMMARY.md

# Load detailed results
python -c "import json; data = json.load(open('experiments/macro_weight_test/results.json')); print(f'Top result: {data[0]}')"
```

## Key Concepts

### Market State (8 Causal Factors)

Every candle snapshot includes:

```
1. Trend       - SMA crossovers, direction
2. RSI         - Overbought/oversold
3. Momentum    - Rate of price change
4. Volatility  - ATR or realized vol
5. Macro       - Interest rates, sentiment
6. Liquidity   - Order book depth
7. Dealer Pos  - Gamma exposure
8. Time Regime - NY open, London, etc
```

### Evaluation Score [-1, +1]

- **-1.0**: Strong bearish signal
- **-0.5**: Mild bearish
- **0.0**: Neutral
- **+0.5**: Mild bullish
- **+1.0**: Strong bullish signal

### Policy Actions

- `ENTER_FULL`: Open position (high conviction)
- `ENTER_SMALL`: Open position (low conviction)
- `ADD`: Increase size
- `REDUCE`: Decrease size
- `EXIT`: Close position
- `HOLD`: Keep as-is
- `DO_NOTHING`: No action

### Health Status

- `HEALTHY` (1.0x): Engine performing well, use normal sizing
- `DEGRADED` (0.5x): Performance declining, cut position size 50%
- `CRITICAL` (0.0x): Engine failing, stop entries, close positions

## Common Workflows

### Workflow 1: Understand One Day

```python
# "Why did the engine make these trades on Jan 15?"
data = load_data('2024-01-15.csv')

engine = ReplayEngine(symbol='EURUSD', data=data, verbose=True)
snapshots = engine.run_full()

# Inspect specific candle
snap = snapshots[42]
print(f"Eval: {snap.eval_score:.2f}")
print(f"Action: {snap.policy_action}")
print(f"Fill: {snap.fill_price:.4f}")
print(f"PnL: {snap.cumulative_pnl:.2f}")
```

### Workflow 2: Optimize Macro Weight

```python
# "What's the best macro weight for this symbol?"
config = ExperimentConfig(
    name="macro_optimization",
    macro_weight_range=(0.0, 1.0),
    macro_weight_steps=11,  # 0.0 to 1.0 in 0.1 increments
    # Hold others constant
    volatility_threshold_range=(0.02, 0.02),
    volatility_threshold_steps=1,
    reversal_threshold_range=(0.5, 0.5),
    reversal_threshold_steps=1,
    cooldown_range=(5, 5),
    cooldown_steps=1,
    fomc_trade_bans=[False],
    symbols=['EURUSD'],
    start_date="2023-01-01",
    end_date="2024-12-31",
    output_dir="experiments",
    experiment_name="macro_opt",
)

runner = ExperimentRunner(config)
runner.generate_parameter_sets()
runner.run_experiments()
report = runner.compare_results()

# Print top result
top = report['top_10'][0]
print(f"Best macro weight: {top['config_id']}")
print(f"ELO: {top['elo_rating']:.0f}")
```

### Workflow 3: Compare Two Configurations

```python
# "Config A vs Config B - which is better?"
data = load_data('data/EURUSD_2024.csv')

config_a = {'evaluator_weights': {'macro': 0.2}}
config_b = {'evaluator_weights': {'macro': 0.8}}

engine_a = ReplayEngine(symbol='EURUSD', data=data, config=config_a)
engine_b = ReplayEngine(symbol='EURUSD', data=data, config=config_b)

snap_a = engine_a.run_full()
snap_b = engine_b.run_full()

print(f"Config A: PnL={engine_a.cumulative_pnl:.2f}, Trades={len([s for s in snap_a if s.policy_action.startswith('ENTER')])}")
print(f"Config B: PnL={engine_b.cumulative_pnl:.2f}, Trades={len([s for s in snap_b if s.policy_action.startswith('ENTER')])}")
```

### Workflow 4: Regime Analysis

```python
# "How does engine perform in different regimes?"
results = runner.run_experiments()

for result in results:
    print(f"\nConfig {result.config_id}:")
    for regime, metrics in result.regime_performance.items():
        print(f"  {regime}: PnL={metrics['pnl']:.0f}, Sharpe={metrics['sharpe']:.2f}")
```

## Interpreting Results

### Read the Summary

```markdown
# Experiment Summary: macro_weight_tuning_v1

## Statistics

### PnL
- Min: -$500.00
- Max: $5,250.00
- Mean: $2,100.00

### Sharpe Ratio
- Min: 0.200
- Max: 2.500
- Mean: 1.350

## Top 10 Configurations

| Rank | Config ID | ELO | PnL | Sharpe |
|------|-----------|-----|-----|--------|
| 1 | a1b2c3d4 | 1850 | $5,250 | 2.50 |
```

**What it means:**

- **#1 config** (a1b2c3d4): Best overall performance
  - ELO 1850 = "Master" level
  - PnL $5,250 = most profitable
  - Sharpe 2.50 = excellent risk-adjusted returns
  - This is the recommended configuration

### Find Your Use Case

**Goal: Maximize profit**
→ Sort by PnL, pick top result

**Goal: Stable returns**
→ Sort by Sharpe Ratio, pick high value

**Goal: Minimize losses**
→ Sort by Max Drawdown, pick lowest

**Goal: Balanced**
→ Sort by ELO, pick top result

## Export Formats

### JSON Snapshots

```json
{
  "symbol": "EURUSD",
  "snapshots": [
    {
      "candle_index": 0,
      "timestamp": "2024-01-01T00:00:00",
      "open": 1.0500,
      "close": 1.0505,
      "eval_score": 0.25,
      "policy_action": "ENTER_SMALL",
      "position_side": "LONG",
      "cumulative_pnl": 0.00
    },
    ...
  ]
}
```

### Log Files

```
Candle 0: 2024-01-01 00:00:00
  Price: C=1.0505
  Eval: 0.250 (Confidence: 0.650)
  Policy Decision: ENTER_SMALL
  Execution: Fill Price=1.05075, Cost=0.000075
  Position: LONG 0.50
  PnL: Cumulative=0.00
```

## Integration with CLI

### View Replay

```bash
python analytics/run_elo_evaluation.py --replay-day EURUSD --date 2024-01-15
```

### Run Experiment

```bash
python analytics/run_elo_evaluation.py --experiment config/experiment_config.yaml
```

## Performance Tips

### For Replays

- **Small datasets** (< 1000 candles): Run on laptop, < 1 second
- **Large datasets** (252 trading days): Still < 1 second
- **Multiple replays**: Replay in sequence, combine results

### For Experiments

- **Small sweep** (5×3×3×3 = 135 configs): 30-60 seconds
- **Large sweep** (11×5×5×5 = 1375 configs): 5-10 minutes
- **Use parallel=True** for large sweeps (requires more CPU)

## Troubleshooting

### Replay shows "insufficient data"

Early candles need historical context. Expected. Skip first 20-30 candles.

### All policy actions are "DO_NOTHING"

Engine is being conservative. Check:
1. Market state is being built (check `market_state['status']`)
2. Evaluation scores are non-zero
3. Policy thresholds aren't too strict

### Experiment takes forever

Running many configs × symbols × periods. Normal. To speed up:
1. Reduce experiment scope (fewer symbols, shorter period)
2. Use smaller parameter steps
3. Enable `parallel=True` if CPU allows

### Results don't match expectations

Check:
1. Same data? (use same date range)
2. Same config? (hash should match)
3. Determinism: Same inputs → Same outputs always

## Next Steps

1. **Run your first replay**: Step through one day, understand decisions
2. **Run small experiment**: Tune one parameter, see impact
3. **Compare strategies**: Run two configs, find winner
4. **Scale up**: Sweep multiple parameters, find optimal point

## See Also

- [REPLAY_ENGINE_SPEC.md](REPLAY_ENGINE_SPEC.md) - Complete ReplayEngine documentation
- [EXPERIMENT_RUNNER_SPEC.md](EXPERIMENT_RUNNER_SPEC.md) - Complete ExperimentRunner documentation
- [config/experiment_config.yaml](../config/experiment_config.yaml) - Configuration template
- [PROJECT_PLAN_TRADING_STOCKFISH_V1.txt](../PROJECT_PLAN_TRADING_STOCKFISH_V1.txt) - Full roadmap

## Support

For issues:
1. Check logs in `logs/replay/` or `logs/experiments/`
2. Run with `verbose=True` for detailed output
3. Review test cases in `tests/test_replay_engine.py` and `tests/test_experiment_runner.py`
