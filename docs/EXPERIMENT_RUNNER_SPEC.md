# EXPERIMENT RUNNER SPECIFICATION - Phase 5

## Overview

The **ExperimentRunner** is a research framework for systematic parameter tuning and comparison. It:

- **Sweeps parameters**: Generate all combinations of configurable dimensions
- **Runs tournaments**: Execute full trading simulations for each config
- **Collects metrics**: PnL, Sharpe ratio, drawdown, win rate, regime performance
- **Compares results**: Statistical analysis and ranking of configurations
- **Exports reports**: JSON, Markdown, and detailed analysis

## Architecture

### Core Components

```
ExperimentRunner
├── generate_parameter_sets()   # Create all combinations
├── run_experiments()           # Execute all tests
├── compare_results()           # Statistical analysis
└── export_results()            # Save to disk
```

### Configuration

```yaml
# config/experiment_config.yaml
name: "macro_weight_tuning_v1"
description: "Systematic sweep of macro evaluator weight"

# Parameter Sweeps
macro_weight_range: [0.0, 1.0]
macro_weight_steps: 5             # 0.0, 0.25, 0.5, 0.75, 1.0

volatility_threshold_range: [0.01, 0.05]
volatility_threshold_steps: 3     # 0.01, 0.03, 0.05

reversal_threshold_range: [0.3, 0.7]
reversal_threshold_steps: 3       # 0.3, 0.5, 0.7

cooldown_range: [1, 10]
cooldown_steps: 3                 # 1, 5, 10

fomc_trade_bans: [false, true]    # Test both on/off

# Test Parameters
symbols: [EURUSD, GBPUSD, XAUUSD]
start_date: "2023-01-01"
end_date: "2024-12-31"
walkforward_periods: 4             # Quarterly splits

# Execution
parallel: false
max_workers: 4
verbose: true
```

## Data Structures

### ExperimentConfig

Defines what parameters to sweep and how:

```python
@dataclass
class ExperimentConfig:
    name: str                           # Experiment name
    description: str                    # What this tests
    
    # Parameter Ranges (min, max)
    macro_weight_range: Tuple[float, float]
    volatility_threshold_range: Tuple[float, float]
    reversal_threshold_range: Tuple[float, float]
    cooldown_range: Tuple[int, int]
    fomc_trade_bans: List[bool]
    
    # Resolution (number of points to test)
    macro_weight_steps: int
    volatility_threshold_steps: int
    reversal_threshold_steps: int
    cooldown_steps: int
    
    # Test Parameters
    symbols: List[str]
    start_date: str
    end_date: str
    walkforward_periods: int
    
    # Output
    output_dir: str
    experiment_name: str
```

### ParameterSet

A single configuration to test:

```python
@dataclass
class ParameterSet:
    config_id: str                      # Unique ID (MD5 hash)
    parameters: Dict[str, Any]          # Actual parameter values
    
    # Contains:
    # - macro_weight: float
    # - volatility_threshold: float
    # - reversal_threshold: float
    # - cooldown: int
    # - fomc_trade_ban: bool
```

### ExperimentResult

Results from running one parameter set:

```python
@dataclass
class ExperimentResult:
    config_id: str
    status: str                         # COMPLETED, FAILED
    
    # Performance Metrics
    pnl: float                          # Total P&L
    sharpe_ratio: float                 # Risk-adjusted return
    max_drawdown: float                 # Peak-to-trough loss
    win_rate: float                     # % winning trades
    profit_factor: float                # Gross profit / Gross loss
    trades_count: int                   # Total trades
    
    # Regime-Segmented Performance
    regime_performance: Dict[str, Dict[str, float]]
    
    # Walk-Forward Stability
    walkforward_scores: List[float]
    walkforward_stability: float        # StdDev of walkforward scores
    
    # ELO Rating
    elo_rating: float                   # 1600 = baseline
    strength_class: str                 # Beginner, Intermediate, Advanced, Master
    
    # Execution Details
    total_volume: float
    avg_fill_price_slippage: float
    total_commissions: float
    
    # Timing
    start_time: datetime
    end_time: datetime
    error_message: Optional[str]
```

## Usage Examples

### Basic Parameter Sweep

```python
from pathlib import Path
from analytics.experiment_runner import ExperimentRunner, ExperimentConfig

# Create configuration
config = ExperimentConfig(
    name="macro_tuning_q1_2025",
    description="Sweep macro weight with volatility thresholds",
    macro_weight_range=(0.0, 1.0),
    macro_weight_steps=5,
    volatility_threshold_range=(0.01, 0.05),
    volatility_threshold_steps=3,
    symbols=['EURUSD', 'GBPUSD'],
    start_date="2023-01-01",
    end_date="2024-12-31",
    output_dir="experiments",
    experiment_name="macro_tuning_q1_2025",
)

# Create runner
runner = ExperimentRunner(config)

# Generate parameter sets
param_sets = runner.generate_parameter_sets()
print(f"Generated {len(param_sets)} parameter sets")

# Run experiments
results = runner.run_experiments()
print(f"Completed {len(results)} experiments")

# Compare results
report = runner.compare_results()

# Export
exported = runner.export_results()
print(f"Results saved to: {exported['summary']}")
```

### Load Configuration from YAML

```python
from pathlib import Path
from analytics.experiment_runner import ExperimentConfig, ExperimentRunner

# Load config from YAML
config_file = Path("config/experiment_config.yaml")
config = ExperimentConfig.from_yaml(config_file)

# Run experiments
runner = ExperimentRunner(config)
param_sets = runner.generate_parameter_sets()
results = runner.run_experiments()
report = runner.compare_results()
exported = runner.export_results()
```

### Analyze Results

```python
# Access top 10 configurations
top_10 = report['top_10']

for rank, config in enumerate(top_10, 1):
    print(f"#{rank}: Config {config['config_id']}")
    print(f"  ELO: {config['elo_rating']:.0f}")
    print(f"  PnL: ${config['pnl']:.2f}")
    print(f"  Sharpe: {config['sharpe_ratio']:.2f}")
    print(f"  Max DD: {config['max_drawdown']:.1%}")
    print()

# Access statistics
stats = report['statistics']
print(f"PnL Range: ${stats['pnl']['min']:.2f} - ${stats['pnl']['max']:.2f}")
print(f"Average Sharpe: {stats['sharpe_ratio']['mean']:.2f}")
print(f"Avg Max DD: {stats['max_drawdown']['mean']:.1%}")
```

### Custom Parameter Sweep

```python
# Sweep only macro weight, hold others constant
config = ExperimentConfig(
    name="macro_weight_only",
    macro_weight_range=(0.0, 1.0),
    macro_weight_steps=11,          # 0.0, 0.1, 0.2, ..., 1.0
    
    # Hold constant
    volatility_threshold_range=(0.02, 0.02),  # Single value
    volatility_threshold_steps=1,
    reversal_threshold_range=(0.5, 0.5),
    reversal_threshold_steps=1,
    cooldown_range=(5, 5),
    cooldown_steps=1,
    fomc_trade_bans=[False],        # Only one option
    
    symbols=['EURUSD'],
    start_date="2023-01-01",
    end_date="2023-12-31",
    output_dir="experiments",
    experiment_name="macro_weight_only",
)

runner = ExperimentRunner(config)
param_sets = runner.generate_parameter_sets()
print(f"Parameter sets: {len(param_sets)}")  # Should be 11
```

## Output Structure

### Directory Layout

```
experiments/
└── macro_tuning_q1_2025/           # experiment_name
    ├── config.json                 # Sweep configuration
    ├── results.json                # All detailed results
    ├── comparison.json             # Statistics and rankings
    └── SUMMARY.md                  # Human-readable summary
```

### SUMMARY.md Example

```markdown
# Experiment Summary: macro_weight_tuning_v1

## Configuration

- **Description**: Sweep macro evaluator weight with volatility thresholds
- **Experiment Name**: macro_tuning_q1_2025
- **Symbols**: EURUSD, GBPUSD, XAUUSD
- **Period**: 2023-01-01 to 2024-12-31
- **Total Configurations**: 45

## Results

- **Completed**: 45/45
- **Failed**: 0

## Statistics

### PnL
- Min: -$500.00
- Max: $5,250.00
- Mean: $2,100.00
- StdDev: $1,200.00

### Sharpe Ratio
- Min: 0.200
- Max: 2.500
- Mean: 1.350
- StdDev: 0.650

### Max Drawdown
- Min: 0.050
- Max: 0.300
- Mean: 0.150
- StdDev: 0.080

## Top 10 Configurations

| Rank | Config ID | ELO | PnL | Sharpe | Max DD | Win Rate |
|------|-----------|-----|-----|--------|--------|----------|
| 1 | a1b2c3d4 | 1850 | $5,250.00 | 2.500 | 0.100 | 58.0% |
| 2 | e5f6g7h8 | 1820 | $5,000.00 | 2.400 | 0.105 | 57.5% |
| 3 | i9j0k1l2 | 1800 | $4,800.00 | 2.300 | 0.110 | 57.0% |
| ... | ... | ... | ... | ... | ... | ... |
```

## Metrics Explained

### PnL (Profit/Loss)

Total profit or loss from all trades. Higher is better.

```
PnL = Sum of all trade outcomes
```

### Sharpe Ratio

Risk-adjusted return. Measures excess return per unit of risk.

```
Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns
```

Interpretation:
- < 0.5: Poor
- 0.5-1.0: Fair
- 1.0-2.0: Good
- 2.0+: Excellent

### Max Drawdown

Largest peak-to-trough loss. Lower is better.

```
Max DD = (Peak - Trough) / Peak
```

Example: 20% drawdown means a $100 peak fell to $80 at worst.

### Win Rate

Percentage of trades that were profitable.

```
Win Rate = Winning Trades / Total Trades
```

Interpretation:
- < 40%: Most wins come from large winners offsetting many losers
- 40%-50%: Balanced (typical good strategy)
- 50%+: More winners than losers (ideal)

### Profit Factor

Gross profit divided by gross loss. > 1.5 is good.

```
Profit Factor = Sum of Wins / Sum of Losses
```

Interpretation:
- < 1.0: Losing system
- 1.0-1.5: Breakeven to barely profitable
- 1.5-2.0: Good
- 2.0+: Excellent

### ELO Rating

Chess-inspired rating system. Baseline = 1600.

```
ELO = 1600 + (Sharpe Ratio * 200) + (Win Rate * 300)
```

Classification:
- 1200-1400: Beginner
- 1400-1600: Intermediate
- 1600-1800: Advanced
- 1800+: Master

### Walk-Forward Stability

Standard deviation of performance across time periods.

Lower = more stable (good).
Higher = performance varies widely (risky).

```
Stability = StdDev of period returns
```

### Regime Performance

Performance breakdown by market regime:

- **high_vol**: Regime with >2% daily volatility
- **low_vol**: Regime with <0.5% daily volatility
- **risk_on**: Risk appetite up (indices up)
- **risk_off**: Risk aversion (flight to safety)

## Comparison Strategies

### 1. Rank by ELO

```python
top_configs = sorted(results, key=lambda r: r.elo_rating, reverse=True)
```

ELO balances PnL, Sharpe, and win rate automatically.

### 2. Trade-off Analysis

```python
# High Sharpe but lower PnL
conservative = [r for r in results if r.sharpe_ratio > 1.5 and r.pnl > 1000]

# High PnL but lower Sharpe
aggressive = [r for r in results if r.pnl > 3000 and r.sharpe_ratio > 0.5]
```

### 3. Regime Specific

```python
# Best in high volatility
hv_best = max(results, key=lambda r: r.regime_performance['high_vol']['sharpe'])

# Best in low volatility
lv_best = max(results, key=lambda r: r.regime_performance['low_vol']['sharpe'])
```

### 4. Stability Based

```python
# Most stable across time periods
stable = sorted(results, key=lambda r: r.walkforward_stability)[:10]
```

## Parameter Sweep Strategies

### 1. Coarse Grid First

Start with large steps:

```yaml
macro_weight_steps: 3          # 0.0, 0.5, 1.0
volatility_threshold_steps: 3
reversal_threshold_steps: 3
cooldown_steps: 3
```

### 2. Fine Grid After

If peak found, refine:

```yaml
macro_weight_range: [0.3, 0.7]    # Narrower range
macro_weight_steps: 5              # More steps
```

### 3. Linear vs Log Scale

Linear (current):
```python
[0.0, 0.25, 0.5, 0.75, 1.0]  # Equal spacing
```

Log scale (for non-linear effects):
```python
np.logspace(np.log10(0.01), np.log10(100), 5)  # Powers of 10
```

## Reproducibility

All experiments are deterministic if using same:

1. **Data**: Same historical data
2. **Config**: Same parameter ranges and steps
3. **Engine**: Same trading logic and evaluation

This means:
- Results are reproducible across runs
- Can share configs and replicate others' research
- Version control friendly

## Troubleshooting

### "No results to compare"

**Problem**: `compare_results()` returns empty dict

**Solution**: Run experiments first:
```python
runner.run_experiments()
runner.compare_results()
```

### All results identical

**Problem**: All configs produce same results

**Solution**:
1. Check that parameter ranges actually differ
2. Verify engine changes based on parameters
3. Test individual parameter set manually

### Extreme outliers

**Problem**: One config vastly outperforms others

**Solution**:
1. Check if overfitting to data
2. Verify walk-forward stability
3. Run on different symbols/periods

### Export fails

**Problem**: `export_results()` throws error

**Solution**:
1. Ensure `experiments/` directory exists
2. Check disk space
3. Verify write permissions

## See Also

- [REPLAY_ENGINE_SPEC.md](REPLAY_ENGINE_SPEC.md) - Candle-by-candle inspection
- [QUICK_START_PHASE5.md](QUICK_START_PHASE5.md) - Quick integration guide
- [config/experiment_config.yaml](config/experiment_config.yaml) - Configuration template
