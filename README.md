# trading-stockfish
Scientific benchmarking engine for intraday futures trading — regime-aware, risk-governed, and execution-honest.

## Architecture

```
trading_stockfish/
├── engine.py          # BenchmarkEngine — end-to-end orchestration
├── regime/
│   └── detector.py   # RegimeDetector — rolling-vol regime labeling (LOW/NORMAL/HIGH)
├── risk/
│   └── governor.py   # RiskGovernor   — position sizing, drawdown halt, VaR
├── execution/
│   └── tracker.py    # ExecutionTracker — slippage, fill rate, latency
├── signals/
│   └── generator.py  # SignalGenerator — momentum + mean-reversion, regime-aware blend
└── reporting/
    └── logger.py     # ExperimentLogger — structured JSON-lines experiment records
```

## Quick Start

```python
from trading_stockfish import BenchmarkEngine

engine = BenchmarkEngine()
result = engine.run(prices)          # prices: list/array of bar closes

print(f"Total return : {result.total_return:.2%}")
print(f"Max drawdown : {result.max_drawdown:.2%}")
print(f"Sharpe ratio : {result.sharpe_ratio:.2f}")
print(f"Regime breakdown: {result.regime_breakdown}")
```

## Modules

### Regime Detection (`regime/detector.py`)
Classifies each bar into **LOW_VOL**, **NORMAL**, or **HIGH_VOL** using a configurable rolling realised-volatility window.

### Risk Governance (`risk/governor.py`)
- Fixed-fractional position sizing (Kelly-inspired)
- Drawdown halt: stops trading when drawdown exceeds threshold
- Historical VaR at configurable confidence level

### Execution Tracking (`execution/tracker.py`)
Records every order attempt and fill. Computes aggregate statistics: fill rate, mean slippage in basis points, and mean fill latency.

### Signal Generation (`signals/generator.py`)
- **Momentum**: long/short when price return over a lookback window exceeds a threshold
- **Mean-reversion**: z-score based — enter when stretched, exit near mean
- **Combined (regime-aware)**: blends momentum and mean-reversion with weights that adapt to the detected regime

### Experiment Logging (`reporting/logger.py`)
Appends structured `ExperimentRecord` objects to a JSON-lines file for reproducible benchmarking.

## Development

```bash
pip install -e ".[dev]"
pytest
```
