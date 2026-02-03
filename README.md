# Trading Stockfish

A modular AI-augmented trading engine for MetaTrader 5, combining algorithmic analysis with real-time market execution.

## Overview

**Trading Stockfish** is a Python-based trading automation system designed to:
- Build structured market states from live MT5 data (price, volume, spreads, timeframes)
- Evaluate trading decisions using rule-based logic and AI sentiment analysis
- Execute buy/sell/close orders on MT5 with risk management
- Run in real-time loop mode with demo/live trading modes
- Log all decisions and trades for backtesting and audit trails

## Project Architecture

This project follows a modular design philosophy:

```
trading-stockfish/
├── state/          → Market state builders (data ingestion, normalization)
├── engine/         → Trading logic (evaluator, risk filters, decision making)
├── mt5/            → MetaTrader5 integration (live feed, order execution)
├── loop/           → Real-time trading loop (orchestration, timing)
├── logs/           → Trade logs, decision logs, and backtesting data
├── config/         → Configuration files (symbols, parameters, risk limits)
├── utils/          → Utility functions (helpers, validators, formatters)
├── PROJECT_PLAN.md → Complete architecture and specification
└── README.md       → This file
```

## Folder Descriptions

| Folder | Purpose |
|--------|---------|
| **state/** | Pulls live MT5 data (ticks, candles, spreads), builds structured state dictionaries with market variables (RSI, moving averages, sentiment, etc.) |
| **engine/** | Evaluates state dictionaries and returns trading decisions (BUY/SELL/HOLD/CLOSE) based on rules, trends, and risk filters |
| **mt5/** | Connects to MetaTrader5 terminal, fetches live data, and executes orders |
| **loop/** | Real-time heartbeat—evaluates every second, logs decisions, and coordinates execution |
| **logs/** | Storage for trade logs, decision records, and backtesting results |
| **config/** | Symbol lists, risk parameters, timeframes, and tuning variables |
| **utils/** | Common helper functions, validators, and formatters |

## Module Generation

All modules in this project are generated using **AI-assisted development** (GitHub Copilot, Claude, Cursor, etc.) following prompts defined in `PROJECT_PLAN.md`.

### Workflow:
1. Read `PROJECT_PLAN.md` for architecture and specifications
2. Use AI tools to generate each module with targeted prompts
3. Refine modules iteratively (error handling, edge cases, performance)
4. Test in demo mode before enabling live trading

### Getting Started:
See `PROJECT_PLAN.md` for step-by-step module generation prompts.

## Dependencies

- Python 3.12+
- `MetaTrader5` - MT5 Python API
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `requests` - HTTP client (for news/sentiment APIs)
- `python-dotenv` - Environment configuration

## Feature Drift Detection

Use the drift detector to monitor feature health from audit logs.

- Run: `python -m analytics.feature_drift_detector --audit-dir logs/feature_audits --out logs/drift_reports/drift_report.json --window 20 --spike-factor 3 --abs-threshold 3 --missing-frac-threshold 0.8`
- Flags: `window` (baseline runs), `spike-factor` (relative anomaly), `abs-threshold` (minimum absolute spike), `missing-frac-threshold` (persistent missingness cutoff).
- Output: JSON report with run metadata, sorted findings per feature, and aggregates at `logs/drift_reports/drift_report.json`.

## Next Steps

1. Generate `state/state_builder.py` with live MT5 data pulling
2. Generate `engine/evaluator.py` with decision logic
3. Generate `mt5/live_feed.py` and `mt5/orders.py` for MT5 bridge
4. Generate `loop/realtime.py` for the trading heartbeat
5. Test in demo mode

---

*Last Updated: February 2, 2026*
