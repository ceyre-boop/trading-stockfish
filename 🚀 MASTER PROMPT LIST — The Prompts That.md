🚀 MASTER PROMPT LIST — The Prompts That Complete Your Entire Project
These are the major prompts, in the order you’ll use them, to build the whole trading‑stockfish system from scratch.

1. Create the State Builder (core data pipeline)
Prompt:

Code
Generate state/state_builder.py based on the architecture in PROJECT_PLAN.md.
It should connect to MetaTrader5, fetch live ticks and multi-timeframe candles, compute market variables, and return a structured state dict. Include placeholders for sentiment, volatility, and trend regime features. Add robust error handling for stale ticks and missing data.
2. Add Market Indicators & Feature Engineering
Prompt:

Code
Enhance state/state_builder.py by adding technical indicators (ATR, RSI, EMA, volatility metrics, trend regime detection). Follow the state schema in PROJECT_PLAN.md. Ensure all features are added to the state dict cleanly.
3. Build the AI News Interpreter
Prompt:

Code
Generate state/news.py with a function interpret_headlines(headlines) that uses an LLM API (OpenAI/Claude/Gemini) to return structured sentiment variables. Include error handling and a fallback mode.
4. Build the Evaluator (decision engine)
Prompt:

Code
Generate engine/evaluator.py with a function evaluate(state) that returns buy/sell/hold/close decisions based on the rules in PROJECT_PLAN.md. Include risk filters, trend regime logic, sentiment weighting, and safety checks.
5. Add Risk Management Module
Prompt:

Code
Generate engine/risk.py with functions for position sizing, stop-loss placement, take-profit logic, and exposure limits. Integrate with evaluator.py.
6. Build the MT5 Live Data Bridge
Prompt:

Code
Generate mt5/live_feed.py that initializes MT5, fetches ticks, candles, spreads, and symbol info. Include reconnection logic and detailed error handling.
7. Build the MT5 Order Execution Layer
Prompt:

Code
Generate mt5/orders.py with functions buy(), sell(), close(), and modify_order(). Include logging, error handling, and safety checks to prevent duplicate or runaway orders.
8. Build the Real-Time Loop
Prompt:

Code
Generate loop/realtime.py that runs once per second, builds the state, evaluates decisions, logs outputs, and executes orders. Include demo mode (--demo) that disables trading but logs everything.
9. Add Logging & Diagnostics
Prompt:

Code
Generate logs/logger.py with rotating logs for state snapshots, decisions, errors, and MT5 events. Integrate it into all modules.
10. Add Configuration System
Prompt:

Code
Generate config/settings.py that loads environment variables, API keys, MT5 settings, and trading parameters. Use python-dotenv for overrides.
11. Add Utilities & Shared Helpers
Prompt:

Code
Generate utils/helpers.py with shared utilities for time conversion, safe dictionary access, retry logic, and data validation.
12. Build the Backtesting Framework (optional but powerful)
Prompt:

Code
Generate engine/backtester.py that simulates the evaluate() logic on historical data. Include PnL tracking, drawdown metrics, and trade logs.
13. Add Safety Layers
Prompt:

Code
Add safety checks across all modules: max drawdown, max trades per hour, stale data detection, MT5 disconnection handling, and emergency shutdown triggers.
14. Final Integration Test
Prompt:

Code
Generate tests/integration_test.py that runs a full cycle: build state → evaluate → log → (demo) execute. Confirm all modules communicate correctly.
15. Deployment Prep
Prompt:

Code
Generate a deployment guide in DEPLOY.md explaining how to run the system on a VPS or Codespaces, including environment variables, MT5 setup, and process monitoring.
🎯 This is the entire project in prompt form
If you run these prompts in order, refining each module as you go, you will end up with:

a complete state builder

a decision engine

a news interpreter

a risk system

an MT5 bridge

a real‑time loop

logging, config, utilities

optional backtesting

deployment instructions

This is the full blueprint for finishing your project using AI‑assisted development.