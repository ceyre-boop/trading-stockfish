📘 TRADING STOCKFISH — INSTITUTIONAL ENGINE ROADMAP (v4.x → v6.x)
Reference file for Copilot — defines architecture, modules, responsibilities, and development phases
0. Core Philosophy
Trading Stockfish is built using the same principles as the Stockfish chess engine:

state → evaluation → policy → execution → governance → benchmarking

Every module must be:

deterministic

replay‑safe

explainable

adversarially tested

scientifically validated

1. Canonical Market State (Source of Truth)
All modules must consume a unified MarketState object composed of the following blocks:

1.1 Price State
mid price

bid/ask

spreads

multi‑TF candles

returns

micro‑returns

1.2 Order Flow State
aggressive buy volume

aggressive sell volume

net imbalance

sweep detection

spoofing score

quote pulling score

1.3 Liquidity State (v4.0‑C)
top‑of‑book depth

cumulative depth

depth imbalance

liquidity resilience

liquidity pressure

liquidity shock flag

1.4 Volatility State (v4.0‑D)
realized volatility

intraday volatility bands

volatility-of-volatility

volatility regime classification

1.5 Macro/News State
hawkishness

risk sentiment

surprise score

macro regime

1.6 Execution Context
position

exposure

unrealized PnL

realized PnL

last fill price

last action

2. State Builder (Deterministic Data Pipeline)
Responsibilities:
Build MarketState from:

MT5 live feed

historical data (backtest/replay)

Maintain perfect parity between live and historical modes

Apply all feature blocks in deterministic order

Validate timestamps and enforce time‑causality

Provide neutral defaults when data is missing

Modules:
state_builder.py

order_flow_features.py

liquidity_features.py

volatility_features.py (future)

macro_news_engine.py

3. Evaluation Engine (Factorized, Causal, Explainable)
Responsibilities:
Convert MarketState into:

scalar evaluation score

confidence

factor breakdown

Use factorized scoring:

trend factor

order flow factor

liquidity factor

volatility factor

macro factor

Enforce determinism

Provide JSON‑serializable output

Modules:
causal_evaluator.py

evaluator_factors/ (optional future split)

4. Policy Engine (Action Selection)
Responsibilities:
Convert evaluation output into discrete actions:

FLAT

ENTER_LONG

ENTER_SHORT

ADD

REDUCE

EXIT

Condition decisions on:

volatility regime

liquidity regime

macro regime

risk constraints

Modules:
policy_engine.py

5. Execution Layer (Simulated + Live)
Responsibilities:
Provide two interchangeable backends:

ExecutionSimulator (deterministic, liquidity‑aware)

MT5Execution (live trading with safety checks)

Model:

slippage

partial fills

transaction costs

liquidity shocks

Never block or hang in demo/test mode

Modules:
execution_simulator.py

mt5/orders.py

mt5/live_feed.py

6. Governance & Risk Layer
Responsibilities:
Enforce institutional constraints:

max exposure

max drawdown

max trades per hour

stale data detection

kill switch

Provide meta‑rules:

veto unsafe trades

adjust factor weights by regime

enforce cooldown periods

Modules:
risk_engine.py

governance_engine.py

7. Benchmarking & Trading ELO
Responsibilities:
Run tournaments between engine versions

Evaluate performance across:

volatility regimes

liquidity regimes

macro regimes

Produce:

ELO scores

factor contribution reports

execution quality metrics

Modules:
elo_engine/

tournament_runner.py

scenario_library/

8. Research Cockpit & Replay Engine
Responsibilities:
Replay historical days with overlays:

evaluation score

regime

liquidity

volatility

PnL

Provide interactive debugging tools

Provide visualization hooks

Modules:
research_cockpit.py

replay_engine.py

9. Run Modes
The engine must support:

backtest

replay

paper

live

All modes must use the same:

state builder

evaluator

policy

governance

Only the execution backend changes.

10. Development Phases (Completed + Upcoming)
✔ v4.0‑A — Market State Reconstruction
✔ v4.0‑B — Order Flow Features
✔ v4.0‑C — Liquidity Dynamics
⬅ You are here
🔜 v4.0‑D — Volatility & Regime Transitions
v5.x — Policy Engine + Meta‑Governance
v6.x — Trading ELO + Scenario Library + Cockpit
11. Coding Rules for Copilot
Copilot must follow these rules:

Never break determinism

Never introduce randomness

Never change module boundaries

Never modify unrelated files

Always align dataclasses with usage

Always maintain test compatibility

Always preserve replay safety

Always follow the canonical MarketState schema

Always keep execution simulator deterministic and liquidity‑aware

12. Validation Requirements
Every module must have:

unit tests

integration tests

deterministic replay tests

JSON‑serialization tests

time‑causality tests

13. Final Objective
Build a Stockfish‑style trading engine that is:

deterministic

explainable

regime‑aware

liquidity‑aware

macro‑aware

scientifically benchmarked

institution‑grade