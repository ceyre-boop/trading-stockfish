# Operational Modes

## SIMULATION
- Full engine stack, no live orders or external connectors.
- Used for dry runs, policy tests, and feature trials.
- Same logging as other modes; execution adapter is simulation-only.

## PAPER
- Live market data with simulated fills; no real orders.
- Logs decisions exactly as if trading.
- Uses the same engine; execution adapter simulates fills.

## LIVE
- Real orders via connectors (IBKR/FIX/ZMQ or equivalent adapters).
- Requires explicit confirmation and config flag to enable.
- Shares core engine; differs only by execution adapter and risk envelope.

## Common Principles
- All modes share the same core engine and logging pipeline.
- Differences are limited to execution adapter and risk controls.
- Decision logging is identical across modes for comparability.
