# Live Trading Guardrails

Pre-flight checks (must all pass):
- Test suite green.
- SAFE_MODE state known.
- Active policy version recorded and available.
- Connectors healthy (IBKR/FIX/ZMQ or equivalent).

Runtime guardrails:
- Max daily loss threshold.
- Max position size per instrument/aggregate.
- Kill switch (manual + automatic) to stop orders immediately.

Post-session steps:
- Update storage (append-only Parquet) and archive logs.
- Quick research summary (regime/performance snapshot) for the session.
- Log anomalies and safety events.

All guardrails are deterministic and auditable; no live changes occur without explicit confirmation.
