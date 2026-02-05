# Live Dashboard Contract (Phase 11)

Purpose: Provide operators a real-time view of system health.

Key panels
- SAFE_MODE state
- Connector health (per adapter)
- Guardrail status (limits, current utilization)
- PnL and risk (positions, exposure, leverage)
- Regime and volatility indicators
- Active policy version
- Anomaly feed

Implementation options
- CLI or lightweight web UI: `python -m engine.dashboard.live`
- Refresh/heartbeat interval defined; read-only, non-mutating.

Acceptance
- Operators can see system health at a glance with minimal latency.
