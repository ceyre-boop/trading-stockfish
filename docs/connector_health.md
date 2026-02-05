# Connector Health Monitoring (LIVE Execution)

## Purpose
Define the mandatory health checks, thresholds, and failure escalation rules for all LIVE connectors (FIX, ZMQ, IBKR, or any other adapter). Connector health must be monitored continuously and must integrate with guardrails and SAFE_MODE.

## Mandatory Health Dimensions

### 1. Heartbeat Interval
- Each connector must emit heartbeats at a defined interval.
- If heartbeat age exceeds threshold:
	- Emit health event
	- Escalate to guardrail engine
	- SAFE_MODE may be required

### 2. Latency Thresholds
- Round-trip latency must remain below configured threshold.
- If exceeded:
	- Emit health event
	- Mark connector as degraded

### 3. Reconnect Strategy
- On connection loss:
	- Attempt reconnect with exponential backoff
	- Log each attempt
	- After N failures, escalate to guardrail engine

### 4. Failure Thresholds
- Repeated order rejections
- Repeated send failures
- Repeated heartbeat misses
- If failures exceed threshold:
	- Emit health event
	- Trigger SAFE_MODE if required

### 5. Stale Data Detection
- If incoming market data timestamp lags behind wall-clock beyond threshold:
	- Emit stale-data event
	- Escalate to guardrails

### 6. Order Adapter Anomalies
- Unexpected fills
- Missing fills
- Out-of-order events
- If detected:
	- Emit anomaly event
	- Integrate with anomaly detector

## Logging Requirements
- All health events must be logged to logs/health/health_<timestamp>.json
- Logs must include:
	- connector name
	- event type
	- triggering metric
	- action taken
	- SAFE_MODE state

## Integration Requirements
- Health monitor must integrate with:
	- guardrail engine
	- SAFE_MODE subsystem
	- anomaly detector
	- live audit trail

## Acceptance Criteria
- Connector failures must be detected within seconds.
- Health monitor must be deterministic and idempotent.
