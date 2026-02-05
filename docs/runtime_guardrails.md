# Runtime Guardrails (LIVE Execution Safety)

## Purpose
Define the mandatory safety limits and runtime protections that must be enforced during LIVE execution. Guardrails must be able to halt order flow instantly when violated.

## Mandatory Guardrails

### 1. Max Daily Loss
- If realized + unrealized loss exceeds threshold:
	- Trigger guardrail event
	- Halt order flow
	- Enter SAFE_MODE

### 2. Max Position Size
- Position size must not exceed configured limits.
- Violations must block new orders.

### 3. Max Leverage
- Leverage must remain below threshold.
- Violations must trigger SAFE_MODE.

### 4. Max Order Frequency
- Orders per minute must not exceed configured rate.
- Violations must block further orders.

### 5. Max Slippage Tolerance
- If estimated slippage exceeds threshold:
	- Block order
	- Log guardrail event

### 6. Heartbeat Timeout
- If no heartbeat from connector within threshold:
	- Trigger SAFE_MODE
	- Halt order flow

### 7. Connector Failure Thresholds
- Repeated failures must escalate to SAFE_MODE.
- Guardrail must integrate with connector health monitor.

### 8. SAFE_MODE Triggers
- Any guardrail violation must be able to activate SAFE_MODE.
- SAFE_MODE must block all order flow.

### 9. Kill‑Switch Triggers
- Guardrail engine must be able to activate kill‑switch.
- Kill‑switch must halt order flow instantly.

### 10. Fallback Behavior
- On any guardrail activation:
	- Stop sending orders
	- Log event
	- Emit guardrail decision
	- Enter SAFE_MODE if required

## Logging Requirements
- Every guardrail event must be logged.
- Logs must include:
	- timestamp
	- guardrail type
	- triggering metric
	- action taken
	- SAFE_MODE state

## Acceptance Criteria
- Guardrails must be able to stop order flow instantly.
- Guardrails must integrate with:
	- order adapter
	- connector health monitor
	- anomaly detector
	- SAFE_MODE subsystem
