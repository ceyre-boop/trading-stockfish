# Real‑Time Anomaly Detection (LIVE Execution)

## Purpose
Define the mandatory anomaly classes, detection rules, and escalation paths for real‑time monitoring of market conditions, engine behavior, and connector stability. Anomalies must be detected within one bar and escalated to guardrails + SAFE_MODE when required.

## Mandatory Anomaly Classes

### 1. Drift Spikes
- Sudden changes in feature distributions.
- Detection via rolling mean/variance deviation.
- Must emit anomaly event.

### 2. Volatility Shocks
- Sudden increase in realized or implied volatility.
- Must emit anomaly event.
- May require SAFE_MODE depending on severity.

### 3. Regime Flips
- Rapid transition between market regimes.
- Must emit anomaly event.
- Must be logged with regime_before and regime_after.

### 4. Stale Data
- Market data timestamp lags behind wall‑clock.
- Must emit anomaly event.
- Must integrate with connector health monitor.

### 5. Repeated SAFE_MODE Triggers
- SAFE_MODE activated multiple times in short window.
- Must emit anomaly event.
- Must escalate to operator attention.

### 6. Order Adapter Anomalies
- Missing fills
- Unexpected fills
- Out‑of‑order events
- Must emit anomaly event.

### 7. Storage Write Failures
- Failed parquet writes
- Corrupted partitions
- Must emit anomaly event.

## Logging Requirements
- All anomalies must be logged to logs/anomalies/anomaly_<timestamp>.json
- Logs must include:
	- anomaly_type
	- triggering metric
	- severity
	- SAFE_MODE requirement
	- action taken

## Integration Requirements
- Anomaly detector must integrate with:
	- guardrail engine
	- SAFE_MODE subsystem
	- connector health monitor
	- live audit trail

## Acceptance Criteria
- Anomalies must be detected within one bar.
- Detection must be deterministic and idempotent.
- No anomaly may silently fail.
