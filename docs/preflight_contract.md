# Pre‑Flight Contract (LIVE Mode Gatekeeper)

## Purpose
Define the mandatory system checks that must pass before the engine is permitted to enter LIVE mode. These checks ensure safety, correctness, and operational readiness.

## Mandatory Checks (All Required)

### 1. Test Suite Status
- Full test suite must be green.
- No skipped tests unless explicitly allowed.
- No xfail regressions.

### 2. System Clock Synchronization
- Clock must be synchronized within tolerance.
- NTP or OS‑level sync must be active.
- Drift must be below threshold.

### 3. Active Policy Validation
- Active policy must be schema‑compliant.
- Policy version must be valid and not expired.
- No pending policy promotions.
- Policy must match the expected hash.

### 4. SAFE_MODE State
- SAFE_MODE must be explicitly known.
- SAFE_MODE must be inactive before LIVE mode.
- SAFE_MODE transitions must be logged.

### 5. Connector Health
- All connectors must pass health checks.
- Heartbeats must be within threshold.
- No stale data.
- No unresolved reconnect attempts.

### 6. Disk Space Requirements
- Free disk space must exceed threshold.
- Storage directories must be writable.
- Log directories must be writable.

### 7. Environment Validation
- Required environment variables must be loaded.
- Virtual environment must be active.
- Python version must match expected version.

### 8. Kill‑Switch Verification
- Kill‑switch must be reachable.
- Kill‑switch must respond to test signal.
- Kill‑switch must be able to halt order flow instantly.

### 9. Weekly Cycle Status
- No pending weekly cycle.
- Weekly cycle must have completed successfully.
- Weekly cycle report must be present.

### 10. Storage Integrity
- No corrupted parquet partitions.
- All required directories must exist.
- Read/write tests must pass.

## Acceptance Criteria
LIVE mode cannot start unless **all** checks return PASS.

## Logging Requirements
- Pre‑flight results must be written to logs/preflight/preflight_<timestamp>.json.
- Log must include PASS/FAIL for each check.
- Log must include reasons for any failure.

## Enforcement
- Pre‑flight checker must block LIVE mode on any failure.
- No overrides allowed.
- No partial passes allowed.

## Integration
- LIVE mode entrypoint runs pre‑flight before any LIVE execution begins.
- LIVE startup is blocked and recorded to a block log if pre‑flight fails.
- `--skip-preflight` is forbidden in LIVE mode; only non‑LIVE runs may bypass the check.
