# Connector Validation for PAPER Week

Validate adapters before the 7-day PAPER run to ensure SIM/PAPER modes initialize cleanly and never touch LIVE paths.

## How to Run
From project root:
```bash
python scripts/validate_connectors.py
```
Logs: `logs/system/connector_validation_YYYYMMDD.log`
Exit code non-zero on failure.

## Expected PASS Output
```
PASS: connectors initialize cleanly for SIM and PAPER
```
- Adapters created (live=False, disabled=False)
- Heartbeat/status check returns a simulated response

## What to Fix on FAIL
- Adapter init failure: check engine.modes adapters or environment setup.
- Adapter marked live in SIM/PAPER: inspect adapter definitions; ensure LIVE is not used.
- Heartbeat failure: review adapter.place_order behavior; ensure no SAFE_MODE blocking SIM/PAPER.
- Disabled flag set: confirm SAFE_MODE state or adapter disable logic.

Re-run after fixes; proceed with PAPER week only on PASS.
