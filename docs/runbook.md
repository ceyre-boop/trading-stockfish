# Runbook

## LIVE Mode Gate
- Pre-flight is mandatory for LIVE mode; the entrypoint runs `run_preflight` and halts LIVE startup on any failure.
- No overrides: `--skip-preflight` is forbidden in LIVE mode and exits non-zero; it may be used only for SIMULATION or PAPER runs.
- Logs: results are in `logs/preflight/preflight_<timestamp>.json`; block records are in `logs/preflight/preflight_block_<timestamp>.json` with the failure list.
- Failure handling: read the block log, remediate the listed failures (clock sync, policy, connectors, storage, env), rerun pre-flight, and only then retry LIVE. Do not bypass the gate.
