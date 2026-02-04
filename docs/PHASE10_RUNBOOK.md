# Phase 10 Operator Runbook

## Pre-flight (before enabling live or scheduled runs)
- Verify test suite green: `pytest -q`.
- Confirm environment: `.venv` active, `PYTHONPATH=c:/Users/Admin/trading-stockfish`.
- Check guardrails baseline: run `python -m engine.guardrails --test-kill-switch` (or manual call) and ensure orders are disabled.
- Policy present: `storage/policies/policies.parquet` contains last-good (e.g., `phase10_baseline_v1`).
- Storage/log paths writable: `logs/`, `storage/`, `reports/` exist; disk free >10 GB.
- Credentials/connectors (if any) set for PAPER/LIVE; SIM requires none.

## Daily operator checklist (scheduled or manual)
- Run daily: `python -m engine.jobs.daily_run --mode SIMULATION --run-id <YYYYMMDD>`.
- Verify artifacts: `logs/decision_log.jsonl`, `logs/audits/*.json`, `logs/feature_stats.json` updated.
- Storage updated: new `storage/decisions_*.parquet`, `storage/audits_*.parquet`, `storage/stats_*.parquet` written.
- Confirm no order adapter use in SIM; in PAPER/LIVE confirm preflight guardrails passed.
- Spot-check logs for errors in `logs/tasks/` (if scheduled) or console.
- Optional: run research API spot query for last 1–3 days to confirm analytics isn’t empty.

## Weekly operator checklist (scheduled or manual)
- Run weekly: `python -m engine.jobs.weekly_cycle --mode SIMULATION --days 7` (add `--confirm-live` only for live).
- Verify weekly report generated: `reports/weekly_report_<date>.md`.
- Ensure no policy promotion occurred unless explicitly allowed; review policy diff if promotion was enabled.
- Confirm research outputs: regime performance table, feature drift, policy comparison not empty (unless no data).
- Check storage growth and disk headroom; rotate/compress logs if near thresholds.

## Incident response (SIM/PAPER/LIVE)
- Immediate: trigger kill switch: `python - <<"PY"
from engine.guardrails import kill_switch
from engine.modes import Mode
print(kill_switch(Mode.SIMULATION))  # or PAPER/LIVE
PY`
- Halt schedulers: disable daily/weekly tasks in Task Scheduler (or cron) until resolved.
- Capture state: copy recent logs (`logs/`, `logs/tasks/`), current `storage/*` partitions, and latest `reports/`.
- Triage: note error signatures, PnL/position limits, connector health, policy version in use.
- Rollback: if policy regression suspected, revert to last-good policy in `storage/policies/policies.parquet` (promote manually if needed).
- After fix: re-run tests, run a SIM daily, then re-enable schedulers.

## Log retention policy (suggested)
- Retain `logs/decision_log.jsonl` and `logs/audits/*.json` for 30 days; compress older to `.gz` and move to `logs/archive/`.
- Retain `logs/tasks/*.log` (scheduler outputs) for 14 days; compress older.
- Retain Parquet storage (`storage/*`) indefinitely; optional prune to 180–365 days if disk pressure, after exporting snapshots.
- Retain reports (`reports/*.md`/`.json`) indefinitely for auditability; compress older than 180 days if needed.
- Automate via weekly rotation script (e.g., `ops/rotate_logs.ps1`) scheduled in Task Scheduler.

## Quick references
- Daily run: `python -m engine.jobs.daily_run --mode SIMULATION --run-id <YYYYMMDD>`
- Weekly cycle: `python -m engine.jobs.weekly_cycle --mode SIMULATION --days 7`
- Kill switch: `python - <<"PY" ... kill_switch(Mode.SIMULATION) ... PY`
- Tests: `pytest -q`
