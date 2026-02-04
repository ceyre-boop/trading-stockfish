# PAPER Week Execution (Phase 10)

Use this to launch and monitor the 7-day PAPER validation.

## Import scheduled tasks (Windows)
- Daily PAPER: `schtasks /create /tn "TradingStockfish_Daily_PAPER" /xml scripts\tasks\run_daily_paper_task.xml /f`
- Weekly cycle: `schtasks /create /tn "TradingStockfish_Weekly_Cycle" /xml scripts\tasks\run_weekly_cycle_task.xml /f`
- Heartbeat (optional, 5–10 min cadence): `schtasks /create /tn "TradingStockfish_Heartbeat" /tr "powershell -File %cd%\scripts\heartbeat_kill_switch.ps1" /sc minute /mo 5 /f`

Adjust start times/paths if the project root differs; XML defaults to `%SystemDrive%\Users\Admin\trading-stockfish`.

## Before Day 1 (quick gate)
- `.venv` present; `PYTHONPATH` points to project root.
- `storage/policies/policies.parquet` has at least one policy.
- Disk space > 100 GB; clock synced.
- Kill switch callable; connectors set to PAPER.

## During PAPER week
- Daily (automated): batch script logs to `logs/scheduled/daily_%DATE%.log`; expect decisions/audits/stats + storage partitions.
- Daily (2-minute review): scan `logs/`, check `storage/decisions|audits|stats` grew, ensure no zero-byte files, SAFE_MODE quiet.
- End of Day 7: weekly cycle log `logs/scheduled/weekly_%DATE%.log`; report at `reports/weekly_report_<date>.md`; no unintended promotions.
- Optional: run retrain `python -m engine.jobs.retrain_priors --days 7`; new model + metadata in `models/` only if metrics improve.
- Heartbeat: `scripts/heartbeat_kill_switch.ps1` writes to `logs/tasks/heartbeat_YYYYMMDD.log` and should show `"disabled": true` for SIMULATION adapter.

## Success criteria
- 7 consecutive PAPER runs, no manual reruns.
- All daily artifacts present; storage partitions valid (non-empty) each day.
- Weekly report generated and readable; no crash on empty partitions.
- No unexpected SAFE_MODE activations; guardrails callable.
- Test suite stays green (`pytest -q`).

When these are met, Phase 10 is validated; proceed to LIVE planning.
