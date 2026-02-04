# Operator Checklist (Phase 10)

## 1) Environment & Runtime
- Activate venv: `C:\Users\Admin\trading-stockfish\.venv\Scripts\activate` (Task Scheduler: use full interpreter path).
- Set `PYTHONPATH=C:\Users\Admin\trading-stockfish`.
- Disk space: ensure >10 GB free before runs.
- Log rotation: weekly rotation/compression of `logs/` and `logs/tasks/`; prune archives per retention policy.
- Scheduler readiness: Task Scheduler/cron entries enabled for daily and weekly jobs; outputs logged to `logs/tasks/*.log`.

## 2) Pre-Run Safety Checks
- Tests green: `pytest -q` (latest commit).
- SAFE_MODE state known and recorded.
- Active policy present: `storage/policies/policies.parquet` contains last-good version.
- Connectors healthy for selected mode (SIM/PAPER/LIVE); credentials loaded for PAPER/LIVE.
- Kill switch reachable: `kill_switch(Mode.SIMULATION)` callable without error.

## 3) Daily Run Procedure
- Run: `python -m engine.jobs.daily_run --mode <SIMULATION|PAPER|LIVE> --run-id <YYYYMMDD>` (LIVE requires confirm flag if configured).
- Verify artifacts: `logs/decision_log.jsonl`, `logs/feature_audits/*.json`, `logs/feature_stats.json` updated for the run.
- Storage updated: new `storage/decisions_*.parquet`, `storage/audits_*.parquet`, `storage/stats_*.parquet` written.
- Order flow expectations: no orders in SIMULATION; simulated fills only in PAPER; real orders only in LIVE with guardrails on.

## 4) Post-Run Quick Checks
- Parquet partitions created for today; no zero-byte files in `storage/*`.
- Research API spot-check (last 1–3 days) returns non-empty regime performance or drift tables.
- Task log (if scheduled) shows success exit code; no uncaught errors.

## 5) Weekly Cycle
- Run: `python -m engine.jobs.weekly_cycle --mode <SIMULATION|PAPER|LIVE> --days 7` (LIVE requires confirm flag if configured).
- Verify weekly report: `reports/weekly_report_<date>.md` present with regime, drift, policy comparison sections.
- Promotions: none unless explicitly enabled; review any policy change before promotion.
- Backfill: if gaps detected, run storage backfill for the window before rerunning weekly report.

## 6) ML Retraining (weekly/monthly optional)
- Run: `python -m engine.jobs.retrain_priors --days 7` (or date window).
- Verify outputs: new `models/<model>/<version_tag>/` with metadata JSON and weights; metadata timestamp UTC.
- Promote only when metrics improve vs prior (accuracy up, brier down) and after review.

## 7) Monthly Deep Review
- Evaluate long-horizon regime performance and feature decay using research API.
- Review SAFE_MODE frequency/causes and guardrail triggers.
- Assess policy evolution across stored versions; prune or tag deprecated ones.
- Quantify ML prior value-add vs baseline; decide promotions/demotions.

## 8) Guardrails
- Preflight: `preflight_check(tests_green, policy_path, safe_mode_state, connectors_healthy)` should return ok before LIVE.
- Runtime: `runtime_limits(pnl_today, max_daily_loss, position, max_position)` configured and monitored.
- Kill switch: test in SIMULATION path; ensure adapter disables orders.

## 9) Filesystem Health
- `logs/` rotates; no runaway growth; archives intact.
- `storage/` grows predictably with daily partitions; no zero-byte parquet files.
- `models/` contains versioned artifacts with metadata; old versions pruned per policy.
- `reports/` present for weekly and monthly cycles; accessible for audits.
