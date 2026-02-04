# PAPER-Week Operational Validation (Phase 10)

This is the 7-day PAPER-mode validation gate for Phase 10. Run with no manual intervention beyond start/stop and log review.

## 1) Daily Run Procedure (PAPER mode)
- Command (per day): `python -m engine.jobs.daily_run --mode PAPER --run-id auto`
- Confirm after each run:
  - Decisions logged: `logs/decision_log.jsonl` appended.
  - Audits logged: `logs/feature_audits/*.json` created for the run.
  - Stats snapshot written: `logs/feature_stats.json` updated.
  - Storage updated: new `storage/decisions_YYYYMMDD.parquet`, `storage/audits_YYYYMMDD.parquet`, `storage/stats_YYYYMMDD.parquet`.
  - SAFE_MODE: no triggers unless expected; investigate any trigger.
  - Connectors: no external calls beyond simulated PAPER fills.

## 2) Daily Validation Checks
- Parquet partitions present for the day; no zero-byte files in `storage/*`.
- Research API quick summary (last 1–3 days) returns:
  - Regime performance table non-empty when data exists.
  - Feature drift table non-empty when stats exist.
  - Policy version comparison returns rows when policies are present.
- Scheduler logs: `logs/scheduled/daily_%DATE%.log` present and clean.

## 3) Weekly Cycle Trigger (end of week)
- Command: `python -m engine.jobs.weekly_cycle --mode SIMULATION --days 7`
- Confirm:
  - Weekly report generated: `reports/weekly_report_<date>.md`.
  - No unintended promotions (unless explicitly enabled and reviewed).
  - Storage backfill performed if gaps detected before report.
  - Analytics stable (no crashes on empty/invalid partitions).

## 4) ML Retraining (optional)
- Command: `python -m engine.jobs.retrain_priors --days 7`
- Confirm:
  - New model version directory under `models/` with metadata and weights.
  - Metadata correct: UTC timestamp, training window captured, metrics recorded.
  - Promotion only if metrics improve vs prior (accuracy up, brier down) after review.

## 5) Guardrails Verification (daily spot-check)
- `preflight_check()` passes with current policy and connector health.
- `runtime_limits()` not exceeded for PnL/position thresholds.
- `kill_switch()` tested in SIMULATION path; adapter disables orders.

## 6) Success Criteria
- 7 consecutive PAPER daily runs succeed (no retries needed).
- All daily logs present in `logs/scheduled/` and artifacts present in `logs/`.
- All storage partitions valid (non-empty Parquet for decisions/audits/stats each day).
- Weekly report generated at week end.
- No unexpected SAFE_MODE activations.
- Full test suite remains green throughout the week.

When all criteria are met, Phase 10 PAPER-week validation is complete.
