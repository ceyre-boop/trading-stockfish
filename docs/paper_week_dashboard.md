# PAPER Week Dashboard (Operator)

Use this sheet for the daily 2-minute check during the 7-day PAPER validation.

## Daily Run Status
- [ ] Run date: ____
- [ ] Run ID: ____
- [ ] Mode: PAPER
- [ ] Task Scheduler triggered (daily_run): [ ] Yes [ ] No
- [ ] Exit code/logs clean (no errors/warnings of concern)

## Storage Growth
- Decisions partition appended (storage/decisions/decisions_YYYYMMDD.parquet): [ ] Yes [ ] No
- Audits partition appended (storage/audits/audits_YYYYMMDD.parquet): [ ] Yes [ ] No (expect >=1 row)
- Stats partition appended (storage/stats/stats_YYYYMMDD.parquet): [ ] Yes [ ] No
- Zero-byte or corrupted parquet detected: [ ] No issues [ ] Issue (note below)

## Guardrail State
- Kill-switch status: [ ] Disabled (expected) [ ] Triggered (investigate)
- Any guardrail alerts: [ ] None [ ] Present (capture evidence)

## SAFE_MODE State
- SAFE_MODE active: [ ] No (expected) [ ] Yes (investigate cause)
- Notes: ________________________________

## Research API Quick Summary
- Regime performance checked: [ ] Yes [ ] No (rows: ____)
- Drift check run: [ ] Yes [ ] No (notes: __________)
- Policy comparison (current vs prior): [ ] Reviewed [ ] Not reviewed

## Task Scheduler Confirmation
- Last run time (daily task): __________
- Next run time scheduled: __________
- Task state: [ ] Ready [ ] Running [ ] Disabled

## Notes / Follow-ups
- ______________________________________
- ______________________________________

## Day 2 Snapshot
- run_id: ______
- decisions count: ______
- audits count: ______ (expect >=1)
- stats present: [ ] Yes [ ] No
- SAFE_MODE: [ ] No [ ] Yes (investigate)
- anomalies: ______________________________________

## Day 3 Snapshot
- decisions count: 1
- audits count: 1
- stats present: [x] Yes [ ] No
- SAFE_MODE: [x] No [ ] Yes
- anomalies: Scheduler missed run; manual recovery executed
- scheduler: manual recovery run

## Day 4 Snapshot
- decisions count: 1
- audits count: 1
- stats present: [x] Yes [ ] No
- SAFE_MODE: [x] No [ ] Yes
- anomalies: Scheduler missed run; manual recovery executed
- scheduler: manual recovery run

## Day 5 Snapshot
- decisions count: 1
- audits count: 1
- stats present: [x] Yes [ ] No
- SAFE_MODE: [x] No [ ] Yes
- anomalies: Scheduler missed run; manual recovery executed
- scheduler: manual recovery run

## Day 4 Pre-run Checklist
- [ ] clock sync OK
- [ ] policy_config.json valid
- [ ] connectors valid
- [ ] scheduler enabled
- [ ] next-run time correct

## Day 5 Pre-run Checklist
- [ ] clock sync OK
- [ ] policy_config.json valid
- [ ] connectors valid
- [ ] scheduler enabled
- [ ] next-run time correct

## Day 6 Snapshot
- decisions count: 1
- audits count: 1
- stats present: [x] Yes [ ] No
- SAFE_MODE: [x] No [ ] Yes
- anomalies: Scheduler missed run; manual recovery executed
- scheduler: manual recovery run

## Day 6 Pre-run Checklist
- [ ] clock sync OK
- [ ] policy_config.json valid
- [ ] connectors valid
- [ ] scheduler enabled
- [ ] next-run time correct

## Day 7 Pre-run Checklist
- [ ] clock sync OK
- [ ] policy_config.json valid
- [ ] connectors valid
- [ ] scheduler enabled
- [ ] next-run time correct

## Day 7 Snapshot
- decisions count: 1
- audits count: 1 (expect >=1)
- stats present: [x] Yes [ ] No
- SAFE_MODE: [x] No [ ] Yes
- anomalies: Scheduler missed run; manual recovery executed
- scheduler: manual recovery run

## Weekly Cycle Pre-Check
- [ ] clock sync OK
- [ ] policy_config.json valid
- [ ] connectors valid
- [ ] days 1-7 storage continuity OK (decisions/audits/stats)
- [ ] SAFE_MODE inactive
- [ ] manual invocation ready: python scripts/prepare_weekly_cycle.py (scheduler not required)

## Weekly Cycle Result
- report generated: Yes (reports/weekly_report_2026-02-04.md)
- SAFE_MODE: [x] No [ ] Yes
- storage continuity: [x] OK (day1-day7 decisions/audits/stats present)
- anomalies: manual recoveries completed for days 4-7; weekly cycle clean
- scheduler: not used (manual invocation)
- operator notes: SIMULATION mode; no policy promotions detected; logs/scheduled/weekly_cycle_20260203.log; review logged in logs/scheduled/weekly_cycle_review.log

## Phase 10 Completion Summary
- 7/7 days validated (manual recoveries on days 4-7)
- Weekly cycle validated in SIMULATION
- No data corruption detected
- SAFE_MODE remained inactive
- System ready for Phase 11 hardening
