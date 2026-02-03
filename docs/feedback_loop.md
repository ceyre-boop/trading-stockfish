# Stats \u2192 Policy Feedback Loop (Nightly/Weekly)

## Inputs

## Steps (deterministic cadence)
1) Load audits and decision logs for the configured window (e.g., last N days).
2) Run drift analysis (existing feature_drift_detector).
3) Run stats harness (feature importance, stability, regime stats).
4) Propose updated policy (base weights, trust, regime multipliers).
5) Run evaluation gate (policy gating) on backtest/replay metrics.
6) If gate passes \u2192 promote new policy; if gate fails \u2192 keep last-good.
7) Archive old policy and write a feedback summary.

## Outputs

## Sequence (text diagram)

## Notes
# Stats → Policy Feedback Loop (Nightly/Weekly)

Deterministic, offline-only pipeline that turns audits and decision logs into an evaluated policy candidate. Live decision logic is untouched; outputs are written to disk for promotion/rollback.

## 1) Inputs
- Feature audits (existing artifacts with snapshots and issues)
- Decision logs (Phase 7 JSONL stream)
- Current policy_config.json (last-good policy)

## 2) Steps
1. Load audits + decision logs for the configured window (e.g., last N days).
2. Run drift analysis using the existing drift module.
3. Run the stats harness (feature importance, stability, regime stats).
4. Propose an updated policy (base weights, trust, regime multipliers).
5. Run the evaluation gate (policy gating) on backtest/replay metrics.
6. If PASS → promote the new policy.
7. If FAIL → keep last-good policy and log the rejection.
8. Archive the previous policy (always keep an audit trail).
9. Write a feedback summary (JSON/markdown) with drift, stats, gate decision, and promotion result.

## 3) Outputs
- New policy_config.json (if promoted; otherwise candidate is archived as rejected)
- Policy version metadata (run_id, timestamp_utc, source window)
- Feedback summary (JSON or markdown) with drift summary, stats summary, gate result, and policy paths

## 4) Sequence (text diagram)
- Scheduler/cron triggers `run_feedback_loop(run_id)`
- Load windowed audits + decision logs
- Drift analysis → `drift_result`
- Stats harness → `stats_result` (importance, stability, regime metrics)
- Policy proposal → `candidate_policy` (base_weights, trust, regime_multipliers)
- Backtest/replay → `backtest_result`
- Evaluation gate(`candidate_policy`, `backtest_result`) → PASS/FAIL + reasons
- If PASS: archive current policy; write candidate to policy_config.json
- If FAIL: keep last-good policy; log rejection reasons
- Write feedback summary to logs/feedback/feedback_run_<run_id>.json
- Archive trail ensures deterministic rollback

## Notes
- Deterministic: same inputs → same outputs.
- Offline: no live decision logic changes during the loop.
