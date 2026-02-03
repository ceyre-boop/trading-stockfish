# Policy Gating (Evaluation Gate)

Deterministic gate that decides whether a candidate policy produced by the feedback loop is promoted or rejected. No randomness or heuristic overrides are allowed; identical inputs yield identical outcomes and reason lists.

## Metrics Evaluated
- Overall PnL: cumulative return over the evaluation window.
- Sharpe ratio: risk-adjusted performance using configured risk-free rate and return periodicity.
- Hit rate: fraction of trades that are profitable.
- Max drawdown: peak-to-trough loss over the window.
- Per-regime performance: PnL and hit rate segmented by macro regimes (e.g., RISK_ON/RISK_OFF) and session regimes (e.g., ASIA/LONDON/NEW_YORK/HIGH_VOL).
- Stability metrics: return variance, consistency across segments (e.g., rolling windows or deterministic splits), and absence of regime-specific collapse.

## Thresholds and Constraints
- Minimum Sharpe threshold: Sharpe >= `min_sharpe` (configured constant).
- Maximum allowed drawdown: max drawdown <= `max_drawdown`.
- Minimum hit rate: hit rate >= `min_hit_rate`.
- Per-regime constraints: for every regime slice, PnL and hit rate must stay above `regime_min_pnl` and `regime_min_hit_rate`; no slice may breach catastrophic thresholds such as `regime_max_drawdown`.
- Stability requirements: return variance <= `max_return_variance` and consistency >= `min_consistency`.

## Decision Logic
- PASS if all thresholds and constraints above are satisfied.
- FAIL if any single threshold is violated.
- Reasons are enumerated deterministically as a list of failed checks (e.g., `sharpe_below_min`, `drawdown_above_max`, `regime_hit_rate_breach_HIGH_VOL`).

## Examples
- Passing candidate: Sharpe 1.25 (>=1.0), max drawdown 6% (<=10%), hit rate 54% (>=50%), all regime slices positive PnL with hit rate >=50%, stability variance within bound — result: PASS.
- Failing candidate: Sharpe 0.70 (<1.0) while max drawdown 8% (<=10%) and hit rate 52% (>=50%); although most metrics pass, Sharpe breach triggers `sharpe_below_min` — result: FAIL.

## Determinism
- Inputs: candidate policy metrics and configured thresholds only.
- No stochastic sampling, randomness, or heuristic overrides.
- Identical inputs yield identical PASS/FAIL decision and identical ordered reason list.
