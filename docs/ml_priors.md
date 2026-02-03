# ML Priors for Offline Training

## ML Targets (priors)
- `macro_up_prob`: Probability that macro regime bias moves upward/"risk-on" in the lookahead window.
- `volatility_spike_prob`: Probability that realized volatility exceeds a fixed threshold in the lookahead window.
- `regime_transition_prob`: Probability that session/macro regime changes to a different state in the next window.
- `directional_confidence`: Confidence that the model’s directional action (LONG/SHORT) is correct, proxied via sign-correct or PnL-normalized correctness.

## Label Definitions (from decisions storage)
Labels are derived deterministically from Parquet storage outputs via the research API; no external data is required.
- `macro_up_prob` label:
  - Compute future macro regime direction using `macro_regimes` progression or a macro bias signal (e.g., next `macro_regime` vs current).
  - Lookahead windows: 1 bar, 5 bars, and 1 session variants.
  - Label = 1 if future regime indicates higher risk-on state; else 0.
- `volatility_spike_prob` label:
  - Use outcome volatility proxy: realized variance or max drawdown over the lookahead.
  - Lookahead windows: 1 bar, 5 bars, 1 session.
  - Label = 1 if realized volatility > threshold (configurable, deterministic); else 0.
- `regime_transition_prob` label:
  - Compare current `session_regime` / `macro_regimes` to the next window’s regime.
  - Lookahead windows: next bar, next 5 bars, next session.
  - Label = 1 if regime changes; else 0.
- `directional_confidence` label:
  - Base: sign correctness (PnL > 0) or scaled PnL / |P&L| cap.
  - Lookahead windows: 1 bar, 5 bars, 1 session.
  - Label = continuous confidence score or binary correctness depending on downstream use; deterministic formula only.

## Input Features
- Canonical logged features (`feature_vector` / `key_features` in decisions storage) aligned to feature registry.
- Regime context: `session_regime`, `macro_regimes`, `timeframe`, policy/version metadata for provenance-aware training splits.
- Optional lagged outcomes: prior `outcome_pnl`, `outcome_hit`, `outcome_max_drawdown` for fixed lags (e.g., 1, 3, 5 observations) computed deterministically.
- Stability/variance metrics from stats storage: `stability`, `variance`, per-feature/per-regime metrics to act as priors or covariates.
- Engine-controlled identifiers for grouping: `run_id`, `policy_version`, `engine_version` for stratified splits.

## Determinism Requirements
- Label generation is pure and reproducible from stored Parquet data; no randomness or online calls.
- Thresholds, lookahead windows, and lag definitions are explicit, versioned, and configuration-driven.
- Feature/label extraction uses fixed ordering and seeded-free computations; any normalization is deterministic.
- No writes occur during research/labeling pipelines; outputs are derived in-memory or written to controlled artifacts with full provenance.
