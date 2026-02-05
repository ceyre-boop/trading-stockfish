# Brain Policy (Phase 12 – Layer 5)

Discrete policy labels derived from ML brain outputs. This layer is scaffolding only; no runtime integration is included.

## Labels
- **PREFERRED**: Meets or exceeds probability and expected reward thresholds with sufficient samples.
- **ALLOWED**: Meets probability threshold but expected reward is below the preferred floor.
- **DISCOURAGED**: Probability of viability is below the minimum threshold (but sample size is sufficient).
- **DISABLED**: Insufficient sample size for the combo.

## Labeling Rules
Given thresholds `{min_samples, prob_good_min, expected_reward_min}`:
1. If `sample_size < min_samples` ⇒ `DISABLED`
2. Else if `prob_good < prob_good_min` ⇒ `DISCOURAGED`
3. Else if `expected_reward < expected_reward_min` ⇒ `ALLOWED`
4. Else ⇒ `PREFERRED`

## Inputs
- ML dataset rows (from `build_brain_dataset`) with strategy/entry identifiers, condition buckets, and aggregate metrics.
- Brain model artifacts (classifier + regressor + encoders) from `train_brain_models`.

## Artifact
- Written to `storage/policies/brain/brain_policy.json` (or threshold-configured output directory).
- Contents:
  - `metadata`: version timestamp, thresholds used, training metadata, model types.
  - `policy`: rows with `strategy_id`, `entry_model_id`, condition fields, `prob_good`, `expected_reward`, `sample_size`, `label`.

## Determinism
- No randomness in labeling; classifier/regressor outputs are deterministic given fixed artifacts.
- Rows are sorted by `(strategy_id, entry_model_id, session, macro, vol, trend, liquidity, tod)` for stable ordering.

## Future Evaluator Use
- Evaluator can consult labels to gate or prioritize strategies/entries under specific market conditions.
- Policy versioning via timestamp allows reproducibility and auditing across training windows.

## Shadow Brain Mode (Phase 12 – Layer 6)
- Evaluator loads `brain_policy.json` (read-only) and looks up the combo `(strategy_id, entry_model_id, session, macro, vol, trend, liquidity, tod)`.
- Recommendations are attached to decision metadata:
  - `brain_label`
  - `brain_prob_good`
  - `brain_expected_reward`
  - `brain_sample_size`
- Behavior is unchanged: no filtering, gating, weighting, or action changes. Output is advisory and logged for validation.

## Brain-Influenced Evaluation (Phase 12 – Layer 6 Part 2)
- Evaluator can optionally weight scores using `brain_label` when brain config is enabled.
- Label effects (deterministic, bounded):
  - `DISABLED`: strategy excluded (score forced to 0 under matched conditions).
  - `DISCOURAGED`: score multiplied by `discouraged_penalty`.
  - `ALLOWED`: no change.
  - `PREFERRED`: score multiplied by `preferred_boost`.
- Guardrails:
  - `SAFE_MODE` suppresses all brain influence and is logged.
  - Risk-limit or policy gating cannot be overridden by brain outputs.
  - Minimum thresholds (`min_sample_size`, `min_prob_good`, `min_expected_reward`) must be met or the combo is treated as disabled.
- Logging:
  - Debug line `Brain influence: strategy_id=X label=Y score_before=... score_after=...`.
  - Decision metadata/logs include `brain_influence_applied` and `brain_adjusted_score`.
- Config toggle: disable brain influence via `config/policy_config.json` (`brain.enabled=false`) or by running in `SAFE_MODE`.
