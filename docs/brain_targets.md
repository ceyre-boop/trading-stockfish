# Brain Targets (Phase 12 – Layer 4 Dataset Scaffolding)

This document defines the ML-ready dataset produced from enriched decision logs. No models or runtime behavior are included in this layer.

## Columns
- **strategy_id**: Strategy identifier (required for ML rows)
- **entry_model_id**: Entry model identifier
- **session** / **macro** / **vol** / **trend** / **liquidity** / **tod**: Expanded from `condition_vector`
- **market_profile_state / market_profile_confidence**: A/M/D state and confidence from structure_brain
- **session_profile / session_profile_confidence**: Session profile classifier output
- **liquidity_bias_side**: Bias derived from liquidity frame (UP/DOWN/NEUTRAL)
- **nearest_target_type / nearest_target_distance**: Liquidity target selected by the liquidity frame and its distance
- **liquidity_sweep_flags**: Sweep flags map from the liquidity frame (if present)
- **displacement_score**: Displacement signal propagated from market profile evidence when available
- **entry_brain_label / entry_brain_prob_good / entry_brain_expected_R**: Shadow tactical brain policy annotations for the chosen entry model (no runtime gating in this layer)
- **sample_size**: Number of decisions in the group
- **win_rate**: Fraction of outcomes > 0
- **avg_reward**: Mean outcome (reward/R-multiple)
- **reward_variance**: Variance of outcome (ddof=0)
- **reward_std**: Standard deviation of outcome (ddof=0)
- **stability_mean_5**: Rolling mean (window=5) of outcome within the group (if timestamps available)
- **stability_std_5**: Rolling std (window=5) of outcome within the group (if timestamps available)

## Grouping Keys
The dataset aggregates by:
- strategy_id
- entry_model_id
- session, macro, vol, trend, liquidity, tod (from condition_vector)

## Metric Notes
- Rows with missing strategy_id, entry_model_id, or outcome are dropped before grouping.
- Condition vectors are expanded into flat columns for deterministic grouping.
- Stability metrics are computed over time within each group when `timestamp_utc` is available (rolling window=5, deterministic ordering).
- Sorting of the final dataset is deterministic on the grouping keys.

## Future ML Usage
- These aggregates provide priors for the ML brain: performance by strategy and entry model under specific conditions.
- Future layers may add attribution, weighting, and model training using these columns. This layer stops at dataset construction.

## Classifier Target (label_good)
- Derived boolean label used by the viability classifier.
- Default rule: label_good = (win_rate >= win_rate_min) AND (avg_reward >= avg_reward_min).
- Threshold defaults (Phase 12 scaffolding): win_rate_min = 0.5, avg_reward_min = 0.0. Thresholds are configurable per training run.
- If a label_good column already exists in the dataset, it is used directly without recomputation.

## Regression Target (expected_reward)
- Target for the reward regressor is the aggregated avg_reward column from the dataset.
- This approximates the expected return for the combo (strategy, entry model, and condition bucket).

## Thresholds
- win_rate_min, avg_reward_min: Define viability boundary for label_good if not already supplied.
- sample_size_warning: Trigger flag in scoring when sample size is below this minimum.
- prob_good_warning: Trigger flag in scoring when classifier probability is below this floor.

## BrainScore Output
- prob_good: Classifier probability that the combo is viable.
- expected_reward: Predicted reward from the regressor.
- sample_size: Propagated sample size for the scored combo (falls back to zero when unknown).
- flags: Diagnostic hints (e.g., low_sample, low_confidence, high_variance) produced during scoring.
