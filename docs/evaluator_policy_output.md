# Evaluator Output with Policy Overlay

When a `policy_config.json` is provided, the causal evaluator output includes:

- `policy_applied`: boolean
- `causal_reasoning`: list of factor entries with
  - `factor`: factor name
  - `score`: raw factor score
  - `weight`: original factor weight
  - `policy_base_weight`: base policy weight (regime-agnostic, default 1.0)
  - `policy_weight`: effective policy weight (base * trust * regime multipliers)
  - `regime_multiplier`: multiplicative adjustments for active regimes (default 1.0)
  - `trust_score`: policy trust (default 1.0)
  - `weighted_score`: score * weight * policy_weight (0 if trust_score==0)

`eval_score` reflects the policy-weighted aggregation (clipped to [-1, 1]). Confidence, thresholds, and regime logic remain unchanged.

Determinism: factors are sorted by name when policy is applied.
