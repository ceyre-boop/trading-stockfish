# Canonical Invariants (Stockfish-Trade)

These invariants are enforced to prevent architectural drift and to mirror the
Stockfish model in official/tournament/live modes.

- **Deterministic evaluator/policy/regime**: No randomness; rule-based only.
- **Canonical evaluator path**: CausalEvaluator + PolicyEngine only when
  OFFICIAL_MODE/CANONICAL_STACK_ONLY are active; legacy evaluator is blocked.
- **Safety/governance gate**: Governance/safety is the sole legality/pruning gate
  before execution.
- **ML advisory-only**: ML hints are bounded, balanced, and cannot alter
  decisions or confidence in canonical modes.
- **Time-causal tournaments**: Official/tournament runs must use the canonical
  stack with time-causal data.
- **Cockpit read-only**: Monitoring/observability modules cannot influence
  decisions.
- **Realtime loop parity**: Realtime loop mirrors Stockfish search loop ordering
  (state → eval → policy → governance → execution).

Environment guards:
- Set `OFFICIAL_MODE=1` (or `CANONICAL_STACK_ONLY=1`) to enforce canonical-only
  operation. The canonical stack validator will refuse non-causal evaluators,
  ML overrides, or bypassing governance in these modes.
