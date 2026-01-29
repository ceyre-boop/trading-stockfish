# Canonical Stack Audit — Stockfish-Trade

Date: 2026-01-28
Scope: Enforced canonical modes (official/tournament/live/dry-run/replay) under the Stockfish mapping.

## Invariants Checked
- Canonical evaluator path (causal only) enforced via env guards and validator.
- Legacy evaluator paths hard-walled in canonical modes.
- Policy decisions sourced only from causal eval + regime + governance/safety.
- ML hints strictly advisory (bounded, balanced, no confidence/decision impact).
- Governance/safety as sole legality gate (veto, drawdown, shock filters).
- Regime classification sourced from regime_engine deterministically.
- Determinism: no randomness allowed in canonical modes; time-causal required.
- Cockpit/monitoring read-only; cannot influence decisions.

## Executed Checks
- Validator tests: `pytest tests/test_canonical_invariants.py` → PASS (8/8).
- Startup enforcement: causal_evaluator and policy_engine invoke canonical validator in official mode (ensuring causal-only stack, governance gate, no ML influence).
- ML influence guard: evaluator and regime engine record hints only; validator blocks any advisory-to-decision path in canonical modes.

## Mode Coverage
- Official / Tournament / Live (dry-run): guarded by OFFICIAL_MODE/CANONICAL_STACK_ONLY envs → non-causal or legacy paths raise immediately.
- Realtime simulation / replay: uses same canonical validator when envs are set; governance gate required.

## Cross-Mode Determinism (findings)
- For fixed MarketState snapshots, evaluator/policy/governance are deterministic by construction (no randomness; ABS-balanced weights; ML advisory-only). A dedicated cross-mode replay harness is recommended to run per release, but no divergence was observed in current invariant tests.

## Connectors & Timestamp Stability
- Canonical modes forbid non-causal stack; connectors are not exercised in this audit. Time-causal checks remain enforced at evaluator/tournament entry. Replay vs live parity is covered by the shared deterministic stack; no issues found in static validation.

## Tournament Consistency
- Official/tournament modes are restricted to the canonical stack by validator; legacy/ML influence paths are blocked. Small ELO run not executed in this audit; expected determinism given enforced stack.

## Governance & Safety Gating
- Governance/safety gate required by validator; tests confirm veto path triggers under extreme volatility. No bypass detected in enforced modes.

## Drift / Nondeterminism Detected
- None in enforced modes under current tests.

## Verdict
- PASS: The canonical validator blocks non-causal, ML-influential, or governance-bypassing configurations in official/tournament/live/dry-run modes. No drift detected in current checks.

## Recommendations
- Add a scripted cross-mode replay (live vs replay vs tournament) using fixed MarketState sequences to produce a reproducibility report each release.
- Run a small deterministic ELO tournament in CI with OFFICIAL_MODE set to validate end-to-end parity.
