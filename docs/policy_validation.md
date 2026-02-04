# Policy Validation for PAPER Week

Policy validation is mandatory before the 7-day PAPER run. A bad policy config can break SAFE_MODE, drift windows, gating, and storage consistency. Validate once before Day 1 and after any policy change.

## How to Run
From project root:
```bash
python scripts/validate_policy_config.py
```
(Uses local files only; no external calls.)

## Expected PASS Output
```
PASS: policy_config.json is valid for PAPER
```
Exit code 0.

## Failure Handling
If you see `FAIL`:
- Missing file: ensure `policy_config.json` exists in project root.
- Missing fields: add required keys (`policy_version`, `base_weights`, `trust`, `regime_multipliers`, `metadata`, optional `safe_mode`).
- Deprecated/unknown fields: remove or migrate them to `metadata` if appropriate.
- Non-finite numbers: ensure all weights/multipliers/trust values are finite (no NaN/inf).
- Trust range: keep trust values within [0.0, 5.0].
- Regime coverage: ensure `regime_multipliers` defines every regime in `docs/session_regimes.md`.
- SAFE_MODE: include `safe_mode` (bool/dict/state) either top-level or under `metadata`.

Re-run after fixing; proceed with PAPER week only on PASS.
