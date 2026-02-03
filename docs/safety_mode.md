# Safety Mode

## Triggers
- Drift thresholds exceeded (feature_drift_detector findings)
- Instability in stats (exploding variance, collapsing stability)
- Repeated gate failures within a window
- External/manual override

## Actions in SAFE_MODE
- Revert to last-good policy (archived known-good)
- Disable or downweight regime multipliers
- Reduce position size / risk scales
- Log safety event and chosen action

## Exit Conditions
- Drift back within bounds
- Stable performance over N consecutive periods/gates
- Manual override to NORMAL

## Example Scenarios
- Drift spike + two failed gates \u2192 enter SAFE_MODE, revert to last-good policy, zero multipliers.
- After three clean gates and no drift \u2192 exit SAFE_MODE, restore active multipliers.

# Safety Mode (Detailed)

Deterministic protection layer that constrains the engine when drift, instability, or repeated gate failures are detected. No randomness or heuristic overrides; given the same inputs, the safety decision is reproducible and offline-safe.

## Triggers
- Drift thresholds exceeded (feature_drift_detector findings) using drift_result.
- Instability in stats (exploding variance, collapsing stability scores from feature stats).
- Repeated policy gate failures across feedback runs (consecutive FAIL decisions).
- External/manual override.

## Actions in SAFE_MODE
- Revert to last-good policy (archived known-good path/version).
- Disable or downweight regime multipliers to reduce sensitivity to classification noise.
- Reduce position size / risk scales (session/risk scale < 1.0).
- Log a safety event with deterministic fields: timestamp_utc (Z-normalized), trigger reasons, applied actions, last_good reference.

## Safety State Model
- States: NORMAL, SAFE_MODE.
- Metadata tracked: last_good_policy path, version, timestamp_utc, trigger reasons, applied safety actions.

## Exit Conditions
- Drift back within acceptable bounds for the configured window.
- Stable stats over N consecutive periods (variance and stability within limits).
- Successful gate passes (candidate policies meet thresholds).
- Manual override to return to NORMAL.

## Examples
- Example 1: Drift spike triggers SAFE_MODE \u2192 revert to last-good policy.
- Example 2: Repeated gate failures trigger SAFE_MODE \u2192 regime multipliers reduced \u2192 exit after stability returns and gates pass.
