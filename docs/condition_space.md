# Condition Space (Phase 12 – Layer 2 Vocabulary)

The condition space defines a deterministic, discretized context vector that the future ML brain will consume. This document is scaffolding only; no runtime behavior changes.

## Axes

1. **session_regime**
   - Examples: RTH, ETH, OPEN, CLOSE, LUNCH, OVERNIGHT
   - Derived from existing session classification (e.g., session_regime classifier) when available; otherwise defaults to observed session tags.

2. **macro_regime**
   - Examples: RISK_ON, RISK_OFF, NEUTRAL
   - Use existing macro regime signals when present; otherwise fallback to placeholder or engine macro flags.

3. **volatility_regime**
   - Values: LOW / NORMAL / HIGH
   - Derived from ATR, realized volatility, or existing volatility regime logic.

4. **trend_regime**
   - Values: UP / DOWN / FLAT
   - Derived from slope, moving averages, or existing trend regime logic (e.g., trend_direction / trend_regime).

5. **liquidity_bucket**
   - Values: LOW / NORMAL / HIGH
   - Derived from spread, depth, volume, or existing liquidity proxies.

6. **time_of_day_bucket**
   - Examples: OPEN, MORNING, MIDDAY, AFTERNOON, CLOSE
   - Derived from timestamps or session-based time segmentation.

## ConditionVector Format

```json
{
  "session": "<session_regime>",
  "macro": "<macro_regime>",
  "vol": "<volatility_regime>",
  "trend": "<trend_regime>",
  "liquidity": "<liquidity_bucket>",
  "tod": "<time_of_day_bucket>"
}
```

## Usage Notes
- Each axis is discretized using existing engine signals where available; otherwise deterministic fallbacks.
- The ConditionVector will be used by the Phase 12 brain for context-aware strategy selection and attribution.
- No ML, no dataset builder, and no runtime behavior changes are introduced in this layer.
