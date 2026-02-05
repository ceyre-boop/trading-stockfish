# Strategy Catalog (Phase 12 – Layer 1 Vocabulary)

This catalog enumerates canonical strategy families and their representative behaviors. It is documentation-only scaffolding for Phase 12 and does not change runtime behavior.

## Trend Strategies
- **Examples:** Moving average crossovers, slope filters, higher-highs/higher-lows
- **Inputs:** Moving averages, slope, price structure
- **Conditions:** Trend regime, session, volatility regime
- **Entry Models:** Breakout, pullback, continuation
- **Exit Models:** Stop/target, trailing, regime-flip

## Momentum Strategies
- **Examples:** RSI, ROC, breakout strength
- **Inputs:** RSI, ROC, momentum indicators
- **Conditions:** Momentum regime, session, volatility regime
- **Entry Models:** Breakout, continuation
- **Exit Models:** Stop/target, time-based, trailing

## Mean Reversion Strategies
- **Examples:** Z-score bands, Bollinger touches, intraday extremes
- **Inputs:** Z-score, Bollinger bands, deviation metrics
- **Conditions:** Low-volatility, range-bound regimes
- **Entry Models:** Fade, pullback
- **Exit Models:** Stop/target, time-based

## Volatility Strategies
- **Examples:** ATR expansion/contraction, range breakouts
- **Inputs:** ATR, range, volatility metrics
- **Conditions:** Volatility regime (LOW/NORMAL/HIGH)
- **Entry Models:** Breakout, continuation
- **Exit Models:** Trailing, regime-flip

## Structure Strategies
- **Examples:** Session opens, VWAP, PDH/PDL, liquidity zones
- **Inputs:** VWAP, session markers, previous day levels
- **Conditions:** Session regime, time-of-day buckets
- **Entry Models:** Breakout, fade
- **Exit Models:** Stop/target, time-based
