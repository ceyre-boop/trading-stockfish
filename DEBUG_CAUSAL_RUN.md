# DEBUG CAUSAL RUN - Complete Guide

## Overview

`--debug-causal-run` is a comprehensive debugging mode that provides **complete transparency** into the trading engine's decision-making process. Every candle's analysis is logged in detail, including:

- **All 8 Causal Factors** (score + reasoning)
- **Causal Evaluation Result** (market regime, score, confidence)
- **Policy Engine Decision** (action, conviction zone, sizing)
- **Risk State** (position size, entry price, stop/target levels)
- **Complete Reasoning Chain** (why each decision was made)

## Purpose

Debug Causal Run is designed for:

1. **Development & Tuning**
   - Understand why trades are being taken/avoided
   - Identify which causal factors dominate decisions
   - Tune factor weights and thresholds
   - Test policy engine behavior under various market conditions

2. **Validation & Verification**
   - Verify time-causality (no lookahead bias)
   - Confirm deterministic behavior
   - Trace decision logic step-by-step
   - Compare expected vs actual trades

3. **Transparency & Explainability**
   - Full audit trail of every decision
   - Export to stakeholders/traders
   - Document trading logic for compliance
   - Investigate specific market conditions

4. **Performance Analysis**
   - Identify which causal factors drive wins/losses
   - Find regime-specific weaknesses
   - Optimize entry/exit criteria
   - Backtest hypothesis changes

## Quick Start

### Basic Usage

```bash
# Run debug mode on real data
python analytics/run_elo_evaluation.py \
  --debug-causal-run \
  --data-path data/ES_1m.csv \
  --symbol ES \
  --timeframe 1m
```

### Auto-Cascading Flags

When you use `--debug-causal-run`, these flags are **automatically enabled**:

```python
--real-tournament           # Use real tournament mode
--official-tournament       # Strict time-causal enforcement
--causal-eval              # Deterministic causal evaluation
--verbose                  # Full output logging
```

**You do NOT need to specify these separately** - they cascade automatically.

### With Optional Filters

```bash
# Debug a specific date range
python analytics/run_elo_evaluation.py \
  --debug-causal-run \
  --data-path data/ES_1m.csv \
  --symbol ES \
  --timeframe 1m \
  --start 2023-01-01 \
  --end 2023-01-31

# Save results to JSON in addition to log
python analytics/run_elo_evaluation.py \
  --debug-causal-run \
  --data-path data/ES_1m.csv \
  --symbol ES \
  --timeframe 1m \
  --output results.json
```

## Output Files

### Debug Log File

**Location:** `logs/debug_causal_run_<SYMBOL>_<TIMEFRAME>_<TIMESTAMP>.log`

**Example:** `logs/debug_causal_run_ES_1m_20240115_143025.log`

### Log Format

Each candle produces structured output:

```
================================================================================
DEBUG CAUSAL RUN LOG
================================================================================
Symbol: ES
Timeframe: 1m
Started: 2024-01-15 14:30:25
Mode: Deterministic Causal Evaluation + Policy Engine
================================================================================

[2024-01-15 14:30:00] Price=4500.25 Vol=       1250000 | Score=0.732 | 
  Factors: trend=0.85 momentum=0.72 volatility=0.45 volume=0.88 mean_reversion=0.60 
  sentiment=0.70 liquidity=0.90 regime=0.65 | 
  Regime=BULL | Zone=STRONG_SIGNAL | Action=BUY | Position=1@4499.50
  Reasoning: Strong uptrend with good liquidity and high volume
  Risk: SL=4495.00 TP=4510.00 RR=2.00

[2024-01-15 14:31:00] Price=4502.10 Vol=       980000 | Score=0.685 | 
  Factors: trend=0.80 momentum=0.65 volatility=0.50 volume=0.75 mean_reversion=0.58 
  sentiment=0.68 liquidity=0.85 regime=0.70 | 
  Regime=BULL | Zone=MODERATE_SIGNAL | Action=HOLD | Position=1@4499.50
  Reasoning: Momentum declining, maintain position
  Risk: SL=4495.00 TP=4510.00 RR=2.00

...

================================================================================
FINAL TOURNAMENT SUMMARY
================================================================================
Total Trades: 47
Winning Trades: 31
Win Rate: 65.96%
Total Return: 12.34%
Sharpe Ratio: 1.85
Max Drawdown: -8.50%
Ended: 2024-01-15 15:45:30
================================================================================
```

## Understanding the Output

### Causal Factors (8 Components)

The engine evaluates 8 independent causal factors for every candle:

| Factor | Range | Meaning | High = | Low = |
|--------|-------|---------|--------|-------|
| **trend** | 0.0-1.0 | Direction strength | Strong uptrend | Weak/sideways |
| **momentum** | 0.0-1.0 | Velocity of price | Fast acceleration | Deceleration |
| **volatility** | 0.0-1.0 | Market uncertainty | High uncertainty | Calm/stable |
| **volume** | 0.0-1.0 | Participation level | High volume | Low volume |
| **mean_reversion** | 0.0-1.0 | Deviation from MA | Oversold/overbought | At equilibrium |
| **sentiment** | 0.0-1.0 | Macro/news bias | Bullish backdrop | Bearish backdrop |
| **liquidity** | 0.0-1.0 | Spread/depth | Wide spread | Tight spread |
| **regime** | 0.0-1.0 | Market condition | Bull/high vol | Bear/low vol |

### Evaluation Score

**Overall Score (0.0 - 1.0):** Combined confidence from all 8 factors

- **0.85-1.00:** Very high confidence - Strong signal
- **0.70-0.85:** High confidence - Moderate signal  
- **0.55-0.70:** Medium confidence - Weak signal
- **0.40-0.55:** Low confidence - Uncertain
- **0.00-0.40:** Very low confidence - No signal

### Conviction Zones (Policy Engine)

After evaluation, the policy engine maps confidence to trading conviction:

| Zone | Score Range | Position Size | Risk Level | When Used |
|------|-------------|----------------|-----------|-----------|
| **STRONG_SIGNAL** | 0.80-1.00 | 100% max | High | Very confident setups |
| **MODERATE_SIGNAL** | 0.65-0.80 | 50-75% max | Medium | Good conditions |
| **WEAK_SIGNAL** | 0.50-0.65 | 25-50% max | Low | Marginal setups |
| **UNCERTAIN** | 0.40-0.50 | 10-25% max | Very Low | Conflicting signals |
| **NO_SIGNAL** | 0.00-0.40 | 0% (no trade) | None | Skip entirely |

### Actions

The policy engine produces one of 8 actions per candle:

```
BUY                  - Open long position (full sizing)
SELL_SHORT           - Open short position (full sizing)
ADD_TO_POSITION      - Pyramid existing long
ADD_TO_SHORT         - Pyramid existing short
CLOSE_POSITION       - Exit long (take profit)
CLOSE_SHORT          - Exit short (take profit)
STOP_OUT             - Emergency exit (stop loss)
HOLD                 - Keep position, no action
```

### Risk Metrics

Every position tracks:

```
SL (Stop Loss)       - Price where to exit if wrong
TP (Take Profit)     - Price where to take profit if right
RR (Risk/Reward)     - Ratio of potential profit to loss
Position Size        - Current exposure
Entry Price          - Where position was entered
```

## Interpretation Examples

### Example 1: Strong Entry Setup

```
[2024-01-15 14:30:00] Price=4500.25 Vol=1250000 | Score=0.823 |
  Factors: trend=0.85 momentum=0.87 volatility=0.42 volume=0.92 
  mean_reversion=0.65 sentiment=0.78 liquidity=0.88 regime=0.71 |
  Regime=BULL | Zone=STRONG_SIGNAL | Action=BUY | Position=1@4500.00
  Reasoning: Strong uptrend with excellent momentum and volume
  Risk: SL=4495.00 TP=4510.00 RR=2.00
```

**Analysis:**
- Score 0.823 = Very high confidence
- Most factors elevated (trend, momentum, volume, liquidity, sentiment)
- STRONG_SIGNAL zone = Full position size
- Good risk/reward ratio (2.0 = make $2 for every $1 at risk)
- **Decision: GOOD SETUP** - All systems aligned

### Example 2: Conflicting Signals

```
[2024-01-15 14:31:00] Price=4502.10 Vol=980000 | Score=0.542 |
  Factors: trend=0.65 momentum=0.48 volatility=0.72 volume=0.58 
  mean_reversion=0.45 sentiment=0.55 liquidity=0.68 regime=0.52 |
  Regime=UNCERTAIN | Zone=UNCERTAIN | Action=HOLD | Position=0@0.00
  Reasoning: Conflicting signals - momentum declining, volatility rising
  Risk: N/A
```

**Analysis:**
- Score 0.542 = Low confidence
- Mixed factors - some bullish (trend=0.65), some bearish (momentum=0.48, mean_reversion=0.45)
- High volatility (0.72) = uncertain environment
- UNCERTAIN zone = No trade
- **Decision: SKIP** - Wait for clarity

### Example 3: Deteriorating Position

```
[2024-01-15 14:32:00] Price=4498.50 Vol=750000 | Score=0.380 |
  Factors: trend=0.35 momentum=0.22 volatility=0.85 volume=0.38 
  mean_reversion=0.72 sentiment=0.25 liquidity=0.60 regime=0.30 |
  Regime=BEAR | Zone=NO_SIGNAL | Action=CLOSE_POSITION | Position=0@0.00
  Reasoning: Trend broken, momentum collapse, stop loss triggered
  Risk: SL=4495.00 TP=4510.00 RR=0.00 (closed)
```

**Analysis:**
- Score 0.380 = Very low confidence (bearish reversal)
- Momentum collapsed (0.22), sentiment turned bearish (0.25)
- Trend weakened (0.35), mean reversion triggered (0.72)
- NO_SIGNAL zone = Exit immediately
- Position closed for small loss (stayed above SL at 4495)
- **Decision: PROTECT CAPITAL** - Exit and reassess

## How to Use for Development

### 1. Identify Weak Signals

Look for trades in WEAK_SIGNAL or UNCERTAIN zones that lose money:

```bash
grep "Zone=WEAK_SIGNAL" logs/debug_causal_run_ES_1m_*.log | grep -v "HOLD"
```

**Question:** Are these trades harmful? Consider filtering them out.

### 2. Find Dominant Factors

Find which factors most often lead to winning trades:

```bash
# Find winning trades
grep -B1 "Action=BUY\|Action=SELL_SHORT" logs/debug_causal_run_ES_1m_*.log | \
  grep "Factors:" | head -20
```

**Insight:** Do all winners have high volume? High liquidity? Adjust weights accordingly.

### 3. Test Factor Changes

1. Edit causal factor weights in `engine/causal_evaluator.py`
2. Run debug mode again
3. Compare logs and results
4. Iterate until satisfied

### 4. Validate Risk Management

Check all positions have appropriate stops/targets:

```bash
grep "Risk:" logs/debug_causal_run_ES_1m_*.log | head -20
```

Ensure:
- Stop loss is always set (not N/A)
- Risk/Reward ≥ 1.5 for most trades
- No extreme position sizes

### 5. Find Edge Opportunities

Look for specific patterns that consistently win:

```bash
# Find all STRONG_SIGNAL entries
grep "Zone=STRONG_SIGNAL" logs/debug_causal_run_ES_1m_*.log | head -30
```

**Question:** What do they have in common? Can we enter earlier?

## Performance Tuning Checklist

### ✓ Before Production

- [ ] Review all 8 causal factors - do weights make sense?
- [ ] Check conviction zone thresholds - are they appropriate?
- [ ] Validate risk management - stops/targets reasonable?
- [ ] Test regime switching - bull/bear handling correct?
- [ ] Verify determinism - same input = same output?
- [ ] Audit lookahead - no future data in calculations?
- [ ] Stress test - what happens in crashes?
- [ ] Monitor edge cases - gaps, volatility spikes, etc.

### ✓ Common Optimizations

1. **Increase threshold for STRONG_SIGNAL**
   - Fewer but higher-quality trades
   - Improve win rate, reduce frequency

2. **Lower threshold for WEAK_SIGNAL**
   - More trading opportunities
   - Increase frequency, possibly lower win rate

3. **Adjust volatility penalty**
   - High volatility = reduce position size
   - Low volatility = increase position size

4. **Tune mean reversion threshold**
   - Higher = only trade extreme oversold/overbought
   - Lower = trade earlier reversals

5. **Weight sentiment more heavily**
   - In trending markets
   - During news events

## Troubleshooting

### "No log file created"

Check:
1. `logs/` directory exists (created automatically)
2. Read/write permissions on logs directory
3. Data path is valid
4. No errors in console output

### "All trades are HOLD"

Likely causes:
1. Causal evaluation score too low for thresholds
2. Data quality issues (missing volume, bad prices)
3. Conviction zone thresholds too high
4. Try with `--verbose` flag for more debugging

### "Log shows good decisions but trades lose money"

Check:
1. Risk management - are stops/targets appropriate?
2. Regime switching - is regime detection working?
3. Slippage assumptions - real execution worse than backtester?
4. Sample bias - is this a statistical anomaly?

### "Evaluation score always 0.5 (average)"

Likely causes:
1. Factors not implemented properly
2. All factors weighted equally
3. Data has no clear patterns
4. Evaluation period too short (need ≥100 candles)

## Advanced: Custom Analysis

### Export Log for Further Analysis

```python
import re
import pandas as pd

log_file = 'logs/debug_causal_run_ES_1m_20240115_143025.log'
candles = []

with open(log_file) as f:
    for line in f:
        if 'Price=' in line:
            # Parse: [2024-01-15 14:30:00] Price=4500.25 Vol=1250000 | Score=0.732
            match = re.search(r'\[(.+?)\].*Price=(\d+\.\d+).*Score=([\d\.]+)', line)
            if match:
                candles.append({
                    'timestamp': match.group(1),
                    'price': float(match.group(2)),
                    'score': float(match.group(3))
                })

df = pd.DataFrame(candles)
print(df.describe())
```

### Generate Trading Plan Report

Create a summary of decision patterns:

```python
# Find best and worst trading periods
df['4h'] = pd.to_datetime(df['timestamp']).dt.hour // 4
hourly_score = df.groupby('4h')['score'].mean()
print("Best trading window (by avg score):")
print(hourly_score.idxmax(), "→ Score", hourly_score.max())
```

## Related Documentation

- [POLICY_ENGINE.md](POLICY_ENGINE.md) - Policy engine decision logic
- [POLICY_ENGINE_INTEGRATION.md](POLICY_ENGINE_INTEGRATION.md) - Integration details
- [RUN_ELO_EVALUATION.md](RUN_ELO_EVALUATION.md) - Full CLI reference
- [CAUSAL_EVALUATOR.md](CAUSAL_EVALUATOR.md) - Causal evaluation theory

## Summary

`--debug-causal-run` provides production-grade transparency for trading system development:

✅ **Complete reasoning audit** - Every candle analyzed in detail
✅ **Deterministic & causal** - No lookahead, reproducible results
✅ **Risk-aware** - Position sizing and stop/target management
✅ **Explainable** - Clear decision logic for every trade
✅ **Development-friendly** - Easy to identify optimization opportunities

Use it to understand, validate, and improve your trading system before deploying to live markets.

---

**Version:** 1.0 | **Status:** Production Ready | **Mode:** Deterministic Causal Evaluation
