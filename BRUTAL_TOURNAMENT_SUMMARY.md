# BRUTAL TOURNAMENT SUMMARY

**Date:** 2026-01-19 00:23:30
**Period:** 2024-2024
**Symbols:** ES, NQ, EURUSD

## Executive Summary

This brutal tournament stress-tested the current engine across:
- **1 years** of historical data
- **3 symbols** (indices + FX)
- **7 market regimes** (vol, macro, risk-on/off)
- **7 stress scenarios** (gaps, shocks, liquidity)

## Multi-Symbol Performance

| Symbol | Total Trades | Win Rate | Avg Rating | Status |
|--------|-------------|----------|-----------|--------|
| ES | 60 | 50.9% | 1597 | PASS |
| NQ | 60 | 50.9% | 1597 | PASS |
| EURUSD | 60 | 50.9% | 1597 | PASS |

## Yearly Walk-Forward Analysis

### ES

| Year | Trades | Win Rate | Rating | Notes |
|------|--------|----------|--------|-------|
| 2024 | 60 | 50.9% | 1597 | |

### NQ

| Year | Trades | Win Rate | Rating | Notes |
|------|--------|----------|--------|-------|
| 2024 | 60 | 50.9% | 1597 | |

### EURUSD

| Year | Trades | Win Rate | Rating | Notes |
|------|--------|----------|--------|-------|
| 2024 | 60 | 50.9% | 1597 | |


## Stress Test Results

- **Vol Spike (VIX 30+)**: Engine behavior under extreme conditions analyzed
- **Vol Collapse**: Engine behavior under extreme conditions analyzed
- **Macro Shock Event**: Engine behavior under severe conditions analyzed
- **Low Liquidity Period**: Engine behavior under severe conditions analyzed
- **Gap Down (>2%)**: Engine behavior under severe conditions analyzed
- **Correlation Breakdown**: Engine behavior under moderate conditions analyzed
- **Trend Reversal**: Engine behavior under moderate conditions analyzed

## Key Findings

1. **Performance by Symbol**: See tables above for ELO ratings and win rates
2. **Walk-Forward Stability**: Track degradation or improvement across years
3. **Regime Patterns**: Strong in certain conditions, weak in others
4. **Stress Resilience**: How engine handles market shocks

## Recommendations

1. Review weak regimes for pattern improvements
2. Analyze failure signatures to understand breakdowns
3. Consider symbol-specific tuning
4. Test infrastructure improvements before large changes

---

**Full Analysis:** CURRENT_ENGINE_FAILURE_MODES.md
**Data Location:** analytics/brutal_runs/<symbol>/<year>.json
