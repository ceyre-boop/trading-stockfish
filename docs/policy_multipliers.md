# Regime-Conditioned Policy Multipliers

Deterministic scaling factors derived from observed performance by regime. They shape policy construction; live decision logic stays unchanged.

## 1) Regime Types
- Macro/volatility: HIGH_VOL, LOW_VOL, MACRO_ON, MACRO_OFF.
- Session: ASIA, LONDON, LONDON_NY_OVERLAP, NY, PRE_SESSION, POST_SESSION, NO_SESSION (engine may also emit OVERLAP variants).
- Internal engine regimes: liquidity/trend/microstructure labels exposed by the regime engine; additional overlaps may be added if documented.

## 2) Multiplier Semantics
- Multipliers scale base weights and/or trust and are applied multiplicatively: effective = base x trust x regime_mult x session_mult.
- Deterministic and reproducible: the same StatsResult input yields the same multipliers.
- Derived from stats harness outputs only; no heuristic or manual tuning.

## 3) Safety Bounds
- Configurable caps/floors (typical bounds: min=0.5, max=2.0).
- Bounds prevent runaway amplification, reduce sensitivity to short windows, and keep evaluator weights stable.

## 4) Source of Multipliers
- Computed from feature performance by regime (e.g., Sharpe, hit rate, stability) present in the StatsResult emitted during the feedback loop.
- Positive relative performance in a regime leads to multiplier > 1.0; negative relative performance leads to multiplier < 1.0, both clipped to bounds.

## 5) Worked Examples
- HIGH_VOL strong momentum: base=1.0, trust=1.0, stats show momentum Sharpe is high in HIGH_VOL -> regime_mult=1.6 -> effective=1.6.
- MACRO_OFF weak mean_reversion: base=1.0, trust=0.9, stats show underperformance in MACRO_OFF -> regime_mult=0.7 -> effective=0.63.

## 6) Structure and Alignment
- Stored under policy_config.regime_multipliers as { regime: { feature: multiplier } }.
- Session multipliers can be layered via session context/modifiers following the same multiplicative rule.
- Evaluator uses effective weight = base x trust x regime multiplier x session multiplier per factor, consistent with policy schema and decision logging.
