"""Policy evaluation gate.

Deterministic threshold checks for candidate policies. Given backtest metrics
and static thresholds, the gate returns a PASS/FAIL decision plus reasons. No
randomness or heuristic overrides are permitted: identical inputs always yield
identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BacktestResult:
    """Container for backtest metrics used by the gating step."""

    pnl: float
    sharpe: float
    hit_rate: float
    max_drawdown: float
    per_regime: Dict[str, float] = field(default_factory=dict)
    macro_regime: Dict[str, float] = field(default_factory=dict)
    session_regime: Dict[str, float] = field(default_factory=dict)
    stability_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GatingConfig:
    """Thresholds for the gate."""

    min_sharpe: float = 0.0
    min_hit_rate: float = 0.0
    max_drawdown: float = 1.0
    min_per_regime: float = -1.0
    max_return_variance: float = 1.0e9
    min_consistency: float = 0.0


@dataclass
class GatingDecision:
    decision: str
    reasons: List[str]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


def evaluate_candidate_policy(
    candidate: Any,
    backtest_result: BacktestResult,
    config: Optional[GatingConfig] = None,
) -> GatingDecision:
    cfg = config or GatingConfig()
    reasons: List[str] = []

    metrics = {
        "pnl": backtest_result.pnl,
        "sharpe": backtest_result.sharpe,
        "hit_rate": backtest_result.hit_rate,
        "max_drawdown": backtest_result.max_drawdown,
        "per_regime": dict(backtest_result.per_regime or {}),
        "macro_regime": dict(backtest_result.macro_regime or {}),
        "session_regime": dict(backtest_result.session_regime or {}),
        "stability_metrics": dict(backtest_result.stability_metrics or {}),
    }

    # Core metrics
    if backtest_result.sharpe < cfg.min_sharpe:
        reasons.append(
            f"sharpe_below_threshold: {backtest_result.sharpe:.6f} < {cfg.min_sharpe:.6f}"
        )
    if backtest_result.hit_rate < cfg.min_hit_rate:
        reasons.append(
            f"hit_rate_below_threshold: {backtest_result.hit_rate:.6f} < {cfg.min_hit_rate:.6f}"
        )
    if backtest_result.max_drawdown > cfg.max_drawdown:
        reasons.append(
            f"max_drawdown_exceeded: {backtest_result.max_drawdown:.6f} > {cfg.max_drawdown:.6f}"
        )

    # Per-regime floors (generic + macro + session)
    def _check_regimes(regime_map: Dict[str, float], label: str) -> None:
        for regime in sorted(regime_map.keys()):
            try:
                perf_val = float(regime_map[regime])
            except Exception:
                continue
            if perf_val < cfg.min_per_regime:
                reasons.append(
                    f"regime_floor_breach[{label}:{regime}]: {perf_val:.6f} < {cfg.min_per_regime:.6f}"
                )

    _check_regimes(backtest_result.per_regime or {}, "regime")
    _check_regimes(backtest_result.macro_regime or {}, "macro")
    _check_regimes(backtest_result.session_regime or {}, "session")

    # Stability checks
    stability = backtest_result.stability_metrics or {}
    variance_val = None
    for key in ("variance", "return_variance"):
        if key in stability:
            try:
                variance_val = float(stability[key])
                break
            except Exception:
                variance_val = None
                break
    if variance_val is not None and variance_val > cfg.max_return_variance:
        reasons.append(
            f"stability_variance_exceeded: {variance_val:.6f} > {cfg.max_return_variance:.6f}"
        )

    consistency_val = None
    for key in ("consistency", "consistency_score"):
        if key in stability:
            try:
                consistency_val = float(stability[key])
                break
            except Exception:
                consistency_val = None
                break
    if consistency_val is not None and consistency_val < cfg.min_consistency:
        reasons.append(
            f"stability_consistency_below_min: {consistency_val:.6f} < {cfg.min_consistency:.6f}"
        )

    decision = "PASS" if not reasons else "FAIL"
    return GatingDecision(decision=decision, reasons=reasons, metrics=metrics)
