"""Deterministic Bayesian Probability Engine v1.

Updates pattern probabilities using fixed priors, deterministic likelihoods, and
bounded posteriors. Designed to be replay/live consistent with no randomness or
external I/O.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping

EPS = 1e-6
MIN_PROB = 0.01
MAX_PROB = 0.99

PATTERNS = (
    "trend_continuation",
    "trend_reversal",
    "sweep_reversal",
    "sweep_continuation",
    "ob_respect",
    "ob_violation",
    "fvg_fill",
    "fvg_reject",
)


@dataclass(frozen=True)
class BayesianStateSnapshot:
    trend_continuation: float
    trend_reversal: float
    sweep_reversal: float
    sweep_continuation: float
    ob_respect: float
    ob_violation: float
    fvg_fill: float
    fvg_reject: float
    confidence_trend_continuation: float
    confidence_trend_reversal: float
    confidence_sweep_reversal: float
    confidence_sweep_continuation: float
    confidence_ob_respect: float
    confidence_ob_violation: float
    confidence_fvg_fill: float
    confidence_fvg_reject: float
    bayesian_update_strength: float


def _clamp_prob(value: float) -> float:
    return max(MIN_PROB, min(MAX_PROB, value))


def _bayes_update(prior: float, likelihood: float) -> float:
    numerator = likelihood * prior
    denominator = numerator + (1 - likelihood) * (1 - prior) + EPS
    return _clamp_prob(numerator / denominator)


def _volatility_factor(state: Mapping[str, Any]) -> float:
    # Prefer stable volatility for higher confidence
    vol_regime = (state.get("volatility_regime") or "NORMAL").upper()
    penalties = {
        "EXTREME": 0.6,
        "HIGH": 0.75,
        "NORMAL": 1.0,
        "LOW": 1.0,
    }
    return penalties.get(vol_regime, 0.9)


def _evidence_consistency(likelihood: float) -> float:
    if likelihood >= 0.75:
        return 0.25
    if likelihood >= 0.6:
        return 0.15
    if likelihood >= 0.5:
        return 0.1
    return 0.05


def _confidence(
    prior: float, posterior: float, likelihood: float, state: Mapping[str, Any]
) -> float:
    update_mag = abs(posterior - prior)
    vol_factor = _volatility_factor(state)
    return max(
        0.0, min(1.0, update_mag * 1.5 * vol_factor + _evidence_consistency(likelihood))
    )


def _priors(state: Mapping[str, Any]) -> Dict[str, float]:
    trend_strength = float(state.get("trend_strength", 0.0) or 0.0)
    cont = _clamp_prob(0.55 + min(max(trend_strength, 0.0), 1.0) * 0.25)
    reversal = _clamp_prob(1.05 - cont)
    priors = {
        "trend_continuation": cont,
        "trend_reversal": reversal,
        "sweep_reversal": _clamp_prob(float(state.get("p_sweep_reversal", 0.5) or 0.5)),
        "sweep_continuation": _clamp_prob(
            float(state.get("p_sweep_continuation", 0.5) or 0.5)
        ),
        "ob_respect": _clamp_prob(float(state.get("p_ob_hold", 0.5) or 0.5)),
        "ob_violation": _clamp_prob(float(state.get("p_ob_fail", 0.5) or 0.5)),
        "fvg_fill": _clamp_prob(float(state.get("p_fvg_fill", 0.5) or 0.5)),
    }
    priors["fvg_reject"] = _clamp_prob(1.0 - priors["fvg_fill"] + 0.05)
    return priors


def _likelihoods(state: Mapping[str, Any]) -> Dict[str, float]:
    trend_strength = float(state.get("trend_strength", 0.0) or 0.0)
    momentum_regime = str(state.get("momentum_regime", "CHOP") or "CHOP").upper()
    strong_trend = momentum_regime in {
        "TREND",
        "STRONG_TREND",
        "BULL_TREND",
        "BEAR_TREND",
    }
    choppy = momentum_regime == "CHOP"
    divergence_present = bool(
        state.get("rsi_bearish_divergence")
        or state.get("rsi_bullish_divergence")
        or state.get("macd_bearish_divergence")
        or state.get("macd_bullish_divergence")
    )
    has_sweep = str(state.get("last_sweep_direction", "NONE") or "NONE") != "NONE"
    absorption = bool(state.get("has_absorption"))
    exhaustion = bool(state.get("has_exhaustion"))
    imbalance = float(state.get("footprint_imbalance", 0.0) or 0.0)
    ob_touched = str(state.get("last_touched_ob_type", "NONE") or "NONE") != "NONE"
    mitigation = bool(state.get("has_mitigation"))
    flip_zone = bool(state.get("has_flip_zone"))
    has_fvg = bool(state.get("has_fvg"))
    vol_regime = (
        state.get("expected_volatility_state")
        or state.get("volatility_regime")
        or "LOW"
    ).upper()

    likelihoods = {
        "trend_continuation": 0.55
        + 0.25 * min(max(trend_strength, 0.0), 1.0)
        + (0.1 if strong_trend else 0.0)
        - (0.05 if choppy and trend_strength <= 0.3 else 0.0),
        "trend_reversal": 0.45
        + (0.18 if divergence_present else 0.0)
        + (0.08 if choppy and trend_strength <= 0.3 else 0.0)
        + (0.05 if not strong_trend else -0.05),
        "sweep_reversal": 0.4
        + (0.2 if has_sweep else 0.0)
        + (0.15 if absorption or exhaustion else 0.0),
        "sweep_continuation": 0.45
        + (0.2 if has_sweep else 0.0)
        + (0.1 if imbalance > 0 else 0.0),
        "ob_respect": 0.45
        + (0.2 if ob_touched else 0.0)
        + (0.1 if mitigation else 0.0)
        + (0.05 if not flip_zone else -0.05),
        "ob_violation": 0.45
        + (0.15 if ob_touched else 0.0)
        + (0.1 if flip_zone else 0.0),
        "fvg_fill": 0.4
        + (0.2 if has_fvg else 0.0)
        + (0.1 if vol_regime in {"HIGH", "EXTREME"} else 0.05),
        "fvg_reject": 0.4
        + (0.15 if not has_fvg else 0.0)
        + (0.1 if vol_regime in {"LOW"} else 0.05),
    }
    # Clamp likelihoods to [0.05, 0.95] to keep deterministic bounds
    return {k: max(0.05, min(0.95, v)) for k, v in likelihoods.items()}


def compute_bayesian_probabilities(state: Mapping[str, Any]) -> Dict[str, float]:
    """Compute posterior probabilities and confidences for pattern categories."""
    priors = _priors(state)
    likelihoods = _likelihoods(state)
    results: Dict[str, float] = {}
    updates = []
    for pattern in PATTERNS:
        prior = priors.get(pattern, 0.5)
        likelihood = likelihoods.get(pattern, 0.5)
        posterior = _bayes_update(prior, likelihood)
        confidence = _confidence(prior, posterior, likelihood, state)
        results[f"bayes_{pattern}"] = posterior
        results[f"bayes_{pattern}_confidence"] = confidence
        updates.append(abs(posterior - prior))
    results["bayesian_update_strength"] = (
        sum(updates) / len(updates) if updates else 0.0
    )
    return results


def snapshot_from_dict(state: Mapping[str, Any]) -> BayesianStateSnapshot:
    bayes = compute_bayesian_probabilities(state)
    return BayesianStateSnapshot(
        trend_continuation=bayes["bayes_trend_continuation"],
        trend_reversal=bayes["bayes_trend_reversal"],
        sweep_reversal=bayes["bayes_sweep_reversal"],
        sweep_continuation=bayes["bayes_sweep_continuation"],
        ob_respect=bayes["bayes_ob_respect"],
        ob_violation=bayes["bayes_ob_violation"],
        fvg_fill=bayes["bayes_fvg_fill"],
        fvg_reject=bayes["bayes_fvg_reject"],
        confidence_trend_continuation=bayes["bayes_trend_continuation_confidence"],
        confidence_trend_reversal=bayes["bayes_trend_reversal_confidence"],
        confidence_sweep_reversal=bayes["bayes_sweep_reversal_confidence"],
        confidence_sweep_continuation=bayes["bayes_sweep_continuation_confidence"],
        confidence_ob_respect=bayes["bayes_ob_respect_confidence"],
        confidence_ob_violation=bayes["bayes_ob_violation_confidence"],
        confidence_fvg_fill=bayes["bayes_fvg_fill_confidence"],
        confidence_fvg_reject=bayes["bayes_fvg_reject_confidence"],
        bayesian_update_strength=bayes["bayesian_update_strength"],
    )
