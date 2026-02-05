from typing import Any

from .decision_frame import DecisionFrame


def eligible_sweep_displacement_reversal(frame: DecisionFrame) -> bool:
    """Override: require sweep signal and sufficient displacement evidence."""
    if frame is None:
        return False
    signals = getattr(frame, "entry_signals_present", {}) or {}
    sweep = bool(signals.get("sweep")) if isinstance(signals, dict) else False
    evidence = getattr(frame, "market_profile_evidence", {}) or {}
    displacement_score: Any = None
    if isinstance(evidence, dict):
        displacement_score = evidence.get("displacement_score")
    try:
        disp_ok = displacement_score is None or float(displacement_score) >= 0.4
    except Exception:
        disp_ok = False
    return sweep and disp_ok


def eligible_mean_reversion_range_extreme(frame: DecisionFrame) -> bool:
    """Override: avoid sweep context and keep displacement mild."""
    if frame is None:
        return False
    signals = getattr(frame, "entry_signals_present", {}) or {}
    sweep = bool(signals.get("sweep")) if isinstance(signals, dict) else False
    evidence = getattr(frame, "market_profile_evidence", {}) or {}
    displacement_score: Any = None
    if isinstance(evidence, dict):
        displacement_score = evidence.get("displacement_score")
    try:
        mild_disp = displacement_score is None or abs(float(displacement_score)) <= 0.3
    except Exception:
        mild_disp = True
    return (not sweep) and mild_disp
