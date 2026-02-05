from typing import Callable, Dict, List, Optional

from .decision_frame import DecisionFrame
from .entry_models import ENTRY_MODELS, EntryModelDefinition


def _list_ok(value: Optional[str], allowed: List[str]) -> bool:
    if not allowed:
        return True
    if value is None:
        return False
    return str(value).upper() in {v.upper() for v in allowed}


def _liquidity_ok(liq: Optional[Dict], required: Dict[str, List[str]]) -> bool:
    if not required:
        return True
    liq = liq or {}
    bias = str(liq.get("bias", "")).upper()
    sweep_state = str(liq.get("sweep_state", "")).upper()
    distance_bucket = str(liq.get("distance_bucket", "")).upper()

    def _match(key: str, val: str) -> bool:
        allowed = required.get(key) or []
        if not allowed:
            return True
        allowed_upper = {a.upper() for a in allowed}
        if "ANY" in allowed_upper:
            return True
        return val.upper() in allowed_upper

    return (
        _match("bias_side", bias)
        and _match("sweep_state", sweep_state)
        and _match("distance_bucket", distance_bucket)
    )


def _regime_ok(cv: Optional[Dict[str, str]], key: str, allowed: List[str]) -> bool:
    if not allowed:
        return True
    if not cv:
        return False
    val = cv.get(key)
    if val is None:
        return False
    return str(val).upper() in {v.upper() for v in allowed}


def _signals_ok(
    signals: Dict[str, bool],
    required: Dict[str, bool],
    displacement_score: Optional[float],
) -> bool:
    signals = signals or {}
    required = required or {}
    disp_needed = bool(required.get("needs_displacement"))
    if disp_needed:
        if displacement_score is None or displacement_score <= 0.4:
            return False
    checks = {
        "needs_sweep": "sweep",
        "needs_fvg": "fvg",
        "needs_ob": "ob",
        "needs_ifvg": "ifvg",
    }
    for req_key, sig_key in checks.items():
        if required.get(req_key) and not signals.get(sig_key, False):
            return False
    return True


def is_entry_eligible(entry: EntryModelDefinition, frame: DecisionFrame) -> bool:
    if callable(entry.eligibility_fn):
        try:
            return bool(entry.eligibility_fn(frame))
        except Exception:
            return False

    mp_state = frame.market_profile_state
    session_profile = frame.session_profile
    liq = frame.liquidity_frame if isinstance(frame.liquidity_frame, dict) else {}
    signals = frame.entry_signals_present or {}
    displacement_score = None
    evidence = frame.market_profile_evidence or {}
    if isinstance(evidence, dict):
        displacement_score = evidence.get("displacement_score")

    cv = frame.condition_vector or {}

    base_ok = (
        _list_ok(mp_state, entry.required_market_profile_states)
        and _list_ok(session_profile, entry.required_session_profiles)
        and _liquidity_ok(liq, entry.required_liquidity_context)
        and _regime_ok(cv, "vol", entry.required_vol_regimes)
        and _regime_ok(cv, "trend", entry.required_trend_regimes)
        and _signals_ok(signals, entry.required_signals, displacement_score)
    )
    return bool(base_ok)


def get_eligible_entry_models(frame: DecisionFrame) -> List[str]:
    eligible: List[str] = []
    for model in ENTRY_MODELS.values():
        try:
            if is_entry_eligible(model, frame):
                eligible.append(model.id)
        except Exception:
            continue
    return sorted(eligible)
