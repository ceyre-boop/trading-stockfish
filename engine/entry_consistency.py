from typing import Any, Dict, List, Optional

import pandas as pd

from .decision_frame import DecisionFrame
from .entry_eligibility import get_eligible_entry_models, is_entry_eligible
from .entry_models import ENTRY_MODELS

_VALID_LABELS = {"PREFERRED", "ALLOWED", "DISCOURAGED", "DISABLED"}


def _lookup_policy_row(
    entry_id: str, frame: DecisionFrame, policy_df: pd.DataFrame
) -> Optional[pd.Series]:
    if policy_df is None or policy_df.empty:
        return None
    candidates = policy_df[policy_df.get("entry_model_id") == entry_id]
    if candidates.empty:
        return None

    mp_state = frame.market_profile_state or "UNKNOWN"
    session_profile = frame.session_profile or "UNKNOWN"
    liq_bias = None
    if isinstance(frame.liquidity_frame, dict):
        liq_bias = frame.liquidity_frame.get("bias")
    liq_bias = liq_bias or "UNKNOWN"

    exact = candidates[
        (candidates.get("market_profile_state") == mp_state)
        & (candidates.get("session_profile") == session_profile)
        & (candidates.get("liquidity_bias_side") == liq_bias)
    ]
    if not exact.empty:
        return exact.iloc[0]
    return candidates.iloc[0]


def _missing_required_signals(entry_id: str, frame: DecisionFrame) -> bool:
    entry = ENTRY_MODELS.get(entry_id)
    if entry is None:
        return False
    required = entry.required_signals or {}
    if not required:
        return False
    signals = frame.entry_signals_present or {}

    if required.get("needs_displacement"):
        displacement_score = None
        if isinstance(frame.market_profile_evidence, dict):
            displacement_score = frame.market_profile_evidence.get("displacement_score")
        if displacement_score is None or displacement_score <= 0.4:
            return True

    mapping = {
        "needs_sweep": "sweep",
        "needs_fvg": "fvg",
        "needs_ob": "ob",
        "needs_ifvg": "ifvg",
    }
    for req_key, sig_key in mapping.items():
        if required.get(req_key) and not signals.get(sig_key, False):
            return True
    return False


def validate_entry_consistency(
    frame: DecisionFrame, brain_policy_entries: pd.DataFrame
) -> Dict[str, List[str] | bool]:
    policy_override: List[str] = []
    eligibility_mismatch: List[str] = []
    signal_mismatch: List[str] = []

    eligible = set(frame.eligible_entry_models or get_eligible_entry_models(frame))

    for entry_id in ENTRY_MODELS.keys():
        policy_row = _lookup_policy_row(entry_id, frame, brain_policy_entries)
        if policy_row is None:
            continue
        label = policy_row.get("label")
        if label not in _VALID_LABELS:
            continue

        is_eligible = entry_id in eligible or is_entry_eligible(
            ENTRY_MODELS[entry_id], frame
        )

        if label == "DISABLED" and is_eligible:
            policy_override.append(entry_id)
        if label == "PREFERRED" and not is_eligible:
            eligibility_mismatch.append(entry_id)
        if label == "PREFERRED" and _missing_required_signals(entry_id, frame):
            signal_mismatch.append(entry_id)

    report = {
        "policy_override": sorted(set(policy_override)),
        "eligibility_mismatch": sorted(set(eligibility_mismatch)),
        "signal_mismatch": sorted(set(signal_mismatch)),
    }
    report["ok"] = not any(
        report[k]
        for k in ["policy_override", "eligibility_mismatch", "signal_mismatch"]
    )

    frame.entry_consistency_report = report
    return report
