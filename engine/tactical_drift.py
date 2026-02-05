import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd


@dataclass
class TacticalDriftDiagnostic:
    entry_model_id: str

    prob_shift: float
    expected_R_shift: float
    winrate_shift: float
    confidence_shift: float

    structural_shift: bool
    old_best_state: str
    new_best_state: str

    regret_mean: float
    score_drift_mean: float
    eligibility_drift_rate: float

    stability: float
    commentary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_belief_shift(belief_shift: Any) -> Dict[str, Dict[str, Any]]:
    if belief_shift is None:
        return {}
    if isinstance(belief_shift, Mapping):
        return {k: dict(v) for k, v in belief_shift.items() if isinstance(v, Mapping)}
    if isinstance(belief_shift, Iterable):
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in belief_shift:
            if not isinstance(item, Mapping):
                continue
            entry_id = item.get("entry_model_id")
            if entry_id:
                mapping[entry_id] = dict(item)
        return mapping
    return {}


def _safe_mean(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    return float(series.dropna().mean()) if not series.dropna().empty else 0.0


def _stability(
    prob_shift: float,
    expected_shift: float,
    win_shift: float,
    score_drift_mean: float,
    elig_rate: float,
) -> float:
    val = 1.0 / (
        1.0
        + abs(prob_shift)
        + abs(expected_shift)
        + abs(win_shift)
        + score_drift_mean
        + elig_rate
    )
    return float(max(0.0, min(1.0, val)))


def _commentary(
    structural_shift: bool, delta_conf: float, delta_expected: float, stability: float
) -> str:
    parts = []
    if structural_shift:
        parts.append("Structural shift detected")
    else:
        parts.append("Structure stable")

    if delta_conf > 0:
        parts.append("Confidence increased")
    elif delta_conf < 0:
        parts.append("Confidence decreased")
    else:
        parts.append("Confidence flat")

    if delta_expected > 0:
        parts.append("Expected_R improved")
    elif delta_expected < 0:
        parts.append("Expected_R declined")
    else:
        parts.append("Expected_R flat")

    if stability >= 0.75:
        parts.append("Overall stable")
    elif stability >= 0.4:
        parts.append("Moderate drift")
    else:
        parts.append("High drift")

    return "; ".join(parts)


def compute_tactical_drift(
    belief_shift: Any,
    replay_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    shift_map = _coerce_belief_shift(belief_shift)
    replay_df = replay_df.copy() if replay_df is not None else pd.DataFrame()

    if not replay_df.empty:
        for col in ["regret", "score_drift"]:
            if col in replay_df.columns:
                replay_df[col] = pd.to_numeric(replay_df[col], errors="coerce")
        if "eligibility_drift" in replay_df.columns:
            replay_df["eligibility_drift"] = replay_df["eligibility_drift"].astype(bool)

    diagnostics: Dict[str, Dict[str, Any]] = {}
    entry_ids = sorted(shift_map.keys())

    for entry_id in entry_ids:
        shift = shift_map.get(entry_id, {})
        prob_shift = float(shift.get("delta_prob_select", 0.0) or 0.0)
        expected_shift = float(shift.get("delta_expected_R", 0.0) or 0.0)
        win_shift = float(shift.get("delta_winrate", 0.0) or 0.0)
        conf_shift = float(shift.get("delta_confidence", 0.0) or 0.0)

        structural_shift = bool(shift.get("structural_shift", False))
        old_best = str(shift.get("old_best_state", "UNKNOWN"))
        new_best = str(shift.get("new_best_state", "UNKNOWN"))

        subset = (
            replay_df[replay_df["entry_model_id"] == entry_id]
            if (not replay_df.empty and "entry_model_id" in replay_df.columns)
            else pd.DataFrame()
        )
        regret_mean = (
            _safe_mean(subset["regret"]) if "regret" in subset.columns else 0.0
        )
        score_drift_mean = (
            _safe_mean(subset["score_drift"])
            if "score_drift" in subset.columns
            else 0.0
        )
        elig_rate = 0.0
        if "eligibility_drift" in subset.columns and not subset.empty:
            elig_rate = float(subset["eligibility_drift"].mean())

        stability = _stability(
            prob_shift, expected_shift, win_shift, abs(score_drift_mean), elig_rate
        )
        commentary = _commentary(
            structural_shift, conf_shift, expected_shift, stability
        )

        diag = TacticalDriftDiagnostic(
            entry_model_id=entry_id,
            prob_shift=prob_shift,
            expected_R_shift=expected_shift,
            winrate_shift=win_shift,
            confidence_shift=conf_shift,
            structural_shift=structural_shift,
            old_best_state=old_best,
            new_best_state=new_best,
            regret_mean=regret_mean,
            score_drift_mean=abs(score_drift_mean),
            eligibility_drift_rate=elig_rate,
            stability=stability,
            commentary=commentary,
        )
        diagnostics[entry_id] = diag.to_dict()

    return dict(sorted(diagnostics.items(), key=lambda kv: kv[0]))


def write_tactical_drift(
    diags: Dict[str, Dict[str, Any]],
    path: str | Path = Path("storage/reports/tactical_drift.json"),
) -> None:
    ordered = dict(sorted(diags.items(), key=lambda kv: kv[0]))
    payload = []
    for entry_id, diag in ordered.items():
        record = dict(diag)
        record["entry_model_id"] = entry_id
        payload.append(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
