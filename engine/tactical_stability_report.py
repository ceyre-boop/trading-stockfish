import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


@dataclass
class TacticalStabilityRecord:
    entry_model_id: str

    avg_prob_select: float
    avg_expected_R: float
    winrate: float
    sample_size: int
    best_state: str
    worst_state: str
    risk_expected_R: float
    risk_aggressiveness: str

    delta_prob_select: float
    delta_expected_R: float
    delta_winrate: float
    structural_shift: bool

    regret_mean: float
    score_drift_mean: float
    eligibility_drift_rate: float
    stability: float

    prior_expected_R: float
    prior_alignment: float

    commentary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_map(obj: Any) -> Dict[str, Dict[str, Any]]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return {k: dict(v) for k, v in obj.items() if isinstance(v, Mapping)}
    if isinstance(obj, Iterable):
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in obj:
            if not isinstance(item, Mapping):
                continue
            entry_id = item.get("entry_model_id")
            if entry_id:
                mapping[entry_id] = dict(item)
        return mapping
    return {}


def _commentary(
    structural_shift: bool,
    delta_expected: float,
    stability: float,
    prior_alignment: float,
) -> str:
    parts = []
    if structural_shift:
        parts.append("Structural shift detected")
    else:
        parts.append("Structure stable")

    if delta_expected > 0:
        parts.append("Expected_R improving")
    elif delta_expected < 0:
        parts.append("Expected_R declining")
    else:
        parts.append("Expected_R flat")

    if prior_alignment is not None:
        if prior_alignment < 0.2:
            parts.append("Priors aligned")
        else:
            parts.append("Priors misaligned")

    if stability >= 0.75:
        parts.append("Stable tactic")
    elif stability >= 0.4:
        parts.append("Moderate stability")
    else:
        parts.append("Unstable tactic")

    return "; ".join(parts)


def build_tactical_stability_report(
    tips: Any,
    priors: Any,
    belief_shift: Any,
    drift: Any,
) -> Dict[str, Dict[str, Any]]:
    tip_map = _coerce_map(tips)
    prior_map = _coerce_map(priors)
    shift_map = _coerce_map(belief_shift)
    drift_map = _coerce_map(drift)

    all_ids = sorted(
        set(tip_map.keys())
        | set(prior_map.keys())
        | set(shift_map.keys())
        | set(drift_map.keys())
    )
    report: Dict[str, Dict[str, Any]] = {}

    for entry_id in all_ids:
        tip = tip_map.get(entry_id, {})
        prior = prior_map.get(entry_id, {})
        shift = shift_map.get(entry_id, {})
        drift_row = drift_map.get(entry_id, {})

        avg_prob = float(tip.get("avg_prob_select", 0.0) or 0.0)
        avg_expected = float(tip.get("avg_expected_R", 0.0) or 0.0)
        winrate = float(
            tip.get("winrate", tip.get("empirical_success_rate", 0.0)) or 0.0
        )
        sample_size = int(tip.get("sample_size") or tip.get("count") or 0)
        best_state = str(
            tip.get("best_market_profile_state", tip.get("best_state", "UNKNOWN"))
        )
        worst_state = str(
            tip.get("worst_market_profile_state", tip.get("worst_state", "UNKNOWN"))
        )
        risk_expected = (
            tip.get("risk_expected_R")
            if tip.get("risk_expected_R") is not None
            else prior.get("risk_expected_R")
        )
        risk_aggr = (
            tip.get("risk_aggressiveness")
            if tip.get("risk_aggressiveness") is not None
            else prior.get("risk_aggressiveness")
        )

        delta_prob = float(shift.get("delta_prob_select", 0.0) or 0.0)
        delta_expected = float(shift.get("delta_expected_R", 0.0) or 0.0)
        delta_win = float(shift.get("delta_winrate", 0.0) or 0.0)
        structural_shift = bool(shift.get("structural_shift", False))

        regret_mean = float(drift_row.get("regret_mean", 0.0) or 0.0)
        score_drift_mean = float(drift_row.get("score_drift_mean", 0.0) or 0.0)
        elig_rate = float(drift_row.get("eligibility_drift_rate", 0.0) or 0.0)
        stability = float(drift_row.get("stability", 0.0) or 0.0)

        prior_expected = prior.get("risk_expected_R")
        if prior_expected is None:
            prior_expected = prior.get("base_expected_R_prior")
        prior_expected_val = (
            float(prior_expected) if prior_expected is not None else 0.0
        )
        prior_alignment = (
            abs(avg_expected - prior_expected_val)
            if prior_expected is not None
            else 0.0
        )

        commentary = _commentary(
            structural_shift, delta_expected, stability, prior_alignment
        )

        record = TacticalStabilityRecord(
            entry_model_id=entry_id,
            avg_prob_select=avg_prob,
            avg_expected_R=avg_expected,
            winrate=winrate,
            sample_size=sample_size,
            best_state=best_state,
            worst_state=worst_state,
            risk_expected_R=risk_expected,
            risk_aggressiveness=risk_aggr,
            delta_prob_select=delta_prob,
            delta_expected_R=delta_expected,
            delta_winrate=delta_win,
            structural_shift=structural_shift,
            regret_mean=regret_mean,
            score_drift_mean=score_drift_mean,
            eligibility_drift_rate=elig_rate,
            stability=stability,
            prior_expected_R=prior_expected_val,
            prior_alignment=prior_alignment,
            commentary=commentary,
        )
        report[entry_id] = record.to_dict()

    return dict(sorted(report.items(), key=lambda kv: kv[0]))


def write_tactical_stability_report(
    report: Dict[str, Dict[str, Any]],
    path: str | Path = Path("storage/reports/tactical_stability_report.json"),
) -> None:
    ordered = dict(sorted(report.items(), key=lambda kv: kv[0]))
    payload = []
    for entry_id, rec in ordered.items():
        record = dict(rec)
        record["entry_model_id"] = entry_id
        payload.append(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
