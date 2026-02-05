import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class BeliefShift:
    entry_model_id: str

    delta_prob_select: float
    delta_expected_R: float
    delta_winrate: float

    old_best_state: str
    new_best_state: str
    structural_shift: bool

    old_confidence: float
    new_confidence: float
    delta_confidence: float

    prior_expected_R: Optional[float]
    prior_alignment: Optional[float]

    commentary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_belief_map(belief_map: Any) -> Dict[str, Dict[str, Any]]:
    if belief_map is None:
        return {}
    if isinstance(belief_map, Mapping):
        # Already keyed by entry id
        return {k: dict(v) for k, v in belief_map.items() if isinstance(v, Mapping)}
    if isinstance(belief_map, Iterable):
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in belief_map:
            if not isinstance(item, Mapping):
                continue
            entry_id = item.get("entry_model_id")
            if entry_id:
                mapping[entry_id] = dict(item)
        return mapping
    return {}


def _confidence_from_entry(entry: Dict[str, Any]) -> float:
    count = entry.get("count") or entry.get("sample_size") or 0
    try:
        count_val = max(0, int(count))
    except Exception:
        count_val = 0
    return float(min(1.0, math.log(count_val + 1) / 5.0))


def _best_state(entry: Dict[str, Any]) -> str:
    for key in ["best_market_profile_state", "best_state"]:
        if key in entry and entry.get(key) is not None:
            return str(entry.get(key))
    return "UNKNOWN"


def _winrate(entry: Dict[str, Any]) -> float:
    for key in ["winrate", "empirical_success_rate"]:
        if key in entry and entry.get(key) is not None:
            try:
                return float(entry.get(key))
            except Exception:
                continue
    return 0.0


def _expected_r(entry: Dict[str, Any]) -> float:
    try:
        return float(entry.get("avg_expected_R", 0.0) or 0.0)
    except Exception:
        return 0.0


def compute_belief_shift(
    old_belief_map: Any,
    new_belief_map: Any,
    entry_priors: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    old_map = _coerce_belief_map(old_belief_map)
    new_map = _coerce_belief_map(new_belief_map)
    entry_priors = entry_priors or {}

    all_ids = sorted(set(old_map.keys()) | set(new_map.keys()))
    shifts: Dict[str, Dict[str, Any]] = {}

    for entry_id in all_ids:
        old_entry = old_map.get(entry_id, {})
        new_entry = new_map.get(entry_id, {})

        old_prob = float(old_entry.get("avg_prob_select", 0.0) or 0.0)
        new_prob = float(new_entry.get("avg_prob_select", 0.0) or 0.0)

        old_expected = _expected_r(old_entry)
        new_expected = _expected_r(new_entry)

        old_win = _winrate(old_entry)
        new_win = _winrate(new_entry)

        old_conf = _confidence_from_entry(old_entry)
        new_conf = _confidence_from_entry(new_entry)

        old_best = _best_state(old_entry)
        new_best = _best_state(new_entry)

        prior_expected = None
        prior_align = None
        prior_entry = (
            entry_priors.get(entry_id) if isinstance(entry_priors, dict) else None
        )
        if isinstance(prior_entry, Mapping):
            val = prior_entry.get("risk_expected_R") or prior_entry.get(
                "base_expected_R_prior"
            )
            if val is not None:
                try:
                    prior_expected = float(val)
                    prior_align = abs(new_expected - prior_expected)
                except Exception:
                    prior_expected = None
                    prior_align = None

        structural_shift = old_best != new_best

        delta_prob = round(new_prob - old_prob, 6)
        delta_expected = round(new_expected - old_expected, 6)
        delta_win = round(new_win - old_win, 6)
        delta_conf = round(new_conf - old_conf, 6)

        commentary_parts = []
        if delta_expected > 0:
            commentary_parts.append("Higher expected_R")
        elif delta_expected < 0:
            commentary_parts.append("Lower expected_R")
        else:
            commentary_parts.append("Flat expected_R")

        if structural_shift:
            commentary_parts.append(f"Structural tilt toward {new_best}")
        else:
            commentary_parts.append("Structure stable")

        if delta_conf > 0:
            commentary_parts.append("Confidence up")
        elif delta_conf < 0:
            commentary_parts.append("Confidence down")
        else:
            commentary_parts.append("Confidence flat")

        commentary = "; ".join(commentary_parts)

        shift = BeliefShift(
            entry_model_id=entry_id,
            delta_prob_select=delta_prob,
            delta_expected_R=delta_expected,
            delta_winrate=delta_win,
            old_best_state=old_best,
            new_best_state=new_best,
            structural_shift=structural_shift,
            old_confidence=old_conf,
            new_confidence=new_conf,
            delta_confidence=delta_conf,
            prior_expected_R=prior_expected,
            prior_alignment=prior_align,
            commentary=commentary,
        )
        shifts[entry_id] = shift.to_dict()

    return dict(sorted(shifts.items(), key=lambda kv: kv[0]))


def write_belief_shift(
    shifts: Dict[str, Dict[str, Any]],
    path: str | Path = Path("storage/reports/belief_shift.json"),
) -> None:
    ordered = dict(sorted(shifts.items(), key=lambda kv: kv[0]))
    payload = []
    for entry_id, shift in ordered.items():
        record = dict(shift)
        record["entry_model_id"] = entry_id
        payload.append(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
