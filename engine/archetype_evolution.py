import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd


@dataclass
class ArchetypeEvolutionSignal:
    archetype_id: str

    delta_expected_R: float
    delta_winrate: float
    delta_sample_size: int

    structural_shift: bool
    old_best_state: str
    new_best_state: str

    old_stability: float
    new_stability: float
    delta_stability: float

    avg_member_drift: float

    commentary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_map(obj: Any, key_field: str = "archetype_id") -> Dict[str, Dict[str, Any]]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return {k: dict(v) for k, v in obj.items() if isinstance(v, Mapping)}
    if isinstance(obj, Iterable):
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in obj:
            if not isinstance(item, Mapping):
                continue
            key = item.get(key_field)
            if key is None:
                continue
            mapping[str(key)] = dict(item)
        return mapping
    return {}


def _best_state(perf_row: Dict[str, Any]) -> str:
    mpg = perf_row.get("by_market_profile_state") or {}
    if not isinstance(mpg, Mapping) or not mpg:
        return "UNKNOWN"
    best_key = None
    best_val = float("-inf")
    for state, metrics in mpg.items():
        if not isinstance(metrics, Mapping):
            continue
        val = metrics.get("expected_R")
        try:
            val_f = float(val)
        except Exception:
            continue
        if val_f > best_val or (val_f == best_val and (state or "") < (best_key or "")):
            best_val = val_f
            best_key = state
    return "UNKNOWN" if best_key is None else str(best_key)


def _mean(values: Sequence[float]) -> float:
    arr = [v for v in values if v is not None]
    arr = [float(v) for v in arr]
    return float(sum(arr) / len(arr)) if arr else 0.0


def compute_archetype_evolution(
    old_perf: Any,
    new_perf: Any,
    stability_report: Optional[Dict[str, Dict[str, Any]]] = None,
    belief_shift: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    old_map = _coerce_map(old_perf, key_field="archetype_id")
    new_map = _coerce_map(new_perf, key_field="archetype_id")
    stability_map = _coerce_map(stability_report or {}, key_field="entry_model_id")
    shift_map = _coerce_map(belief_shift or {}, key_field="entry_model_id")

    archetype_ids = sorted(set(old_map.keys()) | set(new_map.keys()))
    signals: Dict[str, Dict[str, Any]] = {}

    for aid in archetype_ids:
        old_row = old_map.get(aid, {})
        new_row = new_map.get(aid, {})

        delta_expected = float(new_row.get("mean_expected_R", 0.0) or 0.0) - float(
            old_row.get("mean_expected_R", 0.0) or 0.0
        )
        delta_win = float(new_row.get("winrate", 0.0) or 0.0) - float(
            old_row.get("winrate", 0.0) or 0.0
        )
        delta_samples = int(new_row.get("sample_size", 0) or 0) - int(
            old_row.get("sample_size", 0) or 0
        )

        old_best = _best_state(old_row)
        new_best = _best_state(new_row)
        structural_shift = old_best != new_best

        old_stab = float(old_row.get("avg_stability", 0.0) or 0.0)
        new_stab = float(new_row.get("avg_stability", 0.0) or 0.0)
        delta_stab = new_stab - old_stab

        members = new_row.get("members") or old_row.get("members") or []
        drifts: list[float] = []
        if members:
            for m in members:
                if stability_map and m in stability_map:
                    st = stability_map[m].get("stability")
                    if st is not None:
                        try:
                            drifts.append(float(st))
                            continue
                        except Exception:
                            pass
                if shift_map and m in shift_map:
                    de = shift_map[m].get("delta_expected_R")
                    if de is not None:
                        try:
                            drifts.append(abs(float(de)))
                            continue
                        except Exception:
                            pass
        avg_member_drift = _mean(drifts)

        commentary_parts = []
        if delta_expected > 0:
            commentary_parts.append("Expected_R improving")
        elif delta_expected < 0:
            commentary_parts.append("Expected_R declining")
        else:
            commentary_parts.append("Expected_R flat")

        if delta_stab > 0:
            commentary_parts.append("Stability rising")
        elif delta_stab < 0:
            commentary_parts.append("Stability falling")
        else:
            commentary_parts.append("Stability flat")

        if structural_shift:
            commentary_parts.append(f"Structural shift toward {new_best}")
        else:
            commentary_parts.append("Structure steady")

        commentary = "; ".join(commentary_parts)

        signal = ArchetypeEvolutionSignal(
            archetype_id=aid,
            delta_expected_R=delta_expected,
            delta_winrate=delta_win,
            delta_sample_size=delta_samples,
            structural_shift=structural_shift,
            old_best_state=old_best,
            new_best_state=new_best,
            old_stability=old_stab,
            new_stability=new_stab,
            delta_stability=delta_stab,
            avg_member_drift=avg_member_drift,
            commentary=commentary,
        )
        signals[aid] = signal.to_dict()

    return signals


def write_archetype_evolution(
    signals: Dict[str, Dict[str, Any]],
    path: str | Path = Path("storage/reports/archetype_evolution.json"),
) -> None:
    ordered = dict(sorted(signals.items(), key=lambda kv: kv[0]))
    payload = []
    for aid, rec in ordered.items():
        record = dict(rec)
        record["archetype_id"] = aid
        payload.append(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
