import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd


@dataclass
class CognitiveCycleSnapshot:
    cycle_index: int
    timestamp: str

    tips: Dict[str, Any]
    priors: Dict[str, Any]
    belief_shift: Dict[str, Any]
    drift: Dict[str, Any]
    stability_report: Dict[str, Any]
    archetypes: Dict[str, Any]
    archetype_performance: Dict[str, Any]
    archetype_evolution: Dict[str, Any]

    avg_stability: float
    dominant_archetype: str
    commentary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveTimeline:
    cycles: List[CognitiveCycleSnapshot]

    def to_dict(self) -> Dict[str, Any]:
        return {"cycles": [c.to_dict() for c in self.cycles]}


def _mean_stability(stability_report: Dict[str, Any]) -> float:
    if not stability_report:
        return 0.0
    values: List[float] = []
    for _, row in stability_report.items():
        if not isinstance(row, Mapping):
            continue
        val = row.get("stability")
        if val is None:
            continue
        try:
            values.append(float(val))
        except Exception:
            continue
    return float(sum(values) / len(values)) if values else 0.0


def _dominant_archetype(archetype_performance: Dict[str, Any]) -> str:
    if not archetype_performance:
        return "UNKNOWN"
    best_id = "UNKNOWN"
    best_val = float("-inf")
    for aid, row in archetype_performance.items():
        if not isinstance(row, Mapping):
            continue
        val = row.get("mean_expected_R")
        try:
            val_f = float(val)
        except Exception:
            continue
        if val_f > best_val or (val_f == best_val and str(aid) < best_id):
            best_val = val_f
            best_id = str(aid)
    return best_id


def _commentary(
    avg_stability: float, dominant_archetype: str, delta_signal: float
) -> str:
    stab_note = (
        "Stable"
        if avg_stability >= 0.75
        else "Mixed" if avg_stability >= 0.4 else "Unstable"
    )
    trend_note = (
        "strengthening"
        if delta_signal > 0
        else "weakening" if delta_signal < 0 else "steady"
    )
    return f"{stab_note} cognitive state; dominant {dominant_archetype}; trajectory {trend_note}."


def build_cycle_snapshot(
    cycle_index: int,
    timestamp: str,
    tips: Dict[str, Any],
    priors: Dict[str, Any],
    belief_shift: Dict[str, Any],
    drift: Dict[str, Any],
    stability_report: Dict[str, Any],
    archetypes: Dict[str, Any],
    archetype_performance: Dict[str, Any],
    archetype_evolution: Dict[str, Any],
) -> CognitiveCycleSnapshot:
    avg_stability = _mean_stability(stability_report)
    dominant = _dominant_archetype(archetype_performance)

    # Derive a simple delta signal from evolution by averaging delta_expected_R across archetypes
    delta_vals: List[float] = []
    for _, row in archetype_evolution.items():
        if not isinstance(row, Mapping):
            continue
        val = row.get("delta_expected_R")
        if val is None:
            continue
        try:
            delta_vals.append(float(val))
        except Exception:
            continue
    delta_signal = float(sum(delta_vals) / len(delta_vals)) if delta_vals else 0.0

    commentary = _commentary(avg_stability, dominant, delta_signal)

    return CognitiveCycleSnapshot(
        cycle_index=cycle_index,
        timestamp=timestamp,
        tips=tips or {},
        priors=priors or {},
        belief_shift=belief_shift or {},
        drift=drift or {},
        stability_report=stability_report or {},
        archetypes=archetypes or {},
        archetype_performance=archetype_performance or {},
        archetype_evolution=archetype_evolution or {},
        avg_stability=avg_stability,
        dominant_archetype=dominant,
        commentary=commentary,
    )


def _load_timeline(path: Path) -> CognitiveTimeline:
    if not path.exists():
        return CognitiveTimeline(cycles=[])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return CognitiveTimeline(cycles=[])
    cycles_raw = payload.get("cycles") if isinstance(payload, dict) else []
    cycles: List[CognitiveCycleSnapshot] = []
    for item in cycles_raw:
        if not isinstance(item, Mapping):
            continue
        try:
            cycles.append(
                CognitiveCycleSnapshot(
                    cycle_index=int(item.get("cycle_index", 0)),
                    timestamp=str(item.get("timestamp", "")),
                    tips=item.get("tips") or {},
                    priors=item.get("priors") or {},
                    belief_shift=item.get("belief_shift") or {},
                    drift=item.get("drift") or {},
                    stability_report=item.get("stability_report") or {},
                    archetypes=item.get("archetypes") or {},
                    archetype_performance=item.get("archetype_performance") or {},
                    archetype_evolution=item.get("archetype_evolution") or {},
                    avg_stability=float(item.get("avg_stability", 0.0) or 0.0),
                    dominant_archetype=str(item.get("dominant_archetype", "UNKNOWN")),
                    commentary=str(item.get("commentary", "")),
                )
            )
        except Exception:
            continue
    cycles = sorted(cycles, key=lambda c: c.cycle_index)
    return CognitiveTimeline(cycles=cycles)


def append_cycle_to_timeline(
    timeline_path: str, snapshot: CognitiveCycleSnapshot
) -> CognitiveTimeline:
    path = Path(timeline_path)
    timeline = _load_timeline(path)
    timeline.cycles.append(snapshot)
    timeline.cycles = sorted(timeline.cycles, key=lambda c: c.cycle_index)
    payload = timeline.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return timeline
