import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd


@dataclass
class ArchetypePerformance:
    archetype_id: str

    mean_expected_R: float
    winrate: float
    sample_size: int

    by_market_profile_state: Dict[str, Dict[str, float]]
    by_session_profile: Dict[str, Dict[str, float]]
    by_vol_regime: Dict[str, Dict[str, float]]
    by_trend_regime: Dict[str, Dict[str, float]]

    avg_stability: float

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
            if key:
                mapping[str(key)] = dict(item)
        return mapping
    return {}


def _safe_mean(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    series = pd.to_numeric(series, errors="coerce")
    series = series.dropna()
    return float(series.mean()) if not series.empty else 0.0


def _group_perf(df: pd.DataFrame, col: str) -> Dict[str, Dict[str, float]]:
    if df is None or df.empty or col not in df.columns:
        return {}
    groups = {}
    for key, group in df.groupby(col):
        key_val = "UNKNOWN" if pd.isna(key) else str(key)
        groups[key_val] = {
            "expected_R": _safe_mean(group.get("expected_R")),
            "winrate": _safe_mean(group.get("entry_success")),
            "samples": int(len(group)),
        }
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def _commentary(
    mean_r: float, winrate: float, structural_note: str, stability: float
) -> str:
    perf_note = (
        "High performance"
        if mean_r > 1.0
        else "Moderate performance" if mean_r >= 0.0 else "Weak performance"
    )
    win_note = (
        "Strong winrate"
        if winrate >= 0.55
        else "Neutral winrate" if winrate >= 0.45 else "Low winrate"
    )
    stab_note = (
        "Stable"
        if stability >= 0.75
        else "Mixed stability" if stability >= 0.4 else "Unstable"
    )
    return f"{perf_note}; {win_note}; {structural_note}; {stab_note}"


def compute_archetype_performance(
    archetypes: Any,
    replay_df: pd.DataFrame,
    stability_report: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    archetype_map = _coerce_map(archetypes, key_field="archetype_id")
    stability_map = _coerce_map(stability_report or {}, key_field="entry_model_id")

    replay_df = replay_df.copy() if replay_df is not None else pd.DataFrame()
    if not replay_df.empty:
        for col in ["expected_R", "entry_success"]:
            if col in replay_df.columns:
                replay_df[col] = pd.to_numeric(replay_df[col], errors="coerce")

    report: Dict[str, Dict[str, Any]] = {}

    for archetype_id, payload in sorted(archetype_map.items(), key=lambda kv: kv[0]):
        members = payload.get("members") or []
        subset = (
            replay_df[replay_df.get("entry_model_id").isin(members)]
            if (not replay_df.empty and "entry_model_id" in replay_df.columns)
            else pd.DataFrame()
        )

        mean_expected = _safe_mean(subset.get("expected_R"))
        winrate = _safe_mean(subset.get("entry_success"))
        sample_size = int(len(subset))

        mpg = _group_perf(subset, "market_profile_state")
        spg = _group_perf(subset, "session_profile")
        vrg = _group_perf(subset, "vol_regime")
        trg = _group_perf(subset, "trend_regime")

        stabilities = []
        if stability_map:
            for m in members:
                st = stability_map.get(m, {}).get("stability")
                if st is not None:
                    try:
                        stabilities.append(float(st))
                    except Exception:
                        continue
        avg_stability = (
            float(sum(stabilities) / len(stabilities)) if stabilities else 0.0
        )

        structural_note = (
            "Structural shift risk"
            if payload.get("label", "").lower().startswith("unstable")
            else "Structure consistent"
        )
        commentary = _commentary(mean_expected, winrate, structural_note, avg_stability)

        perf = ArchetypePerformance(
            archetype_id=archetype_id,
            mean_expected_R=mean_expected,
            winrate=winrate,
            sample_size=sample_size,
            by_market_profile_state=mpg,
            by_session_profile=spg,
            by_vol_regime=vrg,
            by_trend_regime=trg,
            avg_stability=avg_stability,
            commentary=commentary,
        )
        report[archetype_id] = perf.to_dict()

    return report


def write_archetype_performance(
    perf: Dict[str, Dict[str, Any]],
    path: str | Path = Path("storage/reports/archetype_performance.json"),
) -> None:
    ordered = dict(sorted(perf.items(), key=lambda kv: kv[0]))
    payload = []
    for archetype_id, rec in ordered.items():
        record = dict(rec)
        record["archetype_id"] = archetype_id
        payload.append(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
