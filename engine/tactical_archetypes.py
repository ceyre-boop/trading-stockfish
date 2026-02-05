import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class TacticalArchetype:
    archetype_id: str
    members: list[str]
    centroid: dict
    label: str
    commentary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_FEATURE_COLUMNS = [
    "avg_prob_select",
    "avg_expected_R",
    "winrate",
    "sample_size",
    "risk_expected_R",
    "risk_aggressiveness",
    "base_prob_prior",
    "base_expected_R_prior",
    "confidence",
    "delta_prob_select",
    "delta_expected_R",
    "delta_winrate",
    "delta_confidence",
    "structural_shift",
    "regret_mean",
    "score_drift_mean",
    "eligibility_drift_rate",
    "stability",
]

_AGGRO_MAP = {"CONSERVATIVE": 0.0, "NEUTRAL": 1.0, "AGGRESSIVE": 2.0}


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


def _aggro_code(value: Any) -> float:
    if value is None:
        return 1.0
    key = str(value).upper()
    return _AGGRO_MAP.get(key, 1.0)


def build_archetype_features(
    tips: Any,
    priors: Any,
    belief_shift: Any,
    drift: Any,
) -> pd.DataFrame:
    tip_map = _coerce_map(tips)
    prior_map = _coerce_map(priors)
    shift_map = _coerce_map(belief_shift)
    drift_map = _coerce_map(drift)

    entry_ids = sorted(
        set(tip_map.keys())
        | set(prior_map.keys())
        | set(shift_map.keys())
        | set(drift_map.keys())
    )
    rows = []

    for entry_id in entry_ids:
        tip = tip_map.get(entry_id, {})
        prior = prior_map.get(entry_id, {})
        shift = shift_map.get(entry_id, {})
        drift_row = drift_map.get(entry_id, {})

        row = {
            "entry_model_id": entry_id,
            "avg_prob_select": float(tip.get("avg_prob_select", 0.0) or 0.0),
            "avg_expected_R": float(tip.get("avg_expected_R", 0.0) or 0.0),
            "winrate": float(
                tip.get("winrate", tip.get("empirical_success_rate", 0.0)) or 0.0
            ),
            "sample_size": float(tip.get("sample_size") or tip.get("count") or 0.0),
            "risk_expected_R": float(
                tip.get("risk_expected_R", prior.get("risk_expected_R", 0.0)) or 0.0
            ),
            "risk_aggressiveness": _aggro_code(
                tip.get("risk_aggressiveness", prior.get("risk_aggressiveness"))
            ),
            "base_prob_prior": float(
                prior.get("base_prob_prior", prior.get("avg_prob_select", 0.0)) or 0.0
            ),
            "base_expected_R_prior": float(
                prior.get("base_expected_R_prior", prior.get("risk_expected_R", 0.0))
                or 0.0
            ),
            "confidence": float(
                prior.get("confidence", tip.get("confidence_score", 0.0)) or 0.0
            ),
            "delta_prob_select": float(shift.get("delta_prob_select", 0.0) or 0.0),
            "delta_expected_R": float(shift.get("delta_expected_R", 0.0) or 0.0),
            "delta_winrate": float(shift.get("delta_winrate", 0.0) or 0.0),
            "delta_confidence": float(shift.get("delta_confidence", 0.0) or 0.0),
            "structural_shift": float(1.0 if shift.get("structural_shift") else 0.0),
            "regret_mean": float(drift_row.get("regret_mean", 0.0) or 0.0),
            "score_drift_mean": float(drift_row.get("score_drift_mean", 0.0) or 0.0),
            "eligibility_drift_rate": float(
                drift_row.get("eligibility_drift_rate", 0.0) or 0.0
            ),
            "stability": float(drift_row.get("stability", 0.0) or 0.0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("entry_model_id").reset_index(drop=True)


def cluster_tactics(feature_df: pd.DataFrame, k: int = 3) -> Dict[str, str]:
    if feature_df is None or feature_df.empty:
        return {}
    n = len(feature_df)
    k = max(1, min(k, n))
    features = feature_df[_FEATURE_COLUMNS]
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    cluster_map = {}
    for entry_id, label in zip(feature_df["entry_model_id"], labels):
        cluster_map[entry_id] = f"ARCHETYPE_{int(label)}"
    return cluster_map


def _label_from_centroid(centroid: Dict[str, float]) -> str:
    aggro_val = centroid.get("risk_aggressiveness", 1.0)
    stability = centroid.get("stability", 0.0)
    exp_r = centroid.get("avg_expected_R", 0.0)

    if aggro_val >= 1.5:
        aggro_label = "Aggressive"
    elif aggro_val >= 0.8:
        aggro_label = "Balanced"
    else:
        aggro_label = "Conservative"

    if stability >= 0.75:
        stab_label = "Stable"
    elif stability >= 0.5:
        stab_label = "Moderate"
    else:
        stab_label = "Unstable"

    if exp_r >= 1.5:
        reward_label = "High-Reward"
    elif exp_r >= 0.5:
        reward_label = "Mid-Reward"
    else:
        reward_label = "Low-Reward"

    return f"{aggro_label} {stab_label} {reward_label}"


def _commentary_from_centroid(centroid: Dict[str, float]) -> str:
    return (
        f"Stability={centroid.get('stability', 0):.2f}; "
        f"Aggro={centroid.get('risk_aggressiveness', 0):.2f}; "
        f"ExpR={centroid.get('avg_expected_R', 0):.2f}; "
        f"Drift={centroid.get('score_drift_mean', 0):.2f}"
    )


def build_tactical_archetypes(
    feature_df: pd.DataFrame, cluster_map: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    if feature_df is None or feature_df.empty:
        return {}
    if not cluster_map:
        return {}

    # Attach cluster labels
    feature_df = feature_df.copy()
    feature_df["archetype_id"] = feature_df["entry_model_id"].map(cluster_map)

    archetypes: Dict[str, Dict[str, Any]] = {}

    for archetype_id, group in feature_df.groupby("archetype_id"):
        centroid_series = group[_FEATURE_COLUMNS].mean()
        centroid = {k: float(v) for k, v in centroid_series.to_dict().items()}
        members = sorted(group["entry_model_id"].tolist())
        label = _label_from_centroid(centroid)
        commentary = _commentary_from_centroid(centroid)
        archetype = TacticalArchetype(
            archetype_id=archetype_id,
            members=members,
            centroid=centroid,
            label=label,
            commentary=commentary,
        )
        archetypes[archetype_id] = archetype.to_dict()

    return dict(sorted(archetypes.items(), key=lambda kv: kv[0]))


def write_tactical_archetypes(
    archetypes: Dict[str, Dict[str, Any]],
    path: str | Path = Path("storage/reports/tactical_archetypes.json"),
) -> None:
    ordered = dict(sorted(archetypes.items(), key=lambda kv: kv[0]))
    payload = []
    for archetype_id, rec in ordered.items():
        record = dict(rec)
        record["archetype_id"] = archetype_id
        payload.append(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
