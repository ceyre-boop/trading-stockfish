import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd


@dataclass
class TacticalIntuitionProfile:
    entry_model_id: str

    avg_prob_select: float
    avg_expected_R: float
    winrate: float
    sample_size: int

    best_market_profile_state: str
    worst_market_profile_state: str
    best_session_profile: str
    worst_session_profile: str

    risk_expected_R: float
    risk_MAE_bucket: str
    risk_MFE_bucket: str
    risk_time_horizon: str
    risk_aggressiveness: str

    confidence_score: float
    commentary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_belief_map(belief_map: Any) -> Dict[str, Dict[str, Any]]:
    if belief_map is None:
        return {}
    if isinstance(belief_map, Mapping):
        return dict(belief_map)
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


def _group_best_worst(df: pd.DataFrame, column: str) -> (str, str):
    if column not in df.columns or df.empty:
        return "UNKNOWN", "UNKNOWN"
    grouped = (
        df.groupby(column)["expected_R"].mean().reset_index()
        if "expected_R" in df.columns
        else pd.DataFrame(columns=[column, "expected_R"])
    )
    if grouped.empty:
        return "UNKNOWN", "UNKNOWN"
    grouped = grouped.sort_values(["expected_R", column], na_position="last")
    best = grouped.iloc[-1][column]
    worst = grouped.iloc[0][column]
    best_val = "UNKNOWN" if pd.isna(best) else str(best)
    worst_val = "UNKNOWN" if pd.isna(worst) else str(worst)
    return best_val, worst_val


def _confidence(sample_size: int) -> float:
    if sample_size is None or sample_size <= 0:
        return 0.0
    return float(min(1.0, math.log(sample_size + 1) / 5.0))


def _commentary(
    entry_id: str,
    risk_aggressiveness: str,
    risk_time_horizon: str,
    best_mp: str,
    worst_mp: str,
) -> str:
    agg = risk_aggressiveness or "UNKNOWN_AGGRO"
    horizon = risk_time_horizon or "UNKNOWN_HORIZON"
    best = best_mp or "UNKNOWN"
    worst = worst_mp or "UNKNOWN"
    return f"{agg.title()} profile ({horizon}) with strength in {best} and vulnerability in {worst}."


def build_tactical_intuition_profiles(
    belief_map: Any,
    replay_df: pd.DataFrame,
    entry_models: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    belief_lookup = _coerce_belief_map(belief_map)
    replay_df = replay_df.copy() if replay_df is not None else pd.DataFrame()
    if not replay_df.empty and "expected_R" in replay_df.columns:
        try:
            replay_df["expected_R"] = pd.to_numeric(
                replay_df["expected_R"], errors="coerce"
            )
        except Exception:
            pass

    tips: Dict[str, Dict[str, Any]] = {}

    for entry_id, model_def in entry_models.items():
        stats = belief_lookup.get(entry_id, {})
        avg_prob_select = float(stats.get("avg_prob_select", 0.0) or 0.0)
        avg_expected_R = float(stats.get("avg_expected_R", 0.0) or 0.0)
        winrate = stats.get("winrate")
        if winrate is None:
            winrate = stats.get("empirical_success_rate", 0.0)
        winrate = float(winrate or 0.0)
        sample_size = int(stats.get("sample_size") or stats.get("count") or 0)

        if not replay_df.empty and "entry_model_id" in replay_df.columns:
            subset = replay_df[replay_df["entry_model_id"] == entry_id]
        else:
            subset = pd.DataFrame()
        best_mp, worst_mp = _group_best_worst(subset, "market_profile_state")
        best_session, worst_session = _group_best_worst(subset, "session_profile")

        risk_profile = getattr(model_def, "risk_profile", {}) or {}
        tip = TacticalIntuitionProfile(
            entry_model_id=entry_id,
            avg_prob_select=avg_prob_select,
            avg_expected_R=avg_expected_R,
            winrate=winrate,
            sample_size=sample_size,
            best_market_profile_state=best_mp,
            worst_market_profile_state=worst_mp,
            best_session_profile=best_session,
            worst_session_profile=worst_session,
            risk_expected_R=risk_profile.get("expected_R"),
            risk_MAE_bucket=risk_profile.get("mae_bucket"),
            risk_MFE_bucket=risk_profile.get("mfe_bucket"),
            risk_time_horizon=risk_profile.get("time_horizon"),
            risk_aggressiveness=risk_profile.get("aggressiveness"),
            confidence_score=_confidence(sample_size),
            commentary=_commentary(
                entry_id,
                risk_profile.get("aggressiveness"),
                risk_profile.get("time_horizon"),
                best_mp,
                worst_mp,
            ),
        )
        tips[entry_id] = tip.to_dict()

    return dict(sorted(tips.items(), key=lambda kv: kv[0]))


def write_entry_tips(
    tips: Dict[str, Dict[str, Any]],
    path: str | Path = Path("storage/reports/entry_tips.json"),
) -> None:
    ordered = dict(sorted(tips.items(), key=lambda kv: kv[0]))
    payload: List[Dict[str, Any]] = []
    for entry_id, tip in ordered.items():
        record = dict(tip)
        record["entry_model_id"] = entry_id
        payload.append(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
