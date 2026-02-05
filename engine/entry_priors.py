import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class EntryModelPrior:
    entry_model_id: str

    base_prob_prior: float
    base_expected_R_prior: float
    confidence: float

    preferred_market_profile_states: List[str]
    discouraged_market_profile_states: List[str]

    preferred_session_profiles: List[str]
    discouraged_session_profiles: List[str]

    risk_expected_R: float
    risk_aggressiveness: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_tips(tips: Any) -> Dict[str, Dict[str, Any]]:
    if tips is None:
        return {}
    if isinstance(tips, Mapping):
        return {k: dict(v) for k, v in tips.items() if isinstance(v, Mapping)}
    if isinstance(tips, Iterable):
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in tips:
            if not isinstance(item, Mapping):
                continue
            entry_id = item.get("entry_model_id")
            if entry_id:
                mapping[entry_id] = dict(item)
        return mapping
    return {}


def _pref_list(primary: Optional[str]) -> List[str]:
    if primary is None:
        return []
    val = str(primary)
    if val.upper() == "UNKNOWN":
        return []
    return [val]


def build_entry_priors(tips: Any) -> Dict[str, Dict[str, Any]]:
    tip_map = _coerce_tips(tips)
    priors: Dict[str, Dict[str, Any]] = {}

    for entry_id, tip in tip_map.items():
        base_prob = float(tip.get("avg_prob_select", 0.0) or 0.0)
        base_expected = float(tip.get("avg_expected_R", 0.0) or 0.0)
        confidence = float(tip.get("confidence_score", 0.0) or 0.0)

        best_mp = tip.get("best_market_profile_state")
        worst_mp = tip.get("worst_market_profile_state")
        best_session = tip.get("best_session_profile")
        worst_session = tip.get("worst_session_profile")

        prior = EntryModelPrior(
            entry_model_id=entry_id,
            base_prob_prior=base_prob,
            base_expected_R_prior=base_expected,
            confidence=confidence,
            preferred_market_profile_states=_pref_list(best_mp),
            discouraged_market_profile_states=_pref_list(worst_mp),
            preferred_session_profiles=_pref_list(best_session),
            discouraged_session_profiles=_pref_list(worst_session),
            risk_expected_R=tip.get("risk_expected_R"),
            risk_aggressiveness=tip.get("risk_aggressiveness"),
        )
        priors[entry_id] = prior.to_dict()

    return dict(sorted(priors.items(), key=lambda kv: kv[0]))


def write_entry_priors(
    priors: Dict[str, Dict[str, Any]],
    path: str | Path = Path("storage/reports/entry_priors.json"),
) -> None:
    ordered = dict(sorted(priors.items(), key=lambda kv: kv[0]))
    payload = []
    for entry_id, prior in ordered.items():
        record = dict(prior)
        record["entry_model_id"] = entry_id
        payload.append(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
