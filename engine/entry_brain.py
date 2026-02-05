import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from .decision_frame import DecisionFrame
from .entry_eligibility import is_entry_eligible
from .entry_models import ENTRY_MODELS, EntryModelDefinition, EntryModelSpec
from .entry_selector_scoring import score_entry_selector


class BrainScore(TypedDict):
    entry_id: str
    raw_score: float
    adjusted_score: float
    policy_label: str


class BrainPolicy:
    DEFAULT_MULTIPLIERS: Dict[str, float] = {
        "PREFERRED": 1.5,
        "ALLOWED": 1.0,
        "DISABLED": 0.0,
        "DISCOURAGED": 0.8,
    }

    def __init__(
        self,
        policy: Optional[Dict[str, str]] = None,
        selector_artifacts: Optional[Any] = None,
        multipliers: Optional[Dict[str, float]] = None,
    ):
        self.policy = {k: v for k, v in (policy or {}).items()}
        self.selector_artifacts = selector_artifacts
        self.multipliers = {**self.DEFAULT_MULTIPLIERS, **(multipliers or {})}

    def lookup(self, entry_id: str, frame: DecisionFrame) -> str:
        return self.policy.get(entry_id, "DISABLED")

    def multiplier_for(self, label: str) -> float:
        return float(self.multipliers.get(label, 0.0))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        selector_artifacts: Optional[Any] = None,
        multipliers: Optional[Dict[str, float]] = None,
    ) -> "BrainPolicy":
        policy_map: Dict[str, str] = {}
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            records = payload.get("policy") if isinstance(payload, dict) else None
            if isinstance(records, list):
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    entry_id = rec.get("entry_model_id")
                    label = rec.get("label")
                    if entry_id and label:
                        policy_map[str(entry_id)] = str(label)
        except Exception:
            policy_map = {}
        return cls(
            policy=policy_map,
            selector_artifacts=selector_artifacts,
            multipliers=multipliers,
        )


def _resolve_model(
    model: EntryModelSpec | EntryModelDefinition,
) -> EntryModelDefinition:
    if isinstance(model, EntryModelDefinition):
        return model
    entry_id = model.get("id") if isinstance(model, dict) else None
    if entry_id and entry_id in ENTRY_MODELS:
        return ENTRY_MODELS[entry_id]
    raise KeyError(f"Unknown entry model: {entry_id}")


def score_entry_models(
    frame: DecisionFrame,
    candidate_models: List[EntryModelSpec | EntryModelDefinition],
    brain_policy: BrainPolicy,
) -> Dict[str, BrainScore]:
    eligible_models: List[EntryModelDefinition] = []
    for model in candidate_models:
        resolved = _resolve_model(model)
        try:
            if is_entry_eligible(resolved, frame):
                eligible_models.append(resolved)
        except Exception:
            continue

    if not eligible_models:
        return {}

    eligible_ids = [m.id for m in eligible_models]
    raw_scores = score_entry_selector(
        frame, eligible_ids, brain_policy.selector_artifacts
    )

    results: Dict[str, BrainScore] = {}
    for model in eligible_models:
        entry_id = model.id
        payload = raw_scores.get(entry_id, {}) if isinstance(raw_scores, dict) else {}
        raw_score = float(payload.get("expected_R", 0.0) or 0.0)
        policy_label = (
            brain_policy.lookup(entry_id, frame)
            if hasattr(brain_policy, "lookup")
            else "UNKNOWN"
        )
        multiplier = (
            brain_policy.multiplier_for(policy_label)
            if hasattr(brain_policy, "multiplier_for")
            else 1.0
        )
        if policy_label == "DISABLED":
            multiplier = 0.0
        adjusted = float(raw_score * float(multiplier))

        results[entry_id] = BrainScore(
            entry_id=entry_id,
            raw_score=raw_score,
            adjusted_score=adjusted,
            policy_label=policy_label,
        )

    return results
