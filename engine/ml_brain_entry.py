from typing import Any, Dict, List

from .decision_frame import DecisionFrame
from .entry_features import extract_entry_features
from .ml_brain import BrainScore, score_combo


def score_entry_models(
    frame: DecisionFrame,
    eligible_models: List[str],
    brain_artifacts: Any,
) -> Dict[str, BrainScore]:
    results: Dict[str, BrainScore] = {}
    for entry_id in eligible_models:
        try:
            features = extract_entry_features(entry_id, frame)
            score = score_combo(
                strategy_id=entry_id,
                entry_model_id=entry_id,
                condition_vector=features,
                artifacts=brain_artifacts,
            )
            results[entry_id] = score
        except Exception:
            continue
    return results
