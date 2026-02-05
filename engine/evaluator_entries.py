from typing import Dict, List

from .decision_frame import DecisionFrame
from .entry_brain import BrainPolicy, BrainScore, score_entry_models
from .entry_models import EntryModelDefinition, EntryModelSpec


def evaluate_entries(
    frame: DecisionFrame,
    candidate_models: List[EntryModelSpec | EntryModelDefinition],
    brain_policy: BrainPolicy,
) -> Dict[str, BrainScore]:
    return score_entry_models(frame, candidate_models, brain_policy)
