from typing import Any, Dict, List, Tuple

from .action_pairing import generate_candidate_actions
from .decision_actions import DecisionAction
from .decision_frame import DecisionFrame
from .endgame_tablebases import EndgameTablebasesV1
from .entry_brain import BrainPolicy
from .entry_models import EntryModelSpec
from .ev_brain_model import EVBrainV1
from .opening_book import OpeningBookV1
from .search_scoring import (
    apply_endgame_tablebases,
    apply_opening_book_scores,
    score_actions_with_cache,
)
from .transposition_table import TranspositionTable


class SearchEngineV1:
    def __init__(
        self,
        ev_brain: EVBrainV1,
        brain_policy: BrainPolicy,
        opening_book: OpeningBookV1,
        tablebases: EndgameTablebasesV1,
        *,
        n_paths: int = 64,
        horizon_bars: int = 60,
        seed: int = 42,
        cache_max_size: int = 50000,
    ) -> None:
        self.ev_brain = ev_brain
        self.brain_policy = brain_policy
        self.opening_book = opening_book
        self.tablebases = tablebases
        self.n_paths = n_paths
        self.horizon_bars = horizon_bars
        self.seed = seed
        self.tt = TranspositionTable(max_size=cache_max_size)

    def rank_actions(
        self,
        frame: DecisionFrame,
        position_state: Any,
        entry_models: List[EntryModelSpec],
        risk_envelope: Any,
        template_policy: Dict[str, Any] | None = None,
    ) -> List[Tuple[DecisionAction, Dict[str, Any]]]:
        candidate_actions = generate_candidate_actions(
            frame,
            position_state,
            entry_models,
            self.brain_policy,
            risk_envelope,
        )

        scored = score_actions_with_cache(
            frame,
            position_state,
            candidate_actions,
            ev_brain=self.ev_brain,
            brain_policy=self.brain_policy,
            risk_envelope=risk_envelope,
            n_paths=self.n_paths,
            horizon_bars=self.horizon_bars,
            seed=self.seed,
            table=self.tt,
            template_policy=template_policy,
        )

        actions: List[DecisionAction] = [a for a, _ in scored]
        score_dicts: List[Dict[str, Any]] = [dict(d) for _, d in scored]

        apply_opening_book_scores(
            self.opening_book,
            frame,
            position_state,
            actions,
            score_dicts,
        )

        apply_endgame_tablebases(
            self.tablebases,
            frame,
            position_state,
            actions,
            score_dicts,
        )

        indexed: List[Tuple[int, DecisionAction, Dict[str, Any]]] = []
        for idx, (action, score_dict) in enumerate(zip(actions, score_dicts)):
            indexed.append((idx, action, score_dict))

        indexed.sort(key=lambda item: (-(item[2].get("unified_score", 0.0)), item[0]))

        return [(action, score_dict) for _, action, score_dict in indexed]
