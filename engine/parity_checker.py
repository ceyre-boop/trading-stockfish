from typing import Any, Dict, List

from .decision_frame import DecisionFrame
from .search_engine_v1 import SearchEngineV1


class ParityChecker:
    def __init__(self, search_engine: SearchEngineV1):
        self.search_engine = search_engine

    def compare_live_vs_replay(
        self,
        frame: DecisionFrame,
        position_state: Any,
        entry_models: List[Any],
        risk_envelope: Any,
        live_ranked_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        replay_ranked = self.search_engine.rank_actions(
            frame,
            position_state,
            entry_models,
            risk_envelope,
        )

        def _simplify(item):
            action, score = item
            return {
                "action": getattr(action, "__dict__", action),
                "unified_score": (
                    score.get("unified_score", None)
                    if isinstance(score, dict)
                    else None
                ),
                "scores": score,
            }

        live_simple = []
        for rec in live_ranked_actions or []:
            if isinstance(rec, tuple) and len(rec) == 2:
                live_simple.append(_simplify(rec))
            elif isinstance(rec, dict):
                live_simple.append(rec)

        replay_simple = [_simplify(r) for r in replay_ranked]

        match = live_simple == replay_simple

        live_top = live_simple[0] if live_simple else None
        replay_top = replay_simple[0] if replay_simple else None

        differences: List[str] = []
        if not match:
            if live_top != replay_top:
                differences.append("top_action_mismatch")
            if len(live_simple) != len(replay_simple):
                differences.append("length_mismatch")
            else:
                for idx, (l, r) in enumerate(zip(live_simple, replay_simple)):
                    if l != r:
                        differences.append(f"diff_at_{idx}")
                        break

        return {
            "match": match,
            "differences": differences,
            "live_top": live_top,
            "replay_top": replay_top,
        }
