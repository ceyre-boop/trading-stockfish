from typing import Dict, List

from .decision_actions import ActionType, DecisionAction
from .decision_frame import DecisionFrame
from .opening_book import build_action_id


def _normalize(text: str) -> str:
    return (text or "").lower()


def _is_close_manage(action: DecisionAction) -> bool:
    if action.action_type != ActionType.MANAGE_POSITION:
        return False
    payload = action.manage_payload or {}
    intent = _normalize(payload.get("action")) if isinstance(payload, dict) else ""
    return intent in {"close", "close_position", "scale_out", "reduce", "exit"}


class EndgameTablebasesV1:
    def __init__(self):
        self.forbid_penalty = -1e6
        self.pref_strong = 0.75
        self.pref_medium = 0.5
        self.pref_small = 0.25
        self.discourage = -0.1

    def lookup(
        self,
        frame: DecisionFrame,
        position_state,
        candidate_actions: List[DecisionAction],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        unrealized_R = 0.0
        if isinstance(position_state, dict):
            unrealized_R = float(position_state.get("unrealized_R", 0.0) or 0.0)
        is_open = isinstance(position_state, dict) and bool(
            position_state.get("is_open", False)
        )
        drawdown_increasing = (
            bool(position_state.get("drawdown_increasing", False))
            if isinstance(position_state, dict)
            else False
        )
        risk_env = (
            position_state.get("risk_envelope", {})
            if isinstance(position_state, dict)
            else {}
        )
        max_daily_risk_hit = (
            bool(risk_env.get("max_daily_risk_hit", False))
            if isinstance(risk_env, dict)
            else False
        )

        news_state = _normalize(getattr(frame, "news_state", "")) if frame else ""
        vol_regime = _normalize(getattr(frame, "vol_regime", "")) if frame else ""
        htf_target_hit = (
            bool(getattr(frame, "htf_target_hit", False)) if frame else False
        )

        for action in candidate_actions:
            aid = build_action_id(action)
            score = 0.0

            is_open_action = action.action_type in (
                ActionType.OPEN_LONG,
                ActionType.OPEN_SHORT,
            )
            is_no_trade = action.action_type == ActionType.NO_TRADE
            is_close_manage = _is_close_manage(action)
            size_bucket = _normalize(action.size_bucket)

            # (A) Deep ITM runners
            if unrealized_R >= 3.0:
                if is_open_action:
                    score += self.forbid_penalty
                if is_close_manage:
                    score += self.pref_strong
                if is_no_trade:
                    score += self.discourage

            # (B) News spikes
            if news_state == "high_impact":
                if is_open_action:
                    score += self.forbid_penalty
                if is_close_manage and is_open:
                    score += self.pref_strong
                if is_no_trade:
                    score += self.pref_small

            # (C) Volatility explosions
            if vol_regime == "extreme":
                if is_open_action and size_bucket in {"large", "xl"}:
                    score += self.forbid_penalty
                elif is_open_action:
                    score += self.discourage
                if is_no_trade:
                    score += self.pref_small
                if is_close_manage and drawdown_increasing:
                    score += self.pref_medium

            # (D) HTF target hits
            if htf_target_hit:
                if is_open_action:
                    score += self.forbid_penalty
                if is_close_manage:
                    score += self.pref_medium

            # (E) Max-risk conditions
            if max_daily_risk_hit:
                if is_open_action:
                    score += self.forbid_penalty
                if is_no_trade:
                    score += self.pref_small
                if is_close_manage:
                    score += self.pref_medium

            scores[aid] = float(score)

        return scores
