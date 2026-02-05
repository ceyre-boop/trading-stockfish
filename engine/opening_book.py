from typing import Dict, List

from .decision_actions import ActionType, DecisionAction
from .decision_frame import DecisionFrame


def _normalize(text: str) -> str:
    return (text or "").upper()


def _action_id(action: DecisionAction) -> str:
    action_type = (
        action.action_type.value
        if hasattr(action.action_type, "value")
        else str(action.action_type)
    )
    entry_id = action.entry_model_id or ""
    direction = _normalize(action.direction)
    size = _normalize(action.size_bucket)
    return f"{action_type}:{entry_id}:{direction}:{size}"


def build_action_id(action: DecisionAction) -> str:
    return _action_id(action)


def _entry_archetype(entry_model_id: str) -> str:
    entry_upper = _normalize(entry_model_id)
    if any(tag in entry_upper for tag in ("MR", "MEAN", "REVERT")):
        return "MEAN_REVERSION"
    if any(tag in entry_upper for tag in ("BO", "BREAK", "BRK", "MOMO", "MOM")):
        return "BREAKOUT"
    return "GENERIC"


class OpeningBookV1:
    def __init__(self):
        # Rules are embedded; keep deterministic and extensible.
        self.forbid_penalty = -1e6
        self.discourage_penalty = -0.25
        self.pref_small_boost = 0.25
        self.pref_strong_boost = 0.5

    def lookup(
        self,
        frame: DecisionFrame,
        position_state,
        candidate_actions: List[DecisionAction],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        session = _normalize(getattr(frame, "session_profile", "")) if frame else ""
        trend = _normalize(getattr(frame, "trend_regime", "")) if frame else ""
        vol = _normalize(getattr(frame, "vol_regime", "")) if frame else ""
        liquidity_state = ""
        if frame and isinstance(frame.liquidity_frame, dict):
            liquidity_state = (frame.liquidity_frame.get("state") or "").lower()

        trend_up = "UP" in trend
        trend_down = "DOWN" in trend
        strong_trend = "STRONG" in trend

        for action in candidate_actions:
            aid = _action_id(action)
            score = 0.0

            is_open = action.action_type in (
                ActionType.OPEN_LONG,
                ActionType.OPEN_SHORT,
            )
            is_no_trade = action.action_type == ActionType.NO_TRADE
            direction = _normalize(action.direction)
            size_bucket = _normalize(action.size_bucket)
            archetype = (
                _entry_archetype(action.entry_model_id or "") if is_open else "GENERIC"
            )

            # Session-based heuristics
            if session == "PROFILE_1A":
                if is_open:
                    aligned_up = trend_up and direction == "LONG"
                    aligned_down = trend_down and direction == "SHORT"
                    counter_up = trend_up and direction == "SHORT"
                    counter_down = trend_down and direction == "LONG"

                    if aligned_up or aligned_down:
                        score += self.pref_strong_boost
                    if archetype == "MEAN_REVERSION":
                        score += self.discourage_penalty
                    if counter_up or counter_down:
                        score += self.forbid_penalty
                if is_no_trade:
                    score += self.discourage_penalty

            elif session == "PROFILE_1B":
                if is_open:
                    if archetype == "MEAN_REVERSION":
                        score += self.pref_strong_boost
                    if archetype == "BREAKOUT":
                        score += self.discourage_penalty
                    if size_bucket in ("LARGE", "XL"):
                        score += (
                            self.forbid_penalty
                            if strong_trend
                            else self.discourage_penalty
                        )
                if is_no_trade:
                    score += self.discourage_penalty / 2.0

            elif session == "PROFILE_1C":
                if is_no_trade:
                    score += self.pref_strong_boost
                if is_open:
                    score += self.discourage_penalty
                    # allow if strong displacement via high vol + strong trend
                    if vol == "HIGH" and strong_trend:
                        score += 0.2

            # Liquidity constraints
            if liquidity_state == "thin":
                if size_bucket in ("LARGE", "XL"):
                    score += self.forbid_penalty
                if is_open:
                    score += self.discourage_penalty

            # Volatility constraints
            if vol == "HIGH" and is_open:
                stop = action.stop_structure or {}
                if "pct" in stop:
                    try:
                        pct = float(stop.get("pct"))
                        if pct >= 0.01:
                            score += self.pref_small_boost
                    except Exception:
                        pass
            if vol == "LOW" and is_open and archetype == "BREAKOUT":
                score += self.discourage_penalty

            scores[aid] = float(score)

        return scores
