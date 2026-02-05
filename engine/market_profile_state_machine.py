from typing import Dict, Optional

from .market_profile_features import MarketProfileFeatures

MarketProfileState = str


class MarketProfileStateMachine:
    def __init__(
        self, thresholds: Dict[str, float], min_dwell_bars: int = 3, hysteresis: int = 2
    ):
        self.thresholds = {
            "prob_manipulation": 0.5,
            "prob_distribution": 0.5,
            "prob_accumulation": 0.5,
            "atr_compress": 0.8,
            "displacement_high": 0.6,
            "low_conf": 0.35,
            **(thresholds or {}),
        }
        self.min_dwell_bars = max(1, int(min_dwell_bars))
        self.hysteresis = max(1, int(hysteresis))
        self.state: MarketProfileState = "ACCUMULATION"
        self.state_bars: int = 0
        self.pending: Optional[str] = None
        self.pending_count: int = 0

    def _reset_pending(self):
        self.pending = None
        self.pending_count = 0

    def _low_conf(self, probs: Dict[str, float]) -> bool:
        return max(probs.values()) < self.thresholds["low_conf"]

    def step(
        self, probs: Dict[str, float], features: MarketProfileFeatures
    ) -> Dict[str, Optional[str]]:
        self.state_bars += 1
        from_state = self.state
        to_state = None
        transition_reason = None

        if self._low_conf(probs):
            self.state = "TRANSITION"
            self.state_bars = 0
            self._reset_pending()
            return {
                "state": self.state,
                "confidence": max(probs.values()),
                "from_state": from_state,
                "to_state": self.state,
                "transition_reason": "low_confidence",
            }

        # Helper flags
        any_sweep = any(
            [
                features.swept_pdh,
                features.swept_pdl,
                features.swept_session_high,
                features.swept_session_low,
                features.swept_equal_highs,
                features.swept_equal_lows,
            ]
        )
        trend_dir = features.trend_dir_ltf.upper()

        # Transition conditions
        can_acc_to_man = (
            any_sweep
            and features.displacement_score >= self.thresholds["displacement_high"]
            and probs.get("MANIPULATION", 0.0) >= self.thresholds["prob_manipulation"]
            and self.state == "ACCUMULATION"
            and self.state_bars >= self.min_dwell_bars
        )

        can_man_to_dist = (
            trend_dir in {"UP", "DOWN"}
            and probs.get("DISTRIBUTION", 0.0) >= self.thresholds["prob_distribution"]
            and self.state == "MANIPULATION"
            and self.state_bars >= self.min_dwell_bars
        )

        can_dist_to_acc = (
            trend_dir == "FLAT"
            and features.atr_vs_session_baseline < self.thresholds["atr_compress"]
            and probs.get("ACCUMULATION", 0.0) >= self.thresholds["prob_accumulation"]
            and self.state == "DISTRIBUTION"
            and self.state_bars >= self.min_dwell_bars
        )

        desired_state = None
        if can_acc_to_man:
            desired_state = "MANIPULATION"
            transition_reason = "sweep_displacement"
        elif can_man_to_dist:
            desired_state = "DISTRIBUTION"
            transition_reason = "trend_continuation"
        elif can_dist_to_acc:
            desired_state = "ACCUMULATION"
            transition_reason = "compression"

        if desired_state:
            if self.pending == desired_state:
                self.pending_count += 1
            else:
                self.pending = desired_state
                self.pending_count = 1

            if self.pending_count >= self.hysteresis:
                self.state = desired_state
                self.state_bars = 0
                to_state = desired_state
                self._reset_pending()
        else:
            self._reset_pending()

        return {
            "state": self.state,
            "confidence": max(probs.values()),
            "from_state": from_state,
            "to_state": to_state,
            "transition_reason": transition_reason if to_state else None,
        }
