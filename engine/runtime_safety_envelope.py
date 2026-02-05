from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from .decision_actions import ActionType, DecisionAction
from .decision_frame import DecisionFrame


class SafeMode(str, Enum):
    OFFLINE_ONLY = "offline_only"
    PAPER_ONLY = "paper_only"
    LIVE_THROTTLED = "live_throttled"
    LIVE_FULL = "live_full"


@dataclass
class SafetyDecision:
    allowed: bool
    final_action: DecisionAction
    reason: str
    details: Dict[str, Any]


class RuntimeSafetyEnvelope:
    def __init__(
        self,
        safe_mode: SafeMode,
        global_risk_limits: Dict[str, Any],
        regime_rules: Dict[str, Any],
        brain_trust_thresholds: Dict[str, Any],
    ) -> None:
        self.safe_mode = safe_mode
        self.global_risk_limits = global_risk_limits or {}
        self.regime_rules = regime_rules or {}
        self.brain_trust_thresholds = brain_trust_thresholds or {}

    def _no_trade(self) -> DecisionAction:
        return DecisionAction(action_type=ActionType.NO_TRADE)

    def _mode_guard(
        self, action: DecisionAction
    ) -> tuple[DecisionAction, str, Dict[str, Any], bool]:
        if self.safe_mode == SafeMode.OFFLINE_ONLY:
            return (
                self._no_trade(),
                "safe_mode_offline_only",
                {"mode": self.safe_mode.value},
                True,
            )
        if self.safe_mode == SafeMode.PAPER_ONLY:
            return action, "safe_mode_paper_only", {"mode": self.safe_mode.value}, True
        # LIVE_THROTTLED and LIVE_FULL fall through
        return action, "mode_ok", {"mode": self.safe_mode.value}, True

    def _global_risk_guard(
        self, action: DecisionAction, risk_envelope: Dict[str, Any]
    ) -> tuple[DecisionAction, str, Dict[str, Any], bool]:
        re = risk_envelope or {}
        limits = self.global_risk_limits

        daily_R = float(re.get("daily_R_loss", 0.0) or 0.0)
        max_daily_R = float(limits.get("max_daily_R_loss", float("inf")))
        open_risk = float(re.get("open_risk", 0.0) or 0.0)
        max_open_risk = float(limits.get("max_concurrent_risk", float("inf")))
        position_count = float(re.get("position_count", 0) or 0)
        max_positions_raw = limits.get("max_position_count", float("inf"))
        max_positions = (
            float(max_positions_raw)
            if max_positions_raw not in (None, "inf", float("inf"))
            else float("inf")
        )
        per_symbol_risk = float(re.get("per_symbol_risk", 0.0) or 0.0)
        per_symbol_cap = float(limits.get("per_symbol_risk_cap", float("inf")))

        breached = False
        details: Dict[str, Any] = {}
        if daily_R <= -abs(max_daily_R):
            breached = True
            details["daily_R_loss"] = daily_R
        if open_risk > max_open_risk:
            breached = True
            details["open_risk"] = open_risk
        if position_count > max_positions:
            breached = True
            details["position_count"] = position_count
        if per_symbol_risk > per_symbol_cap:
            breached = True
            details["per_symbol_risk"] = per_symbol_risk

        if breached:
            return self._no_trade(), "global_risk_violation", details, True
        return action, "global_risk_ok", {}, True

    def _regime_guard(
        self, frame: DecisionFrame, action: DecisionAction
    ) -> tuple[DecisionAction, str, Dict[str, Any], bool]:
        reason = "regime_ok"
        details: Dict[str, Any] = {}
        final_action = action

        vol = (getattr(frame, "vol_regime", "") or "").lower()
        liquidity_state = ""
        if getattr(frame, "liquidity_frame", None) and isinstance(
            frame.liquidity_frame, dict
        ):
            liquidity_state = (frame.liquidity_frame.get("state") or "").lower()
        news_state = (getattr(frame, "news_state", "") or "").lower()

        is_open_action = action.action_type in (
            ActionType.OPEN_LONG,
            ActionType.OPEN_SHORT,
        )
        size_bucket = (action.size_bucket or "").upper()

        if vol in {"extreme"} and is_open_action:
            final_action = self._no_trade()
            reason = "regime_vol_extreme"
            details["vol_regime"] = vol

        if liquidity_state in {"broken", "thin"} and is_open_action:
            if size_bucket in {"LARGE", "XL", "AGGRESSIVE"} or size_bucket == "":
                final_action = self._no_trade()
                reason = "regime_liquidity_thin"
                details["liquidity_state"] = liquidity_state

        if news_state == "high_impact" and is_open_action:
            final_action = self._no_trade()
            reason = "regime_news_high_impact"
            details["news_state"] = news_state

        return final_action, reason, details, True

    def _brain_trust_guard(
        self, action: DecisionAction, intent: Dict[str, Any]
    ) -> tuple[DecisionAction, str, Dict[str, Any], bool]:
        thresholds = self.brain_trust_thresholds
        scores = intent.get("scores", {}) if isinstance(intent, dict) else {}
        mcr = scores.get("MCR", {}) if isinstance(scores, dict) else {}

        ev_hat = float(scores.get("EV_brain", 0.0) or 0.0)
        unified = float(scores.get("unified_score", 0.0) or 0.0)
        variance = float(mcr.get("variance_EV", 0.0) or 0.0)
        policy_label = (
            scores.get("policy_label") or intent.get("policy_label") or ""
        ).upper()

        min_ev = thresholds.get("min_ev", -float("inf"))
        min_unified = thresholds.get("min_unified", -float("inf"))
        max_variance = thresholds.get("max_variance", float("inf"))

        details: Dict[str, Any] = {}
        reason = "brain_trust_ok"
        final_action = action

        if ev_hat < min_ev:
            final_action = self._no_trade()
            reason = "brain_trust_low_ev"
            details["EV_brain"] = ev_hat
        if unified < min_unified:
            final_action = self._no_trade()
            reason = "brain_trust_low_unified"
            details["unified_score"] = unified
        if variance > max_variance:
            final_action = self._no_trade()
            reason = "brain_trust_high_variance"
            details["variance_EV"] = variance
        if policy_label in {"DISCOURAGED", "DISABLED"}:
            final_action = self._no_trade()
            reason = "brain_trust_policy_block"
            details["policy_label"] = policy_label

        return final_action, reason, details, True

    def evaluate_intent(
        self,
        frame: DecisionFrame,
        position_state: Any,
        risk_envelope: Any,
        intent: Dict[str, Any],
    ) -> SafetyDecision:
        chosen_action = (
            intent.get("chosen_action") if isinstance(intent, dict) else None
        )
        if not isinstance(chosen_action, DecisionAction):
            chosen_action = self._no_trade()

        final_action, reason, details, allowed = self._mode_guard(chosen_action)
        if self.safe_mode == SafeMode.OFFLINE_ONLY:
            return SafetyDecision(
                allowed=allowed,
                final_action=final_action,
                reason=reason,
                details=details,
            )

        # Global risk guard
        ga_action, ga_reason, ga_details, _ = self._global_risk_guard(
            final_action, risk_envelope or {}
        )
        if ga_reason != "global_risk_ok":
            return SafetyDecision(True, ga_action, ga_reason, ga_details)

        # Regime guard
        rg_action, rg_reason, rg_details, _ = self._regime_guard(frame, ga_action)
        if rg_reason != "regime_ok":
            return SafetyDecision(True, rg_action, rg_reason, rg_details)

        # Brain-trust guard
        bt_action, bt_reason, bt_details, _ = self._brain_trust_guard(rg_action, intent)
        if bt_reason != "brain_trust_ok":
            return SafetyDecision(True, bt_action, bt_reason, bt_details)

        return SafetyDecision(True, bt_action, reason, details)
