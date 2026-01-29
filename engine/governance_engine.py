"""
GovernanceEngine for Trading Stockfish v4.0‑E
Deterministic, replay-safe meta-governance and adaptive veto logic.
"""

from collections import deque
from typing import Any, Dict, Optional


class GovernanceDecision:
    def __init__(
        self,
        approved: bool,
        reason: str,
        adjusted_action: Optional[str] = None,
        adjusted_size: Optional[float] = None,
    ):
        self.approved = approved
        self.reason = reason
        self.adjusted_action = adjusted_action
        self.adjusted_size = adjusted_size

    def to_dict(self):
        return {
            "approved": self.approved,
            "reason": self.reason,
            "adjusted_action": self.adjusted_action,
            "adjusted_size": self.adjusted_size,
        }


class GovernanceEngine:
    def __init__(
        self,
        max_drawdown_threshold: float = -0.05,
        event_safety_factor: float = 0.5,
        transition_threshold: float = 0.2,
        max_trades_per_hour: int = 10,
        buffer_size: int = 60,
    ):
        self.max_drawdown_threshold = max_drawdown_threshold
        self.event_safety_factor = event_safety_factor
        self.transition_threshold = transition_threshold
        self.max_trades_per_hour = max_trades_per_hour
        self.drawdown_buffer = deque(maxlen=buffer_size)
        self.volatility_buffer = deque(maxlen=buffer_size)
        self.liquidity_shock_buffer = deque(maxlen=buffer_size)
        self.volatility_shock_buffer = deque(maxlen=buffer_size)
        self.trade_times = deque(maxlen=buffer_size)
        self.recent_vetoes = deque(maxlen=buffer_size)

    def update_buffers(self, market_state: Any, execution: Dict):
        """Update buffers from either dict or dataclass MarketState.

        This keeps the governance layer deterministic while accepting both
        mapping-based states and typed MarketState objects used by the policy
        pipeline.
        """

        def _get(obj: Any, key: str, default: Any = None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        self.drawdown_buffer.append(execution.get("unrealized_pnl", 0.0))

        vol_state = _get(market_state, "volatility_state", {}) or {}
        liq_state = _get(market_state, "liquidity_state", {}) or {}

        self.volatility_buffer.append(_get(vol_state, "vol_regime", "NORMAL"))
        self.liquidity_shock_buffer.append(_get(liq_state, "liquidity_shock", False))
        self.volatility_shock_buffer.append(_get(vol_state, "volatility_shock", False))
        self.trade_times.append(_get(market_state, "timestamp", 0))

    def trades_in_last_hour(self, current_time: Any) -> int:
        def _to_seconds(val: Any) -> float:
            if hasattr(val, "timestamp"):
                return float(val.timestamp())
            if hasattr(val, "total_seconds"):
                return float(val.total_seconds())
            try:
                return float(val)
            except Exception:
                return 0.0

        now_sec = _to_seconds(current_time)
        return sum(1 for t in self.trade_times if now_sec - _to_seconds(t) < 3600)

    def decide(
        self,
        market_state: Dict,
        eval_output: Dict,
        policy_decision: Dict,
        execution: Dict,
    ) -> GovernanceDecision:
        self.update_buffers(market_state, execution)
        reason = ""
        veto = False
        adjusted_action = None
        adjusted_size = None

        def _get(obj: Any, key: str, default: Any = None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Risk constraint gate (deterministic, honor upstream risk manager decisions)
        if policy_decision.get("risk_allowed") is False:
            veto = True
            reason = policy_decision.get(
                "risk_veto_reason", "RISK_CONSTRAINT_VIOLATION"
            )
            adjusted_action = "FLAT"
            adjusted_size = 0.0
            self.recent_vetoes.append(reason)
            return GovernanceDecision(
                approved=False,
                reason=reason,
                adjusted_action=adjusted_action,
                adjusted_size=adjusted_size,
            )

        amd_state = _get(market_state, "amd_state", {}) or {}
        amd_tag = _get(amd_state, "amd_tag", "NEUTRAL")

        swing_structure = _get(market_state, "swing_structure", None) or _get(
            _get(market_state, "regime_state", {}), "swing_structure", "NEUTRAL"
        )
        trend_direction = _get(market_state, "trend_direction", None) or _get(
            _get(market_state, "regime_state", {}), "trend_direction", "RANGE"
        )
        trend_strength = float(
            _get(market_state, "trend_strength", 0.0)
            or _get(_get(market_state, "regime_state", {}), "trend_strength", 0.0)
            or 0.0
        )
        eval_score = float(eval_output.get("eval_score", 0.0))

        # 1. Drawdown protection
        if (
            not veto
            and execution.get("unrealized_pnl", 0.0) < self.max_drawdown_threshold
        ):
            veto = True
            reason = "DRAWDOWN_LIMIT"
        # 1a. Manipulation veto
        elif amd_tag == "MANIPULATION" and policy_decision.get("action") not in [
            "EXIT",
            "FLAT",
            "REDUCE",
        ]:
            veto = True
            reason = "AMD_MANIPULATION_VETO"
            adjusted_action = "FLAT"
            adjusted_size = 0.0
        # 1b. Counter-trend guard using deterministic swing structure
        elif trend_direction and trend_direction != "RANGE" and trend_strength > 0.0:
            counter_trend = (trend_direction == "UP" and eval_score < 0) or (
                trend_direction == "DOWN" and eval_score > 0
            )
            if counter_trend and policy_decision.get("action") not in [
                "EXIT",
                "REDUCE",
                "FLAT",
            ]:
                veto = True
                reason = "COUNTER_TREND_VETO"
                adjusted_action = "REDUCE"
                base_size = policy_decision.get(
                    "target_size", policy_decision.get("size", 0.0)
                )
                adjusted_size = min(base_size * 0.5, 0.25)
        # 2. Volatility shock veto (harder than regime EXTREME)
        elif _get(
            _get(market_state, "volatility_state", {}), "volatility_shock", False
        ):
            shock_strength = float(
                _get(
                    _get(market_state, "volatility_state", {}),
                    "volatility_shock_strength",
                    0.0,
                )
            )
            if policy_decision.get("action") not in ["EXIT", "REDUCE", "FLAT"]:
                veto = True
                reason = "VOLATILITY_SHOCK"
                adjusted_action = "FLAT"
                adjusted_size = 0.0
            elif shock_strength > 0.7 and policy_decision.get("action") != "FLAT":
                veto = True
                reason = "VOLATILITY_SHOCK_SIZE_REDUCTION"
                adjusted_action = "REDUCE"
                base_size = policy_decision.get(
                    "target_size", policy_decision.get("size", 0.0)
                )
                adjusted_size = min(base_size * 0.5, 0.25)
        # 2b. Volatility override
        elif (
            _get(_get(market_state, "volatility_state", {}), "vol_regime", "NORMAL")
        ) == "EXTREME":
            if policy_decision.get("action") not in ["EXIT", "REDUCE", "FLAT"]:
                veto = True
                reason = "EXTREME_VOLATILITY"
                adjusted_action = "FLAT"
                adjusted_size = 0.0
        # 3. Liquidity shock override
        elif _get(_get(market_state, "liquidity_state", {}), "liquidity_shock", False):
            if policy_decision.get("action") != "FLAT":
                veto = True
                reason = "LIQUIDITY_SHOCK"
                adjusted_action = "EXIT"
                adjusted_size = 0.0
        # 4. Macro event override
        elif (
            _get(_get(market_state, "regime_state", {}), "macro_regime", "") == "EVENT"
        ):
            adjusted_size = policy_decision.get("size", 1.0) * self.event_safety_factor
            reason = "MACRO_EVENT_SIZE_REDUCED"
        # 5. Regime transition override
        elif (
            _get(_get(market_state, "regime_state", {}), "regime_transition", False)
            and eval_output.get("eval_score", 0.0) < self.transition_threshold
        ):
            veto = True
            reason = "REGIME_TRANSITION_UNSAFE"
        # 6. Trade frequency limit
        elif (
            self.trades_in_last_hour(_get(market_state, "timestamp", 0))
            > self.max_trades_per_hour
        ):
            veto = True
            reason = "TRADE_FREQUENCY_LIMIT"
        else:
            reason = "APPROVED"
        self.recent_vetoes.append(reason if veto else None)
        return GovernanceDecision(
            approved=not veto,
            reason=reason,
            adjusted_action=adjusted_action,
            adjusted_size=adjusted_size,
        )
