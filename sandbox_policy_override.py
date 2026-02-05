import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SandboxPolicyOverride:
    """Experimental policy wrapper that makes aggressive entries/exits.

    - Does not mutate the wrapped policy.
    - Only used in sandbox replay; guardrails/health/anomaly remain untouched.
    """

    def __init__(self, base_policy_fn):
        self._base_policy_fn = base_policy_fn

    def evaluate(self, eval_result: Dict[str, Any], snapshot) -> Dict[str, Any]:
        # Start from the real policy output to avoid altering engine defaults.
        base = self._base_policy_fn(eval_result, snapshot)
        result = dict(base)

        confidence = eval_result.get("confidence", 0.0)
        factors = eval_result.get("subsystem_scores", {}) or {}
        trend = snapshot.market_state.get("trend_direction") or "NEUTRAL"
        momentum = snapshot.market_state.get("factors", {}).get("momentum", 0.0)

        aggressive = confidence >= 0.05
        trend_align = trend in {"UP", "BULL"} and momentum > 0
        reversal = trend in {"DOWN", "BEAR"} and momentum < 0

        # Force entries when trend and momentum align and confidence cleared.
        if aggressive and trend_align:
            result["action"] = "ENTER_FULL"
            result["target_size"] = max(1.0, result.get("target_size", 0.0))
            result["reasoning"] = (
                f"Sandbox aggressive entry (conf={confidence:.3f}, trend={trend}, mom={momentum:.3f})"
            )

        # Force exits on reversal hints.
        if aggressive and reversal:
            result["action"] = "EXIT"
            result["target_size"] = 0.0
            result["reasoning"] = (
                f"Sandbox exit on reversal (conf={confidence:.3f}, trend={trend}, mom={momentum:.3f})"
            )

        # Ignore slippage/spread constraints but note it in reasoning/log.
        if result.get("action") not in {"DO_NOTHING", "HOLD"}:
            note = " | sandbox ignoring slippage/spread constraints"
            result["reasoning"] = (result.get("reasoning") or "") + note
            logger.info(
                "SandboxPolicyOverride action=%s size=%.3f conf=%.3f trend=%s mom=%.3f",
                result.get("action"),
                result.get("target_size", 0.0),
                confidence,
                trend,
                momentum,
            )

        return result
