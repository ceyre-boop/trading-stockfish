import pytest

from engine.invariants import InvariantChecker, InvariantViolation


def test_no_order_routing_blocks_when_not_allowed():
    checker = InvariantChecker()
    intent = {"routing_allowed": True}
    mode_info = {"allow_order_routing": False}
    with pytest.raises(InvariantViolation):
        checker.assert_no_order_routing(intent, mode_info)


def test_safety_enforced_requires_downgrade():
    checker = InvariantChecker()
    intent = {
        "chosen_action": {"action_type": "OPEN_LONG"},
        "safety_decision": {
            "final_action": {"action_type": "NO_TRADE"},
            "allowed": False,
            "reason": "risk",
        },
    }
    with pytest.raises(InvariantViolation):
        checker.assert_safety_enforced(intent)


def test_environment_safe_blocks_trading_on_anomaly():
    checker = InvariantChecker()
    intent = {
        "environment": {"anomaly_decision": {"triggered": True}},
        "safety_decision": {
            "final_action": {"action_type": "OPEN_LONG"},
            "allowed": True,
        },
    }
    with pytest.raises(InvariantViolation):
        checker.assert_environment_safe(intent)


def test_brain_artifacts_validation():
    checker = InvariantChecker()
    intent = {"scores": {"unified_score": 1.0, "MCR": {"mean_EV": 0.1}}}
    with pytest.raises(InvariantViolation):
        checker.assert_brain_artifacts_valid(intent)
