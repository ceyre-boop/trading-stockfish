from engine.live_modes import LiveMode
from engine.mode_guard import ModeGuard


def test_mode_guard_blocks_routing_in_sim_modes():
    guard = ModeGuard(LiveMode.SIM_LIVE_FEED)
    base_intent = {"chosen_action": "OPEN_LONG", "metadata": {}}
    enforced = guard.enforce(base_intent)

    assert enforced["routing_allowed"] is False
    assert enforced["position_updates_allowed"] is False
    assert enforced["mode_info"]["mode"] == LiveMode.SIM_LIVE_FEED.value
    # input should remain unchanged (deterministic, no side effects)
    assert "routing_allowed" not in base_intent
    assert base_intent["metadata"] == {}


def test_mode_guard_allows_routing_in_live_throttled():
    guard = ModeGuard(LiveMode.LIVE_THROTTLED)
    intent = {"chosen_action": "OPEN_LONG"}

    enforced_a = guard.enforce(intent)
    enforced_b = guard.enforce(intent)

    assert enforced_a["routing_allowed"] is True
    assert enforced_a["position_updates_allowed"] is True
    assert enforced_a["mode_info"]["mode"] == LiveMode.LIVE_THROTTLED.value
    # deterministic output on repeated calls
    assert enforced_a == enforced_b
