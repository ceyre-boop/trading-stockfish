from engine.live_modes import MODE_CAPABILITIES, LiveMode, ModeCapabilities


def test_mode_capabilities_mapping_complete():
    expected_modes = {
        LiveMode.SIM_REPLAY,
        LiveMode.SIM_LIVE_FEED,
        LiveMode.PAPER_TRADING,
        LiveMode.LIVE_THROTTLED,
        LiveMode.LIVE_FULL,
    }
    assert set(MODE_CAPABILITIES.keys()) == expected_modes


def test_mode_capabilities_fields_correctness():
    replay = MODE_CAPABILITIES[LiveMode.SIM_REPLAY]
    live_feed = MODE_CAPABILITIES[LiveMode.SIM_LIVE_FEED]
    paper = MODE_CAPABILITIES[LiveMode.PAPER_TRADING]
    throttled = MODE_CAPABILITIES[LiveMode.LIVE_THROTTLED]

    assert isinstance(replay, ModeCapabilities)
    assert replay.allow_live_prices is False
    assert replay.allow_order_routing is False

    assert live_feed.allow_live_prices is True
    assert live_feed.allow_order_routing is False

    assert paper.allow_position_updates is True
    assert paper.allow_order_routing is False

    assert throttled.allow_order_routing is True
    assert throttled.allow_execution_reports is True
    assert "Real orders" in throttled.description
