from engine.decision_actions import ActionType, DecisionAction
from engine.mcr_scenarios import MCRActionContext, PricePath
from engine.mcr_simulator import simulate_trade_on_path


def _ctx(action_type: ActionType):
    return MCRActionContext(
        decision_frame_ref="df1",
        initial_position_state={},
        decision_action=DecisionAction(action_type=action_type),
        risk_envelope={},
    )


def test_no_trade_returns_zeroes():
    price_path = PricePath(prices=[100, 101], metadata={"seed": 1})
    result = simulate_trade_on_path(price_path, _ctx(ActionType.NO_TRADE))
    assert result.realized_R == 0.0
    assert result.max_adverse_excursion == 0.0
    assert result.max_favorable_excursion == 0.0
    assert result.time_in_trade_bars == 0
    assert result.closed_by_rule is True


def test_open_long_hits_tp_on_rising_path():
    action = DecisionAction(action_type=ActionType.OPEN_LONG)
    ctx = MCRActionContext("df2", {}, action, {})
    price_path = PricePath(prices=[100, 101, 102.0])
    result = simulate_trade_on_path(price_path, ctx)
    assert result.hit_tp is True
    assert result.hit_stop is False
    assert result.realized_R == 2.0  # (102-100)/1
    assert result.time_in_trade_bars > 0


def test_open_long_hits_stop_on_falling_path():
    action = DecisionAction(action_type=ActionType.OPEN_LONG)
    ctx = MCRActionContext("df3", {}, action, {})
    price_path = PricePath(prices=[100, 99, 98])
    result = simulate_trade_on_path(price_path, ctx)
    assert result.hit_stop is True
    assert result.hit_tp is False
    assert result.realized_R == -1.0  # (99-100)/1


def test_open_short_mirror_behavior():
    action = DecisionAction(action_type=ActionType.OPEN_SHORT)
    ctx = MCRActionContext("df4", {}, action, {})

    # Falling path should hit TP for short
    price_path = PricePath(prices=[100, 99, 98])
    result = simulate_trade_on_path(price_path, ctx)
    assert result.hit_tp is True
    assert result.hit_stop is False
    assert result.realized_R == 2.0  # (100-98)/1

    # Rising path should hit stop for short
    price_path2 = PricePath(prices=[100, 101, 102])
    result2 = simulate_trade_on_path(price_path2, ctx)
    assert result2.hit_stop is True
    assert result2.hit_tp is False
    assert result2.realized_R == -1.0


def test_determinism_same_inputs_same_result():
    action = DecisionAction(action_type=ActionType.OPEN_LONG)
    ctx = MCRActionContext("df5", {}, action, {})
    price_path = PricePath(prices=[100, 101, 102])
    r1 = simulate_trade_on_path(price_path, ctx)
    r2 = simulate_trade_on_path(price_path, ctx)
    assert r1 == r2
