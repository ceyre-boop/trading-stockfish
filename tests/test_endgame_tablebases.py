from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.endgame_tablebases import EndgameTablebasesV1
from engine.opening_book import build_action_id
from engine.search_scoring import apply_endgame_tablebases

CLOSE_PAYLOAD = {"action": "CLOSE"}


def test_action_id_mapping():
    tb = EndgameTablebasesV1()
    action = DecisionAction(
        ActionType.OPEN_LONG, entry_model_id="X", direction="LONG", size_bucket="SMALL"
    )
    scores = tb.lookup(DecisionFrame(), {}, [action])
    assert list(scores.keys()) == ["OPEN_LONG:X:LONG:SMALL"]


def test_deep_itm_forces_manage_and_forbids_open():
    tb = EndgameTablebasesV1()
    frame = DecisionFrame()
    position_state = {"unrealized_R": 3.5, "is_open": True}

    open_action = DecisionAction(
        ActionType.OPEN_LONG, entry_model_id="A", direction="LONG"
    )
    close_action = DecisionAction(
        ActionType.MANAGE_POSITION, manage_payload=CLOSE_PAYLOAD
    )
    no_trade = DecisionAction(ActionType.NO_TRADE)

    scores = tb.lookup(frame, position_state, [open_action, close_action, no_trade])

    assert scores[build_action_id(open_action)] <= -1e5
    assert scores[build_action_id(close_action)] > scores[build_action_id(no_trade)]


def test_news_spike_prefers_close_and_no_trade():
    tb = EndgameTablebasesV1()
    frame = DecisionFrame()
    frame.news_state = "high_impact"
    position_state = {"is_open": True}

    open_action = DecisionAction(
        ActionType.OPEN_SHORT, entry_model_id="B", direction="SHORT"
    )
    close_action = DecisionAction(
        ActionType.MANAGE_POSITION, manage_payload=CLOSE_PAYLOAD
    )
    no_trade = DecisionAction(ActionType.NO_TRADE)

    scores = tb.lookup(frame, position_state, [open_action, close_action, no_trade])

    assert scores[build_action_id(open_action)] <= -1e5
    assert scores[build_action_id(close_action)] > scores[build_action_id(no_trade)]
    assert scores[build_action_id(no_trade)] > 0


def test_volatility_explosion_prefers_safety():
    tb = EndgameTablebasesV1()
    frame = DecisionFrame(vol_regime="extreme")
    position_state = {"drawdown_increasing": True, "is_open": True}

    open_large = DecisionAction(
        ActionType.OPEN_LONG, entry_model_id="C", direction="LONG", size_bucket="XL"
    )
    close_action = DecisionAction(
        ActionType.MANAGE_POSITION, manage_payload=CLOSE_PAYLOAD
    )
    no_trade = DecisionAction(ActionType.NO_TRADE)

    scores = tb.lookup(frame, position_state, [open_large, close_action, no_trade])

    assert scores[build_action_id(open_large)] <= -1e5
    assert scores[build_action_id(no_trade)] > 0
    assert scores[build_action_id(close_action)] > 0


def test_htf_target_hit_prefers_exit():
    tb = EndgameTablebasesV1()
    frame = DecisionFrame()
    frame.htf_target_hit = True
    position_state = {"is_open": True}

    open_action = DecisionAction(
        ActionType.OPEN_LONG, entry_model_id="D", direction="LONG"
    )
    close_action = DecisionAction(
        ActionType.MANAGE_POSITION, manage_payload=CLOSE_PAYLOAD
    )

    scores = tb.lookup(frame, position_state, [open_action, close_action])

    assert scores[build_action_id(open_action)] <= -1e5
    assert scores[build_action_id(close_action)] > 0


def test_max_risk_forbids_new_entries():
    tb = EndgameTablebasesV1()
    frame = DecisionFrame()
    position_state = {"risk_envelope": {"max_daily_risk_hit": True}, "is_open": True}

    open_action = DecisionAction(
        ActionType.OPEN_SHORT, entry_model_id="E", direction="SHORT"
    )
    close_action = DecisionAction(
        ActionType.MANAGE_POSITION, manage_payload=CLOSE_PAYLOAD
    )
    no_trade = DecisionAction(ActionType.NO_TRADE)

    scores = tb.lookup(frame, position_state, [open_action, close_action, no_trade])

    assert scores[build_action_id(open_action)] <= -1e5
    assert scores[build_action_id(no_trade)] > 0
    assert scores[build_action_id(close_action)] > 0


def test_apply_endgame_tablebases_integration():
    tb = EndgameTablebasesV1()
    frame = DecisionFrame()
    position_state = {"unrealized_R": 3.2, "is_open": True}

    open_action = DecisionAction(
        ActionType.OPEN_LONG, entry_model_id="A", direction="LONG"
    )
    close_action = DecisionAction(
        ActionType.MANAGE_POSITION, manage_payload=CLOSE_PAYLOAD
    )
    actions = [open_action, close_action]
    score_dicts = [{"unified_score": 1.0}, {"unified_score": 1.0}]

    apply_endgame_tablebases(tb, frame, position_state, actions, score_dicts)

    scores = tb.lookup(frame, position_state, actions)

    assert score_dicts[0]["endgame_score"] == scores[build_action_id(open_action)]
    assert score_dicts[0]["unified_score"] == 1.0 + scores[build_action_id(open_action)]
    assert score_dicts[1]["endgame_score"] == scores[build_action_id(close_action)]
    assert (
        score_dicts[1]["unified_score"] == 1.0 + scores[build_action_id(close_action)]
    )
