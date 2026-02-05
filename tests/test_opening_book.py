from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.opening_book import OpeningBookV1, build_action_id
from engine.search_scoring import apply_opening_book_scores


def test_action_id_construction():
    action = DecisionAction(
        action_type=ActionType.OPEN_LONG,
        entry_model_id="MODEL_X",
        direction="long",
        size_bucket="medium",
    )
    ob = OpeningBookV1()
    scores = ob.lookup(DecisionFrame(), {}, [action])
    assert list(scores.keys()) == ["OPEN_LONG:MODEL_X:LONG:MEDIUM"]


def test_session_1a_trend_aligned_prefers_continuation():
    frame = DecisionFrame(session_profile="PROFILE_1A", trend_regime="UP_STRONG")
    aligned = DecisionAction(
        action_type=ActionType.OPEN_LONG, entry_model_id="BO_A", direction="LONG"
    )
    mr = DecisionAction(
        action_type=ActionType.OPEN_LONG, entry_model_id="MR_A", direction="LONG"
    )
    counter = DecisionAction(
        action_type=ActionType.OPEN_SHORT, entry_model_id="BO_B", direction="SHORT"
    )

    ob = OpeningBookV1()
    scores = ob.lookup(frame, {}, [aligned, mr, counter])

    assert scores[build_action_id(aligned)] > scores[build_action_id(mr)]
    assert scores[build_action_id(counter)] <= -1e5


def test_session_1b_prefers_mean_reversion_discourages_breakout():
    frame = DecisionFrame(session_profile="PROFILE_1B", trend_regime="NEUTRAL")
    mr = DecisionAction(
        action_type=ActionType.OPEN_LONG, entry_model_id="MEAN_REV_X", direction="LONG"
    )
    breakout = DecisionAction(
        action_type=ActionType.OPEN_LONG, entry_model_id="BREAKOUT_X", direction="LONG"
    )

    ob = OpeningBookV1()
    scores = ob.lookup(frame, {}, [mr, breakout])

    assert scores[build_action_id(mr)] > scores[build_action_id(breakout)]


def test_session_1c_prefers_no_trade():
    frame = DecisionFrame(session_profile="PROFILE_1C")
    no_trade = DecisionAction(action_type=ActionType.NO_TRADE)
    open_long = DecisionAction(
        action_type=ActionType.OPEN_LONG, entry_model_id="GENERIC", direction="LONG"
    )

    ob = OpeningBookV1()
    scores = ob.lookup(frame, {}, [no_trade, open_long])

    assert scores[build_action_id(no_trade)] > scores[build_action_id(open_long)]


def test_liquidity_and_volatility_constraints():
    frame = DecisionFrame(
        session_profile="PROFILE_1A",
        vol_regime="HIGH",
        liquidity_frame={"state": "thin"},
    )
    large = DecisionAction(
        action_type=ActionType.OPEN_LONG,
        entry_model_id="BO_LARGE",
        direction="LONG",
        size_bucket="XL",
    )
    small_wide = DecisionAction(
        action_type=ActionType.OPEN_LONG,
        entry_model_id="BO_SMALL",
        direction="LONG",
        size_bucket="SMALL",
        stop_structure={"pct": 0.02},
    )

    ob = OpeningBookV1()
    scores = ob.lookup(frame, {}, [large, small_wide])

    assert scores[build_action_id(large)] <= -1e5
    assert scores[build_action_id(small_wide)] == 0.0


def test_low_vol_discourages_breakout():
    frame = DecisionFrame(vol_regime="LOW")
    breakout = DecisionAction(
        action_type=ActionType.OPEN_LONG, entry_model_id="BREAKOUT_Z", direction="LONG"
    )

    ob = OpeningBookV1()
    scores = ob.lookup(frame, {}, [breakout])

    assert scores[build_action_id(breakout)] < 0


def test_apply_opening_book_scores_integration():
    frame = DecisionFrame(session_profile="PROFILE_1A", trend_regime="UP")
    actions = [
        DecisionAction(
            action_type=ActionType.OPEN_LONG, entry_model_id="BO_A", direction="LONG"
        ),
        DecisionAction(
            action_type=ActionType.OPEN_SHORT, entry_model_id="BO_B", direction="SHORT"
        ),
    ]
    score_dicts = [{"unified_score": 1.0}, {"unified_score": 1.0}]

    ob = OpeningBookV1()
    apply_opening_book_scores(ob, frame, {}, actions, score_dicts)

    aligned_boost = ob.lookup(frame, {}, [actions[0]])[build_action_id(actions[0])]

    assert score_dicts[0]["opening_book_score"] == aligned_boost
    assert score_dicts[0]["unified_score"] == 1.0 + aligned_boost
    assert score_dicts[1]["opening_book_score"] < 0
    assert score_dicts[1]["unified_score"] < 1.0
