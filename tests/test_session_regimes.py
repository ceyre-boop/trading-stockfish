from datetime import datetime, timezone

from engine.evaluator import evaluate_state
from engine.market_state_builder import build_market_state
from engine.regime_engine import RegimeEngine
from engine.types import MarketState
from session_regime import SessionRegime, compute_session_regime


def _ts(hour: int, minute: int = 0) -> float:
    return datetime(
        2024, 1, 1, hour=hour, minute=minute, tzinfo=timezone.utc
    ).timestamp()


def test_session_boundaries():
    assert compute_session_regime(_ts(0)) == SessionRegime.ASIA
    assert compute_session_regime(_ts(7, 59)) == SessionRegime.ASIA
    assert compute_session_regime(_ts(8)) == SessionRegime.LONDON
    assert compute_session_regime(_ts(13)) == SessionRegime.NEW_YORK


def test_session_overlap_priority():
    # 15:00 UTC falls in LONDON/NY overlap, NY wins
    assert compute_session_regime(_ts(15)) == SessionRegime.NEW_YORK


def test_replay_live_session_parity():
    ts = _ts(6)
    label = compute_session_regime(ts).value
    state = build_market_state(symbol="TEST", order_book_events=[], timestamp=ts)
    assert state["session_regime"] == label


def test_evaluator_session_tilts():
    base = MarketState(
        current_price=101.0,
        ma_short=100.0,
        ma_long=99.0,
        momentum=0.2,
        recent_returns=[0.01] * 5,
        volatility=0.1,
        liquidity=0.5,
        rsi=55.0,
    )
    asia = MarketState(**{**base.__dict__, "session": "ASIA"})
    london = MarketState(**{**base.__dict__, "session": "LONDON"})
    ny = MarketState(**{**base.__dict__, "session": "NEW_YORK"})

    conf_asia = evaluate_state(asia).confidence
    conf_london = evaluate_state(london).confidence
    conf_ny = evaluate_state(ny).confidence

    assert conf_asia < conf_london
    assert conf_ny > conf_london


def test_regime_engine_session_output():
    regime = RegimeEngine(window=3)
    vstate = {"vol_regime": "NORMAL", "realized_vol": 0.1}
    lstate = {"liquidity_resilience": 0.1, "depth_imbalance": 0.0}
    mstate = {"hawkishness": 0.0, "risk_sentiment": 0.0}
    out = regime.compute(
        vstate, lstate, mstate, session_regime=SessionRegime.ASIA.value
    )
    assert out["session_regime"] == SessionRegime.ASIA.value
