import logging

import pytest

import engine.evaluator as evaluator


def _cfg(**overrides):
    cfg = evaluator._load_brain_config()
    cfg.update(overrides)
    return cfg


def _info(label: str, prob: float = 0.9, reward: float = 1.0, sample: int = 50):
    return {
        "brain_label": label,
        "brain_prob_good": prob,
        "brain_expected_reward": reward,
        "brain_sample_size": sample,
    }


def test_brain_disables_strategies():
    cfg = _cfg()
    score, applied = evaluator._apply_brain_weighting(
        score=0.7,
        brain_info=_info("DISABLED"),
        brain_cfg=cfg,
        safe_mode=False,
        risk_limit_hit=False,
        policy_allowed=True,
    )
    assert applied is True
    assert score == 0.0


def test_brain_downweights_discouraged():
    cfg = _cfg(discouraged_penalty=0.5)
    score, applied = evaluator._apply_brain_weighting(
        score=0.8,
        brain_info=_info("DISCOURAGED"),
        brain_cfg=cfg,
        safe_mode=False,
        risk_limit_hit=False,
        policy_allowed=True,
    )
    assert applied is True
    assert score < 0.8
    assert score == pytest.approx(0.4, rel=1e-6)


def test_brain_boosts_preferred():
    cfg = _cfg(preferred_boost=1.2)
    score, applied = evaluator._apply_brain_weighting(
        score=0.5,
        brain_info=_info("PREFERRED"),
        brain_cfg=cfg,
        safe_mode=False,
        risk_limit_hit=False,
        policy_allowed=True,
    )
    assert applied is True
    assert score > 0.5
    assert score == pytest.approx(0.6, rel=1e-6)


def test_brain_respects_SAFE_MODE():
    cfg = _cfg()
    score, applied = evaluator._apply_brain_weighting(
        score=0.5,
        brain_info=_info("PREFERRED"),
        brain_cfg=cfg,
        safe_mode=True,
        risk_limit_hit=False,
        policy_allowed=True,
    )
    assert applied is False
    assert score == 0.5


def test_brain_respects_risk_limits():
    cfg = _cfg()
    score, applied = evaluator._apply_brain_weighting(
        score=0.5,
        brain_info=_info("PREFERRED"),
        brain_cfg=cfg,
        safe_mode=False,
        risk_limit_hit=True,
        policy_allowed=True,
    )
    assert applied is False
    assert score == 0.5


def test_brain_logging_is_deterministic(caplog: pytest.LogCaptureFixture):
    cfg = _cfg()
    caplog.set_level(logging.DEBUG, logger="engine.evaluator")
    evaluator._apply_brain_weighting(
        score=0.5,
        brain_info=_info("PREFERRED"),
        brain_cfg=cfg,
        safe_mode=False,
        risk_limit_hit=False,
        policy_allowed=True,
    )
    logs = "\n".join(caplog.messages)
    assert "Brain influence" in logs
    assert "score_before" in logs
