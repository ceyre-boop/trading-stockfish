from datetime import datetime

from engine.decision_frame import DecisionFrame
from engine.pattern_matcher import PatternMatcher, PatternTemplateMatch
from engine.pattern_templates import PATTERN_TEMPLATES, PatternFamily, PatternTemplate


def _frame(
    regime="A",
    session="1A",
    trend="up",
    vol="normal",
    liquidity_state="normal",
    structure=None,
):
    return DecisionFrame(
        timestamp_utc=datetime.utcnow().isoformat(),
        session_profile=session,
        vol_regime=vol,
        trend_regime=trend,
        liquidity_frame={"state": liquidity_state},
        condition_vector={"regime": regime, "vol": vol},
        entry_consistency_report=structure or {},
    )


def test_no_match_when_preconditions_fail():
    matcher = PatternMatcher()
    frame = _frame(regime="Z", session="X")
    assert matcher.match_templates(frame) == []


def test_match_liquidity_template_on_structure_and_preconditions():
    matcher = PatternMatcher()
    structure = {
        "sweep_present": True,
        "sweep_type": "sell_to_buy",
        "ob_present": True,
        "ob_type": "bullish",
        "displacement": "up",
        "sweep_levels": {"low": 100},
    }
    frame = _frame(
        regime="A",
        session="NY_OPEN",
        trend="range",
        vol="moderate",
        liquidity_state="normal",
        structure=structure,
    )

    matches = matcher.match_templates(frame)
    ids = [m.template_id for m in matches]
    assert "A10_LIQ_SWEEP_REVERSAL" in ids
    for m in matches:
        assert m.match_score == 1.0
        assert isinstance(m, PatternTemplateMatch)


def test_match_requires_structure_flags():
    matcher = PatternMatcher()
    # missing sweep_present should fail A10
    frame = _frame(
        regime="A",
        session="NY_OPEN",
        trend="range",
        vol="moderate",
        liquidity_state="normal",
        structure={},
    )
    matches = matcher.match_templates(frame)
    assert "A10_LIQ_SWEEP_REVERSAL" not in [m.template_id for m in matches]

    # supply sweep present to pass
    frame2 = _frame(
        regime="A",
        session="NY_OPEN",
        trend="range",
        vol="moderate",
        liquidity_state="normal",
        structure={
            "sweep_present": True,
            "sweep_type": "sell_to_buy",
            "ob_present": True,
            "displacement": "up",
        },
    )
    matches2 = matcher.match_templates(frame2)
    assert "A10_LIQ_SWEEP_REVERSAL" in [m.template_id for m in matches2]


def test_displacement_direction_must_match():
    matcher = PatternMatcher()
    structure = {"fvg_present": True, "displacement": "down"}
    frame = _frame(
        regime="A",
        session="1A",
        trend="up",
        vol="normal",
        liquidity_state="normal",
        structure=structure,
    )
    matches = matcher.match_templates(frame)
    assert "A20_LIQ_BUYBACK_CONTINUATION" not in [m.template_id for m in matches]

    structure2 = {
        "fvg_present": True,
        "fvg_type": "bullish",
        "sweep_present": True,
        "sweep_type": "sell_to_buy",
        "ob_present": True,
        "ob_type": "bullish",
        "displacement": "up",
    }
    frame2 = _frame(
        regime="A",
        session="1A",
        trend="up",
        vol="normal",
        liquidity_state="normal",
        structure=structure2,
    )
    matches2 = matcher.match_templates(frame2)
    assert "A20_LIQ_BUYBACK_CONTINUATION" in [m.template_id for m in matches2]


def test_anchors_extracted_deterministically():
    matcher = PatternMatcher()
    structure = {
        "sweep_present": True,
        "sweep_type": "sell_to_buy",
        "sweep_levels": {"low": 99.5, "high": 101.0},
        "ob_present": True,
        "ob_bounds": {"low": 100.1, "high": 100.6},
        "fvg_present": True,
        "fvg_type": "bullish",
        "fvg_bounds": {"low": 100.2, "high": 100.4},
        "displacement": "up",
    }
    frame = _frame(
        regime="A",
        session="1A",
        trend="up",
        vol="normal",
        liquidity_state="normal",
        structure=structure,
    )
    matches = matcher.match_templates(frame)
    assert matches
    anchors = matches[0].anchors
    assert anchors["sweep_levels"] == {"low": 99.5, "high": 101.0}
    assert anchors["ob_bounds"] == {"low": 100.1, "high": 100.6}
    assert anchors["fvg_bounds"] == {"low": 100.2, "high": 100.4}
    assert anchors["displacement"] == "up"


def test_wrong_liquidity_state_blocks_match():
    matcher = PatternMatcher()
    structure = {"sweep_present": True}
    frame = _frame(
        regime="A",
        session="NY_OPEN",
        trend="range",
        vol="moderate",
        liquidity_state="thin",
        structure=structure,
    )
    matches = matcher.match_templates(frame)
    assert "A10_LIQ_SWEEP_REVERSAL" not in [m.template_id for m in matches]


def test_missing_required_flags_blocks():
    matcher = PatternMatcher()
    structure = {"fvg_present": False}
    frame = _frame(
        regime="A",
        session="1A",
        trend="up",
        vol="normal",
        liquidity_state="normal",
        structure=structure,
    )
    matches = matcher.match_templates(frame)
    assert "A20_LIQ_BUYBACK_CONTINUATION" not in [m.template_id for m in matches]


def test_deterministic_ordering_of_matches():
    matcher = PatternMatcher()
    structure = {"sweep_present": True, "fvg_present": True, "displacement": "up"}
    frame = _frame(
        regime="A",
        session="1A",
        trend="up",
        vol="normal",
        liquidity_state="normal",
        structure=structure,
    )
    matches = matcher.match_templates(frame)
    ids = [m.template_id for m in matches]
    assert ids == sorted(ids)
