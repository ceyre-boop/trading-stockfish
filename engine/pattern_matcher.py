from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .decision_frame import DecisionFrame
from .pattern_templates import (
    PATTERN_TEMPLATES,
    PatternFamily,
    PatternTemplate,
    load_default_templates,
)


@dataclass(frozen=True)
class PatternTemplateMatch:
    template_id: str
    eco_code: str
    family: PatternFamily
    match_score: float
    anchors: Dict
    template: Optional[PatternTemplate] = None


class PatternMatcher:
    def __init__(self, templates: Dict[str, PatternTemplate] | None = None) -> None:
        if templates is None and not PATTERN_TEMPLATES:
            load_default_templates()
        self.templates = templates or PATTERN_TEMPLATES
        self.allow_empty_preconditions = templates is not None

    def match_templates(self, frame: DecisionFrame) -> List[PatternTemplateMatch]:
        matches: List[PatternTemplateMatch] = []

        for template_id in sorted(self.templates.keys()):
            template = self.templates[template_id]
            if not _preconditions_satisfied(
                template, frame, allow_empty=self.allow_empty_preconditions
            ):
                continue
            if not _structure_satisfied(template, frame):
                continue

            anchors = _extract_anchors(frame)
            match = PatternTemplateMatch(
                template_id=template.id,
                eco_code=template.eco_code,
                family=template.family,
                match_score=1.0,
                anchors=anchors,
                template=template,
            )
            matches.append(match)

        return matches


def _preconditions_satisfied(
    template: PatternTemplate, frame: DecisionFrame, *, allow_empty: bool
) -> bool:
    pre = template.preconditions or {}

    if not pre:
        return allow_empty

    regime = pre.get("regime")
    frame_regime = None
    if isinstance(getattr(frame, "condition_vector", None), dict):
        frame_regime = frame.condition_vector.get("regime")
    if frame_regime is None:
        frame_regime = getattr(frame, "vol_regime", None)
    if regime is not None and frame_regime != regime:
        return False

    session = pre.get("session")
    if session is not None and getattr(frame, "session_profile", None) != session:
        return False

    vol_regime = pre.get("vol_regime")
    frame_vol = getattr(frame, "vol_regime", None)
    if vol_regime is not None and frame_vol != vol_regime:
        return False

    vol = pre.get("vol")
    if vol is not None and frame_vol != vol:
        return False

    trend_regime = pre.get("trend") or pre.get("trend_regime")
    if (
        trend_regime is not None
        and getattr(frame, "trend_regime", None) != trend_regime
    ):
        return False

    liquidity = pre.get("liquidity")
    liquidity_state = None
    if isinstance(frame.liquidity_frame, dict):
        liquidity_state = frame.liquidity_frame.get(
            "state"
        ) or frame.liquidity_frame.get("liquidity")
    if liquidity is not None and liquidity_state != liquidity:
        return False

    return True


def _structure_satisfied(template: PatternTemplate, frame: DecisionFrame) -> bool:
    ps = template.pattern_structure or {}
    structure = (
        frame.entry_consistency_report
        or frame.entry_brain_scores
        or frame.condition_vector
        or {}
    )
    if not isinstance(structure, dict):
        structure = {}

    def _bool_feature(key: str) -> bool:
        val = structure.get(key)
        return bool(val)

    sweep_req = ps.get("sweep")
    if sweep_req is True and not _bool_feature("sweep_present"):
        return False
    if isinstance(sweep_req, str) and structure.get("sweep_type") != sweep_req:
        return False

    ob_req = ps.get("orderblock") or ps.get("ob")
    if ob_req is True and not _bool_feature("ob_present"):
        return False

    fvg_req = ps.get("fvg")
    if fvg_req is True and not _bool_feature("fvg_present"):
        return False
    if isinstance(fvg_req, str) and structure.get("fvg_type") != fvg_req:
        return False

    displacement_req = ps.get("displacement")
    if displacement_req:
        direction = structure.get("displacement") or structure.get("impulse_direction")
        if displacement_req != direction:
            return False

    return True


def _extract_anchors(frame: DecisionFrame) -> Dict:
    anchors: Dict = {}
    structure = (
        frame.entry_consistency_report
        or frame.entry_brain_scores
        or frame.condition_vector
        or {}
    )
    if isinstance(structure, dict):
        for key in ("sweep_levels", "ob_bounds", "fvg_bounds", "displacement"):
            if key in structure:
                anchors[key] = structure[key]
    return anchors
