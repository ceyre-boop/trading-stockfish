import logging
import math

import pytest

from config.feature_registry import FeatureSpec
from engine.market_state_builder import (
    FeatureAudit,
    _apply_encoding,
    _apply_transform,
    _feature_raw_value,
    _resolve_with_trace,
)


@pytest.fixture
def spec_factory():
    def _make(**overrides):
        base = {
            "name": "feature",
            "type": "scalar",
            "source": "test",
            "description": "",
            "live": True,
            "shape": "scalar",
            "role": [],
            "transform": {"kind": "none", "params": {}},
            "encoding": {"kind": "none", "params": {}},
            "dependencies": [],
            "tags": [],
            "alias": overrides.get("alias", overrides.get("name", "feature")),
            "constraints": {},
        }
        base.update(overrides)
        if "alias" not in overrides:
            base["alias"] = base["name"]
        return FeatureSpec(**base)

    return _make


@pytest.fixture
def state_base():
    return {}


def test_simple_scalar_resolution(spec_factory, state_base):
    state = {"price": 1.2}
    spec = spec_factory(name="price", alias="price")
    val = _feature_raw_value(spec, state)
    assert val == 1.2


def test_nested_resolution(spec_factory):
    state = {"macro": {"score": 0.7}}
    spec = spec_factory(name="macro_score", alias="macro.score")
    val = _feature_raw_value(spec, state)
    assert val == 0.7


def test_missing_intermediate_warns(caplog, spec_factory):
    state = {"macro": {}}
    spec = spec_factory(name="macro_score", alias="macro.score.value")
    with caplog.at_level(logging.WARNING):
        val = _feature_raw_value(spec, state)
    assert val is None
    assert any("missing segment: score" in m for m in caplog.messages)


def test_missing_final_warns(caplog, spec_factory):
    state = {"macro": {"score": {}}}
    spec = spec_factory(name="macro_score", alias="macro.score.value")
    with caplog.at_level(logging.WARNING):
        val = _feature_raw_value(spec, state)
    assert val is None
    assert any("missing segment: value" in m for m in caplog.messages)


def test_list_index_resolution(spec_factory):
    state = {"events": [{"type": "GDP"}]}
    spec = spec_factory(name="event_type", alias="events.0.type")
    val = _feature_raw_value(spec, state)
    assert val == "GDP"


def test_dict_inside_list_resolution(spec_factory):
    state = {"events": [{"meta": {"kind": "NFP"}}]}
    spec = spec_factory(name="event_kind", alias="events.0.meta.kind")
    val = _feature_raw_value(spec, state)
    assert val == "NFP"


def test_type_mismatch_warns(caplog, spec_factory):
    state = {"category": 1.23}
    spec = spec_factory(
        name="cat", alias="category", constraints={"allowed_values": ["A", "B"]}
    )
    with caplog.at_level(logging.WARNING):
        val = _feature_raw_value(spec, state)
    assert val is None
    assert any("Type mismatch" in m for m in caplog.messages)


def test_transform_and_encoding(spec_factory):
    # minmax
    spec_minmax = spec_factory(
        name="minmax",
        alias="x",
        transform={"kind": "minmax", "params": {"min": 0, "max": 10}},
    )
    val_minmax = _apply_transform(5, spec_minmax.transform)
    assert val_minmax == pytest.approx(0.5)

    # log handles zero/negative gracefully
    spec_log = spec_factory(
        name="logv", alias="x", transform={"kind": "log", "params": {}}
    )
    val_log = _apply_transform(-1, spec_log.transform)
    assert val_log == pytest.approx(math.log(1e-9))

    # zscore passthrough
    spec_z = spec_factory(
        name="zscore", alias="x", transform={"kind": "zscore", "params": {}}
    )
    val_z = _apply_transform(2.0, spec_z.transform)
    assert val_z == 2.0

    # one_hot
    spec_one_hot = spec_factory(
        name="onehot", alias="x", encoding={"kind": "one_hot", "params": {}}
    )
    encoded = _apply_encoding("BUY", spec_one_hot.encoding)
    assert encoded == {"BUY": 1}


def test_allowed_values_miss_flag_and_categorical(caplog, spec_factory):
    # Flag
    state_flag = {"event": "Retail Sales"}
    spec_flag = spec_factory(
        name="flag",
        alias="event",
        type="boolean",
        constraints={"allowed_values": ["GDP", "PMI"]},
    )
    with caplog.at_level(logging.WARNING):
        val_flag = _feature_raw_value(spec_flag, state_flag)
    assert val_flag is False
    assert any("Value outside allowed set" in m for m in caplog.messages)

    caplog.clear()

    # Categorical
    state_cat = {"event": "Retail Sales"}
    spec_cat = spec_factory(
        name="cat",
        alias="event",
        type="categorical",
        constraints={"allowed_values": ["GDP", "PMI"]},
    )
    with caplog.at_level(logging.WARNING):
        val_cat = _feature_raw_value(spec_cat, state_cat)
    assert val_cat is None
    assert any("Value outside allowed set" in m for m in caplog.messages)


def test_growth_event_ontology(caplog, spec_factory):
    spec = spec_factory(
        name="growth_event_flag",
        alias="event_type",
        type="boolean",
        constraints={"allowed_values": ["GDP", "PMI", "ISM"]},
    )

    # Case 1: valid
    state_valid = {"event_type": "GDP"}
    with caplog.at_level(logging.WARNING):
        assert _feature_raw_value(spec, state_valid) is True
    assert not caplog.messages

    # Case 2: invalid
    caplog.clear()
    state_invalid = {"event_type": "Retail Sales"}
    with caplog.at_level(logging.WARNING):
        assert _feature_raw_value(spec, state_invalid) is False
    assert any("Growth event not allowed" in m for m in caplog.messages)

    # Case 3: missing
    caplog.clear()
    state_missing = {}
    with caplog.at_level(logging.WARNING):
        assert _feature_raw_value(spec, state_missing) is False
    assert any("Missing alias path" in m for m in caplog.messages)

    # Case 4: list
    caplog.clear()
    state_list = {"event_type": ["GDP", "Inflation"]}
    with caplog.at_level(logging.WARNING):
        assert _feature_raw_value(spec, state_list) is True
    assert not caplog.messages


def test_audit_records_missing_and_constraint(caplog, spec_factory):
    audit = FeatureAudit()
    spec = spec_factory(
        name="growth_event_flag",
        alias="event_type",
        type="boolean",
        constraints={"allowed_values": ["GDP", "PMI"]},
    )
    state = {}
    with caplog.at_level(logging.WARNING):
        val = _feature_raw_value(spec, state, audit=audit)
    assert val is False
    out = audit.to_dict()
    assert out["summary"]["missing_alias"] == 1
    assert len(out["issues"]) == 1
    assert out["issues"][0]["feature"] == "growth_event_flag"

    # constraint violation path
    caplog.clear()
    audit2 = FeatureAudit()
    state2 = {"event_type": "Retail"}
    with caplog.at_level(logging.WARNING):
        val2 = _feature_raw_value(spec, state2, audit=audit2)
    assert val2 is False
    out2 = audit2.to_dict()
    assert out2["summary"]["constraint_violation"] == 1
    assert out2["issues"][0]["value"] == "Retail"


def test_audit_pass_case_no_violation(spec_factory):
    state = {"x": "GDP"}
    spec = spec_factory(
        name="growth_event_flag",
        alias="x",
        type="boolean",
        constraints={"allowed_values": ["GDP"]},
    )
    audit = FeatureAudit()
    val = _feature_raw_value(spec, state, audit=audit)
    assert val is True
    assert audit.to_dict()["summary"]["constraint_violation"] == 0


def test_resolver_traces_missing_segment():
    state = {"a": {"b": {}}}
    val, missing = _resolve_with_trace(state, "a.b.c")
    assert val is None
    assert missing == "c"


def test_resolver_handles_list_index():
    state = {"events": ["a", "b"]}
    val, missing = _resolve_with_trace(state, "events.1")
    assert val == "b"
    assert missing is None

    val2, missing2 = _resolve_with_trace(state, "events.5")
    assert val2 is None
    assert missing2 == "5"
