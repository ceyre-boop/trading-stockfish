import json
import logging
import os
from pathlib import Path

import pytest

import engine.evaluator as evaluator


def _write_policy(tmp_path: Path) -> Path:
    policy_rows = [
        {
            "strategy_id": "s1",
            "entry_model_id": "e1",
            "session": "RTH",
            "macro": "RISK_ON",
            "vol": "HIGH",
            "trend": "UP",
            "liquidity": "DEEP",
            "tod": "OPEN",
            "prob_good": 0.9,
            "expected_reward": 1.2,
            "sample_size": 10,
            "label": "PREFERRED",
        }
    ]
    payload = {"metadata": {}, "policy": policy_rows}
    path = tmp_path / "brain_policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prepare_env(tmp_path: Path) -> None:
    evaluator._BRAIN_POLICY_CACHE.clear()
    os.environ["BRAIN_POLICY_PATH"] = str(tmp_path / "brain_policy.json")


def test_shadow_brain_attaches_recommendations(tmp_path: Path):
    policy_path = _write_policy(tmp_path)
    _prepare_env(tmp_path)

    # Ensure policy is loaded
    df = evaluator.load_brain_policy(policy_path)
    assert not df.empty

    result_payload = {
        "decision": "buy",
        "brain_label": None,
        "brain_prob_good": None,
        "brain_expected_reward": None,
        "brain_sample_size": None,
    }
    entry = {
        "strategy_id": "s1",
        "entry_model_id": "e1",
        "condition_vector": {
            "session": "RTH",
            "macro": "RISK_ON",
            "vol": "HIGH",
            "trend": "UP",
            "liquidity": "DEEP",
            "tod": "OPEN",
        },
    }

    evaluator._apply_shadow_brain_annotations(
        strategy_id="s1",
        entry_model_id="e1",
        condition_vector=entry["condition_vector"],
        result_payload=result_payload,
        entry=entry,
    )

    assert result_payload["brain_label"] == "PREFERRED"
    assert result_payload["brain_prob_good"] == pytest.approx(0.9)
    assert result_payload["brain_expected_reward"] == pytest.approx(1.2)
    assert result_payload["brain_sample_size"] == 10
    assert entry["brain_label"] == "PREFERRED"


def test_shadow_brain_does_not_change_actions(tmp_path: Path):
    _write_policy(tmp_path)
    _prepare_env(tmp_path)

    result_payload = {"decision": "sell"}
    entry = {
        "strategy_id": "s1",
        "entry_model_id": "e1",
        "condition_vector": {
            "session": "RTH",
            "macro": "RISK_ON",
            "vol": "HIGH",
            "trend": "UP",
            "liquidity": "DEEP",
            "tod": "OPEN",
        },
    }

    evaluator._apply_shadow_brain_annotations(
        strategy_id="s1",
        entry_model_id="e1",
        condition_vector=entry["condition_vector"],
        result_payload=result_payload,
        entry=entry,
    )

    assert result_payload["decision"] == "sell"


def test_shadow_brain_logs_recommendations(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    _write_policy(tmp_path)
    _prepare_env(tmp_path)

    caplog.set_level(logging.DEBUG, logger="engine.evaluator")

    result_payload = {"decision": "hold"}
    entry = {
        "strategy_id": "s1",
        "entry_model_id": "e1",
        "condition_vector": {
            "session": "RTH",
            "macro": "RISK_ON",
            "vol": "HIGH",
            "trend": "UP",
            "liquidity": "DEEP",
            "tod": "OPEN",
        },
    }

    evaluator._apply_shadow_brain_annotations(
        strategy_id="s1",
        entry_model_id="e1",
        condition_vector=entry["condition_vector"],
        result_payload=result_payload,
        entry=entry,
    )

    logs = "\n".join(caplog.messages)
    assert "Shadow brain" in logs
    assert "label=PREFERRED" in logs
