import json
from pathlib import Path

import pytest

from analytics.policy_builder import build_policy_from_stats
from engine.evaluator import evaluate_with_causal
from engine.policy_loader import PolicyConfig
from engine.regime_multipliers import (
    MultiplierConfig,
    StatsResult,
    compute_regime_multipliers,
)


def test_compute_regime_multipliers_respects_caps():
    stats = StatsResult(
        feature_performance_by_regime={
            "feature_hot": {"HIGH_VOL": 10.0},
            "feature_cold": {"LOW_VOL": -10.0},
            "feature_mid": {"NORMAL": 0.25},
        }
    )
    cfg = MultiplierConfig(min_multiplier=0.6, max_multiplier=1.5, smoothing=0.4)

    mults = compute_regime_multipliers(stats, cfg)

    assert mults["HIGH_VOL"]["feature_hot"] == pytest.approx(cfg.max_multiplier)
    assert mults["LOW_VOL"]["feature_cold"] == pytest.approx(cfg.min_multiplier)
    assert cfg.min_multiplier <= mults["NORMAL"]["feature_mid"] <= cfg.max_multiplier

    for regime_map in mults.values():
        for val in regime_map.values():
            assert cfg.min_multiplier <= val <= cfg.max_multiplier

    assert mults["HIGH_VOL"]["feature_hot"] <= cfg.max_multiplier
    assert mults["LOW_VOL"]["feature_cold"] >= cfg.min_multiplier


def test_policy_builder_includes_regime_multipliers(tmp_path: Path, monkeypatch):
    class DummySpec:
        def __init__(self):
            self.role = ["signal"]
            self.tags = ["test"]
            self.live = True

    class DummyRegistry:
        def __init__(self):
            self.specs = {"feature_a": DummySpec(), "feature_b": DummySpec()}
            self.version = "test-registry"

    registry = DummyRegistry()
    monkeypatch.setattr("analytics.policy_builder.load_registry", lambda path: registry)

    stats = StatsResult(
        feature_importance={"feature_a": 2.0},
        feature_stability={"feature_a": 0.9},
        feature_performance_by_regime={
            "feature_a": {"HIGH_VOL": 1.0},
            "feature_b": {"LOW_VOL": -1.0},
        },
    )

    out_path = tmp_path / "policy.json"
    policy = build_policy_from_stats(stats, config={"out_path": out_path})

    expected_rm = compute_regime_multipliers(stats)
    assert policy["regime_multipliers"] == expected_rm

    for feature in registry.specs.keys():
        assert feature in policy["features"], "registry feature missing from policy"
        expected_feature_rm = {
            regime: fmap[feature]
            for regime, fmap in expected_rm.items()
            if feature in fmap
        }
        assert policy["features"][feature]["regime_multipliers"] == expected_feature_rm

    saved_policy = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved_policy["regime_multipliers"] == expected_rm


def test_evaluator_applies_regime_multipliers():
    class DummyFactor:
        def __init__(self):
            self.factor_name = "feature_a"
            self.score = 1.0
            self.weight = 1.0
            self.explanation = ""

    class DummyResult:
        def __init__(self):
            self.eval_score = 1.0
            self.confidence = 1.0
            self.scoring_factors = [DummyFactor()]
            self.timestamp = "2026-02-02T00:00:00Z"

    class DummyCausalEvaluator:
        def evaluate(self, market_state):
            return DummyResult()

    class DummyMarketState:
        def __init__(self):
            self.session = "HIGH_VOL"

    policy_dict = {
        "base_weights": {"feature_a": 1.0},
        "trust": {"feature_a": 1.0},
        "regime_multipliers": {"HIGH_VOL": {"feature_a": 2.0}},
    }
    policy = PolicyConfig(policy_dict)

    result = evaluate_with_causal(
        state={"features": {}},
        causal_evaluator=DummyCausalEvaluator(),
        market_state=DummyMarketState(),
        policy=policy,
    )

    factors = result.get("details", {}).get("factors", [])
    assert factors, "Expected factor breakdown present"
    assert factors[0]["policy_weight"] == pytest.approx(2.0)
