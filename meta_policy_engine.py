"""
meta_policy_engine.py
Institutional-grade meta-policy layer for Trading Stockfish v3.2
"""

import copy
import math
from typing import Any, Dict, List, Optional


# --- v3.3: Adaptive Weights ---
class MetaPolicyWeights:
    def __init__(
        self,
        w1=1.0,
        w2=1.0,
        w3=1.0,
        w4=1.0,
        w5=1.0,
        min_weight=0.1,
        max_weight=5.0,
        smoothing=0.1,
    ):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.smoothing = smoothing
        self._prev = self.as_dict()

    def as_dict(self):
        return {
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
            "w4": self.w4,
            "w5": self.w5,
        }

    def update(self, deltas: Dict[str, float]):
        for k, delta in deltas.items():
            v = getattr(self, k)
            v_new = v + self.smoothing * delta
            v_new = max(self.min_weight, min(self.max_weight, v_new))
            setattr(self, k, v_new)
        self._prev = self.as_dict()

    def delta(self):
        return {k: self.as_dict()[k] - self._prev[k] for k in self.as_dict()}


# --- v3.3: Performance Tracker ---
class MetaPolicyPerformanceTracker:
    def __init__(self):
        self.metrics = {
            "pnl": 0.0,
            "risk": 0.0,
            "slippage": 0.0,
            "regime_accuracy": 0.0,
            "scenario_accuracy": 0.0,
        }
        self.history = []

    def update(self, new_metrics: Dict[str, float]):
        for k in self.metrics:
            if k in new_metrics:
                self.metrics[k] = new_metrics[k]
        self.history.append(copy.deepcopy(self.metrics))
        if len(self.history) > 100:
            self.history.pop(0)

    def get(self):
        return copy.deepcopy(self.metrics)

    def reset(self):
        for k in self.metrics:
            self.metrics[k] = 0.0
        self.history.clear()


class MetaPolicyEngine:
    def __init__(
        self,
        weights=None,
        enabled=True,
        adaptive_weighting=False,
        performance_tracker=None,
    ):
        # v3.3: weights can be MetaPolicyWeights or dict
        self.weights = weights or MetaPolicyWeights()
        self.enabled = enabled
        self.adaptive_weighting = adaptive_weighting
        self.performance_tracker = performance_tracker or MetaPolicyPerformanceTracker()

    def update_weights(self, performance_metrics: Dict[str, float]):
        if not self.adaptive_weighting:
            return
        # Deterministic, bounded, smoothed update rules
        # Example: if realized risk is high, increase w2 (risk penalty)
        # If regime accuracy is low, decrease w4 (regime fit weight)
        # If scenario accuracy is low, decrease w3 (scenario robustness)
        # If PnL is high, increase w1 (EV)
        # If slippage is high, increase w5 (execution penalty)
        d = {}
        d["w1"] = 0.2 * math.tanh(performance_metrics.get("pnl", 0.0))
        d["w2"] = 0.2 * math.tanh(performance_metrics.get("risk", 0.0))
        d["w3"] = -0.2 * math.tanh(
            1.0 - performance_metrics.get("scenario_accuracy", 1.0)
        )
        d["w4"] = -0.2 * math.tanh(
            1.0 - performance_metrics.get("regime_accuracy", 1.0)
        )
        d["w5"] = 0.2 * math.tanh(performance_metrics.get("slippage", 0.0))
        self.weights.update(d)

    def meta_score(self, action: Dict[str, Any]) -> float:
        w = self.weights.as_dict() if hasattr(self.weights, "as_dict") else self.weights
        return (
            w["w1"] * action.get("EV", 0.0)
            - w["w2"] * action.get("Risk", 0.0)
            + w["w3"] * action.get("ScenarioRobustness", 0.0)
            + w["w4"] * action.get("RegimeFit", 0.0)
            - w["w5"] * action.get("ExecutionPenalty", 0.0)
        )

    def select_action(self, candidate_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        diagnostics = {
            "candidate_actions": candidate_actions,
            "meta_scores": [],
            "reason_codes": [],
            "chosen_action": None,
            "current_weights": None,
            "weight_deltas": None,
            "performance_metrics": None,
        }
        if not self.enabled or len(candidate_actions) == 1:
            diagnostics["chosen_action"] = candidate_actions[0]
            diagnostics["meta_scores"] = [None for _ in candidate_actions]
            diagnostics["reason_codes"] = [
                "meta-policy disabled" for _ in candidate_actions
            ]
            diagnostics["current_weights"] = (
                self.weights.as_dict()
                if hasattr(self.weights, "as_dict")
                else self.weights
            )
            diagnostics["weight_deltas"] = {
                k: 0.0 for k in diagnostics["current_weights"]
            }
            diagnostics["performance_metrics"] = None
            return diagnostics
        # --- v3.3: Adaptive Weights ---
        perf = self.performance_tracker.get() if self.performance_tracker else {}
        prev_weights = copy.deepcopy(
            self.weights.as_dict() if hasattr(self.weights, "as_dict") else self.weights
        )
        if self.adaptive_weighting:
            self.update_weights(perf)
        scores = []
        for action in candidate_actions:
            score = self.meta_score(action)
            scores.append(score)
        diagnostics["meta_scores"] = scores
        max_idx = max(range(len(scores)), key=lambda i: scores[i])
        diagnostics["chosen_action"] = candidate_actions[max_idx]
        # Reason codes for each action
        for i, action in enumerate(candidate_actions):
            reasons = []
            if action.get("RegimeFit", 0.0) < 0.2:
                reasons.append("low regime fit")
            if action.get("Risk", 0.0) > 0.8:
                reasons.append("risk penalty")
            if action.get("EV", 0.0) > 0.8:
                reasons.append("high EV")
            if action.get("ScenarioRobustness", 0.0) > 0.8:
                reasons.append("robust scenario")
            if not reasons:
                reasons.append("neutral")
            diagnostics["reason_codes"].append(", ".join(reasons))
        diagnostics["current_weights"] = (
            self.weights.as_dict() if hasattr(self.weights, "as_dict") else self.weights
        )
        diagnostics["weight_deltas"] = {
            k: diagnostics["current_weights"][k] - prev_weights[k]
            for k in diagnostics["current_weights"]
        }
        diagnostics["performance_metrics"] = perf
        return diagnostics


# Singleton instance for integration
meta_policy_engine = MetaPolicyEngine()
