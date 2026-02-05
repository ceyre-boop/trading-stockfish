from engine.decision_actions import ActionType, DecisionAction
from engine.decision_frame import DecisionFrame
from engine.search_scoring import score_actions_via_search


class StubBrain:
    def predict(self, X):
        # deterministic ascending scores based on row order
        import numpy as np

        return np.arange(X.shape[0], dtype=float)


class StubPolicy:
    def lookup(self, entry_id, frame):
        return "ALLOWED"

    def multiplier_for(self, label):
        return 0.5 if label == "ALLOWED" else 0.0


class StubMCR:
    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, *args, **kwargs):
        return self.metrics


def test_score_actions_via_search_deterministic(monkeypatch):
    frame = DecisionFrame()
    actions = [
        DecisionAction(action_type=ActionType.NO_TRADE),
        DecisionAction(
            action_type=ActionType.OPEN_LONG, entry_model_id="ENTRY_A", direction="LONG"
        ),
    ]

    # Stub MCR to avoid randomness
    from engine import search_scoring

    metrics = {
        "mean_EV": 1.0,
        "variance_EV": 0.1,
        "tail_risk": -0.2,
        "stop_hit_rate": 0.3,
        "tp_hit_rate": 0.6,
        "avg_time_in_trade": 5.0,
    }
    monkeypatch.setattr(
        search_scoring, "evaluate_action_via_mcr", lambda *a, **k: metrics
    )

    scores1 = score_actions_via_search(
        frame,
        position_state={},
        candidate_actions=actions,
        ev_brain=StubBrain(),
        brain_policy=StubPolicy(),
        risk_envelope={},
        n_paths=2,
        horizon_bars=3,
        seed=11,
    )
    scores2 = score_actions_via_search(
        frame,
        position_state={},
        candidate_actions=actions,
        ev_brain=StubBrain(),
        brain_policy=StubPolicy(),
        risk_envelope={},
        n_paths=2,
        horizon_bars=3,
        seed=11,
    )

    assert scores1 == scores2

    # Check unified score calculation for second action (EV_hat=1 from StubBrain)
    _, score_dict = scores1[1]
    # StubBrain returns 0 for single-row predict, so EV_hat=0 here
    unified_expected = (
        0.0
        + 0.5 * metrics["mean_EV"]
        - 0.2 * metrics["variance_EV"]
        - 0.3 * metrics["tail_risk"]
        + 0.1 * metrics["tp_hit_rate"]
        - 0.1 * metrics["stop_hit_rate"]
        + 0.5  # policy multiplier
    )
    assert abs(score_dict["unified_score"] - unified_expected) < 1e-9

    # NO_TRADE is included
    assert scores1[0][0].action_type == ActionType.NO_TRADE
