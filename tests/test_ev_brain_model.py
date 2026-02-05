import numpy as np
import pandas as pd

from engine.ev_brain_features import FEATURE_COLUMNS, build_feature_matrix
from engine.ev_brain_inference import evaluate_actions
from engine.ev_brain_model import EVBrainV1
from engine.ev_brain_training import train_ev_brain


def _synthetic_dataset():
    rows = [
        {
            "decision_id": "d1",
            "state_market_profile_state": "ACCUMULATION",
            "state_market_profile_confidence": 0.6,
            "state_session_profile": "PROFILE_1A",
            "state_session_profile_confidence": 0.5,
            "state_vol_regime": "NORMAL",
            "state_trend_regime": "UP",
            "state_liquidity_bias": "UP",
            "state_condition_vector": {"session": "RTH", "vol": "NORMAL"},
            "action_type_id": 1,
            "entry_model_id_idx": 2,
            "direction_id": 1,
            "size_bucket_id": 1,
            "stop_structure_json": '{"type":"ATR","mult":2}',
            "tp_structure_json": '{"rr":2.5}',
            "manage_payload_json": None,
            "label_realized_R": 1.0,
        },
        {
            "decision_id": "d2",
            "state_market_profile_state": "DISTRIBUTION",
            "state_market_profile_confidence": 0.4,
            "state_session_profile": "PROFILE_1C",
            "state_session_profile_confidence": 0.3,
            "state_vol_regime": "HIGH",
            "state_trend_regime": "DOWN",
            "state_liquidity_bias": "DOWN",
            "state_condition_vector": {"session": "RTH", "vol": "HIGH"},
            "action_type_id": 2,
            "entry_model_id_idx": 1,
            "direction_id": -1,
            "size_bucket_id": 2,
            "stop_structure_json": '{"type":"ATR","mult":3}',
            "tp_structure_json": '{"rr":1.5}',
            "manage_payload_json": None,
            "label_realized_R": -0.5,
        },
    ]
    # Fill missing required feature columns with None
    for row in rows:
        for col in FEATURE_COLUMNS:
            row.setdefault(col, None)
    return pd.DataFrame(rows)


def test_feature_pipeline_is_deterministic():
    df = _synthetic_dataset().head(1)
    vec1 = build_feature_matrix(df)
    vec2 = build_feature_matrix(df)
    assert np.array_equal(vec1, vec2)
    assert vec1.shape[1] == len(FEATURE_COLUMNS)


def test_ev_brain_trains_and_predicts_deterministically(tmp_path):
    df = _synthetic_dataset()
    model = train_ev_brain(df, version="v1")

    X = build_feature_matrix(df)
    preds1 = model.predict(X)
    preds2 = model.predict(X)
    assert np.allclose(preds1, preds2)

    # Save and load preserves predictions
    model_path = tmp_path / "ev_brain.json"
    model.save(model_path)
    loaded = EVBrainV1.load(model_path)
    preds_loaded = loaded.predict(X)
    assert np.allclose(preds1, preds_loaded)


def test_inference_wrapper_returns_list():
    df = _synthetic_dataset()
    model = train_ev_brain(df, version="v1")

    feature_rows = df.drop(columns=["label_realized_R"]).to_dict(orient="records")
    scores = evaluate_actions(model, feature_rows)

    assert isinstance(scores, list)
    assert len(scores) == len(feature_rows)
