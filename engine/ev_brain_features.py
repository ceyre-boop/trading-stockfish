import hashlib
from typing import Iterable, List

import numpy as np
import pandas as pd

# Deterministic feature ordering (excluding outcome labels)
FEATURE_COLUMNS: List[str] = [
    # state
    "state_market_profile_state",
    "state_market_profile_confidence",
    "state_session_profile",
    "state_session_profile_confidence",
    "state_vol_regime",
    "state_trend_regime",
    "state_liquidity_bias",
    "state_condition_vector",
    # action
    "action_type_id",
    "entry_model_id_idx",
    "direction_id",
    "size_bucket_id",
    "stop_structure_json",
    "tp_structure_json",
    "manage_payload_json",
]


def _stable_hash(value: str) -> float:
    # Deterministic hash to float in [0, 1)
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    intval = int(digest, 16)
    return (intval % 10**9) / float(10**9)


def _encode_value(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return 0.0
    # Everything else -> stable hash of string representation
    return _stable_hash(str(val))


def build_feature_matrix(dataset: pd.DataFrame | Iterable[dict]) -> np.ndarray:
    """Convert EV dataset rows into a numeric matrix for the EV brain.

    Accepts a DataFrame or iterable of dicts. Outcome label columns are ignored; only
    FEATURE_COLUMNS are used, in deterministic order.
    """

    if isinstance(dataset, pd.DataFrame):
        rows = dataset.to_dict(orient="records")
    else:
        rows = list(dataset)

    vectors: List[List[float]] = []
    for row in rows:
        vector: List[float] = []
        for col in FEATURE_COLUMNS:
            vector.append(_encode_value(row.get(col)))
        vectors.append(vector)

    return np.asarray(vectors, dtype=np.float32)
