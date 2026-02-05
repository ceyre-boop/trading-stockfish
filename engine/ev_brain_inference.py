from typing import Dict, List

import numpy as np
import pandas as pd

from .ev_brain_features import build_feature_matrix
from .ev_brain_model import EVBrainV1


def evaluate_actions(ev_brain: EVBrainV1, feature_rows: List[Dict]) -> List[float]:
    if ev_brain is None:
        raise ValueError("ev_brain is required")
    if not feature_rows:
        return []

    df = pd.DataFrame(feature_rows)
    X = build_feature_matrix(df)
    preds = ev_brain.predict(X)
    return preds.tolist()
