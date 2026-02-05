import numpy as np
import pandas as pd

from .ev_brain_features import build_feature_matrix
from .ev_brain_model import EVBrainV1


def train_ev_brain(dataset: pd.DataFrame, *, version: str = "v1") -> EVBrainV1:
    if dataset is None or dataset.empty:
        raise ValueError("Dataset is empty; cannot train EV brain")

    if "label_realized_R" not in dataset.columns:
        raise ValueError("Dataset missing label_realized_R")

    X = build_feature_matrix(dataset)
    y = dataset["label_realized_R"].to_numpy(dtype=np.float32)

    model = EVBrainV1(version=version)
    model.fit(X, y)
    return model
