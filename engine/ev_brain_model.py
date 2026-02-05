from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class _MLPWeights:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def to_json(self) -> str:
        payload = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }
        return json.dumps(payload)

    @staticmethod
    def from_json(raw: str) -> "_MLPWeights":
        data = json.loads(raw)
        return _MLPWeights(
            W1=np.asarray(data["W1"], dtype=np.float32),
            b1=np.asarray(data["b1"], dtype=np.float32),
            W2=np.asarray(data["W2"], dtype=np.float32),
            b2=np.asarray(data["b2"], dtype=np.float32),
        )


class EVBrainV1:
    def __init__(
        self, model: Optional[_MLPWeights] = None, version: str = "v1", seed: int = 42
    ):
        self.version = version
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.model = model

    def _init_weights(self, input_dim: int, hidden_dim: int = 32) -> None:
        # Xavier uniform initialization for stability
        limit1 = np.sqrt(6 / (input_dim + hidden_dim))
        W1 = self.rng.uniform(-limit1, limit1, size=(input_dim, hidden_dim)).astype(
            np.float32
        )
        b1 = np.zeros(hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(6 / (hidden_dim + 1))
        W2 = self.rng.uniform(-limit2, limit2, size=(hidden_dim, 1)).astype(np.float32)
        b2 = np.zeros(1, dtype=np.float32)

        self.model = _MLPWeights(W1=W1, b1=b1, W2=W2, b2=b2)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _huber_gradient(residual: np.ndarray, delta: float = 1.0) -> np.ndarray:
        grad = np.where(np.abs(residual) <= delta, residual, delta * np.sign(residual))
        return grad

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        hidden_dim: int = 32,
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1:
            y = y.reshape(-1)
        n_samples, input_dim = X.shape
        if self.model is None:
            self._init_weights(input_dim, hidden_dim)

        # Full-batch gradient descent for determinism
        for _ in range(epochs):
            h = self._relu(X @ self.model.W1 + self.model.b1)  # (n, hidden)
            preds = h @ self.model.W2 + self.model.b2  # (n, 1)
            preds = preds.reshape(-1)
            residual = preds - y
            grad_out = self._huber_gradient(residual) / n_samples  # (n,)

            # Gradients for W2 and b2
            grad_W2 = h.T @ grad_out.reshape(-1, 1)
            grad_b2 = np.sum(grad_out)

            # Backprop through ReLU
            grad_h = grad_out.reshape(-1, 1) @ self.model.W2.T  # (n, hidden)
            grad_h[h <= 0] = 0.0

            grad_W1 = X.T @ grad_h
            grad_b1 = np.sum(grad_h, axis=0)

            self.model.W2 -= lr * grad_W2.astype(np.float32)
            self.model.b2 -= lr * grad_b2.astype(np.float32)
            self.model.W1 -= lr * grad_W1.astype(np.float32)
            self.model.b1 -= lr * grad_b1.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not initialized or trained")
        h = self._relu(X @ self.model.W1 + self.model.b1)
        preds = h @ self.model.W2 + self.model.b2
        return preds.reshape(-1)

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("No model weights to save")
        payload = {
            "version": self.version,
            "seed": self.seed,
            "weights": json.loads(self.model.to_json()),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "EVBrainV1":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = _MLPWeights(
            W1=np.asarray(raw["weights"]["W1"], dtype=np.float32),
            b1=np.asarray(raw["weights"]["b1"], dtype=np.float32),
            W2=np.asarray(raw["weights"]["W2"], dtype=np.float32),
            b2=np.asarray(raw["weights"]["b2"], dtype=np.float32),
        )
        model = cls(
            model=weights, version=raw.get("version", "v1"), seed=raw.get("seed", 42)
        )
        return model
