from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass
class PLDAConfig:
    lda_dim: int = 32
    reg_cov: float = 1e-2


class SimplePLDA:
    def __init__(self, cfg: PLDAConfig, num_classes: int) -> None:
        self.cfg = cfg
        self.num_classes = num_classes
        self.lda = LinearDiscriminantAnalysis(n_components=min(cfg.lda_dim, num_classes - 1))
        self.class_means: Dict[int, np.ndarray] = {}
        self.cov = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xr = self.lda.fit_transform(X, y)
        for cls in range(self.num_classes):
            cls_feats = Xr[y == cls]
            if len(cls_feats) == 0:
                continue
            self.class_means[cls] = cls_feats.mean(axis=0)
        centered = np.vstack([self.class_means[c] for c in self.class_means])
        cov = np.cov(centered.T) + self.cfg.reg_cov * np.eye(centered.shape[1])
        self.cov = np.linalg.pinv(cov)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xr = self.lda.transform(X)
        scores = []
        for cls in range(self.num_classes):
            mean = self.class_means.get(cls)
            if mean is None:
                mean = np.zeros(Xr.shape[1])
            diff = Xr - mean
            score = -np.sum(diff @ self.cov * diff, axis=1)
            scores.append(score)
        return np.argmax(np.stack(scores, axis=1), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xr = self.lda.transform(X)
        logits = []
        for cls in range(self.num_classes):
            mean = self.class_means.get(cls, np.zeros(Xr.shape[1]))
            diff = Xr - mean
            logits.append(-np.sum(diff @ self.cov * diff, axis=1))
        logits = np.stack(logits, axis=1)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class ConstantPLDA:
    def __init__(self, class_id: int, num_classes: int) -> None:
        self.class_id = class_id
        self.num_classes = num_classes

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - sklearn-style signature
        return np.full(X.shape[0], self.class_id, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        probs = np.zeros((X.shape[0], self.num_classes), dtype=float)
        probs[:, self.class_id] = 1.0
        return probs


__all__ = ["SimplePLDA", "PLDAConfig", "ConstantPLDA"]
