from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class BaselineResult:
    name: str
    accuracy: float


def train_baseline(name: str, features: np.ndarray, labels: np.ndarray, cfg: Dict) -> Tuple[object, BaselineResult]:
    if name == "knn":
        model = KNeighborsClassifier(n_neighbors=cfg.get("k", 3))
    elif name == "lda":
        model = LinearDiscriminantAnalysis()
    else:
        model = Perceptron(max_iter=1000)
    model.fit(features, labels)
    preds = model.predict(features)
    acc = float((preds == labels).mean())
    return model, BaselineResult(name=name, accuracy=acc)


__all__ = ["train_baseline", "BaselineResult"]
