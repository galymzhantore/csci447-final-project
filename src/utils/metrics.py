from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score


@dataclass
class MetricsBundle:
    accuracy: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]]
    confusion: List[List[int]]
    roc_auc: Dict[str, float]


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_proba: Sequence[Sequence[float]] | None, class_names: List[str]) -> MetricsBundle:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    acc = accuracy_score(y_true_arr, y_pred_arr)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, labels=range(len(class_names)), zero_division=0)
    macro = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    per_class = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for name, p, r, f in zip(class_names, precision, recall, f1)
    }
    conf = confusion_matrix(y_true_arr, y_pred_arr, labels=range(len(class_names))).tolist()
    roc: Dict[str, float] = {}
    if y_proba is not None and len(class_names) > 2:
        y_proba_arr = np.asarray(y_proba)
        for idx, name in enumerate(class_names):
            try:
                roc[name] = float(roc_auc_score((y_true_arr == idx).astype(int), y_proba_arr[:, idx]))
            except ValueError:
                roc[name] = float("nan")
    else:
        roc = {name: float("nan") for name in class_names}
    return MetricsBundle(accuracy=float(acc), macro_f1=macro, per_class=per_class, confusion=conf, roc_auc=roc)


__all__ = ["compute_metrics", "MetricsBundle"]
