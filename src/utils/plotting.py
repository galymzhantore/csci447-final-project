from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def save_confusion_matrix(confusion: List[List[int]], labels: List[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_metric_curve(x: Iterable[float], y: Iterable[float], xlabel: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(list(x), list(y), marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_pareto(points: List[Dict[str, float]], path: Path, x_key: str = "latency_ms", y_key: str = "accuracy") -> None:
    fig, ax = plt.subplots()
    xs = [p[x_key] for p in points]
    ys = [p[y_key] for p in points]
    labels = [p.get("label", f"pt{i}") for i, p in enumerate(points)]
    ax.scatter(xs, ys)
    for label, x, y in zip(labels, xs, ys):
        ax.annotate(label, (x, y))
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title("Pareto Front")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


__all__ = ["save_confusion_matrix", "save_metric_curve", "save_pareto"]
