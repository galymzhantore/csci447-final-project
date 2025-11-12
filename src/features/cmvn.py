from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


class CMVN:
    def __init__(self) -> None:
        self.count = 0
        self.mean = None
        self.var = None

    def update(self, feats: np.ndarray) -> None:
        feats = feats.astype(np.float64)
        if self.mean is None:
            self.mean = feats.mean(axis=1)
            self.var = feats.var(axis=1)
            self.count = feats.shape[1]
        else:
            self.count += feats.shape[1]
            new_mean = feats.mean(axis=1)
            new_var = feats.var(axis=1)
            self.mean = (self.mean + new_mean) / 2
            self.var = (self.var + new_var) / 2

    def apply(self, feats: np.ndarray) -> np.ndarray:
        if self.mean is None or self.var is None:
            return feats
        return ((feats - self.mean[:, None]) / (np.sqrt(self.var[:, None] + 1e-9))).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    @classmethod
    def load(cls, path: Path) -> "CMVN":
        cmvn = cls()
        if path.exists():
            data = np.load(path)
            cmvn.mean = data["mean"]
            cmvn.var = data["var"]
            cmvn.count = int(data["count"])
        return cmvn


def apply_cmvn(feats: np.ndarray, mode: str, cmvn_stats: Dict | None = None) -> np.ndarray:
    if mode == "none":
        return feats
    if mode == "utterance":
        mean = feats.mean(axis=1, keepdims=True)
        std = feats.std(axis=1, keepdims=True) + 1e-9
        return ((feats - mean) / std).astype(np.float32)
    if mode == "global" and cmvn_stats:
        mean = cmvn_stats["mean"][:, None]
        std = np.sqrt(cmvn_stats["var"] + 1e-9)[:, None]
        return ((feats - mean) / std).astype(np.float32)
    return feats


__all__ = ["CMVN", "apply_cmvn"]
