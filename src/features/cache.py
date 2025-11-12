from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np


class FeatureCache:
    def __init__(self, root: Path, feature_type: str) -> None:
        self.root = Path(root)
        self.feature_type = feature_type
        self.root.mkdir(parents=True, exist_ok=True)

    def _hash(self, audio_path: str) -> str:
        return hashlib.sha1(audio_path.encode("utf-8")).hexdigest()

    def path_for(self, audio_path: str) -> Path:
        return self.root / self.feature_type / f"{self._hash(audio_path)}.npy"

    def load(self, audio_path: str) -> Optional[np.ndarray]:
        path = self.path_for(audio_path)
        if path.exists():
            return np.load(path)
        return None

    def save(self, audio_path: str, feats: np.ndarray) -> Path:
        path = self.path_for(audio_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, feats.astype(np.float32))
        return path


__all__ = ["FeatureCache"]
