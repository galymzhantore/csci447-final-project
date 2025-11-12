from __future__ import annotations

from typing import Dict

import numpy as np

from .fbank import compute_fbank
from .mfcc import compute_mfcc


def extract_features(waveform: np.ndarray, sr: int, feature_cfg: Dict) -> np.ndarray:
    feat_type = feature_cfg.get("type", "mfcc")
    params = feature_cfg.get(feat_type, {}) | {
        "frame_length_ms": feature_cfg.get("frame_length_ms", 25),
        "frame_shift_ms": feature_cfg.get("frame_shift_ms", 10),
    }
    if feat_type == "fbank":
        feats = compute_fbank(waveform, sr, params)
    else:
        feats = compute_mfcc(waveform, sr, params)
    return feats.astype(np.float32)


__all__ = ["extract_features"]
