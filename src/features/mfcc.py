from __future__ import annotations

from typing import Dict

import librosa
import numpy as np


def compute_mfcc(waveform: np.ndarray, sample_rate: int, cfg: Dict) -> np.ndarray:
    n_mfcc = cfg.get("n_mfcc", 20)
    n_mels = cfg.get("n_mels", 80)
    hop = int(sample_rate * cfg.get("frame_shift_ms", 10) / 1000)
    win = int(sample_rate * cfg.get("frame_length_ms", 25) / 1000)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc, n_mels=n_mels, hop_length=hop, n_fft=win)
    features = [mfcc]
    if cfg.get("add_deltas", True):
        features.append(librosa.feature.delta(mfcc))
    if cfg.get("add_delta_deltas", True):
        features.append(librosa.feature.delta(mfcc, order=2))
    stacked = np.vstack(features)
    return stacked.astype(np.float32)


__all__ = ["compute_mfcc"]
