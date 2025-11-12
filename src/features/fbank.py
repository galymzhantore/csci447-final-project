from __future__ import annotations

from typing import Dict

import librosa
import numpy as np


def compute_fbank(waveform: np.ndarray, sample_rate: int, cfg: Dict) -> np.ndarray:
    hop = int(sample_rate * cfg.get("frame_shift_ms", 10) / 1000)
    win = int(sample_rate * cfg.get("frame_length_ms", 25) / 1000)
    spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=cfg.get("n_mels", 40), hop_length=hop, n_fft=win)
    log_spec = librosa.power_to_db(spec)
    return log_spec.astype(np.float32)


__all__ = ["compute_fbank"]
