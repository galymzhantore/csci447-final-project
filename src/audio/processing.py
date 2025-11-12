from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


def load_audio(path: str | Path, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    waveform, sr = librosa.load(path, sr=None, mono=True)
    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    return waveform.astype(np.float32), sr


def peak_normalize(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = np.max(np.abs(waveform))
    if peak < eps:
        return waveform
    return waveform / peak


def trim_silence(waveform: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed.astype(np.float32)


def energy_vad(waveform: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    frame_length = 400
    hop = 160
    energy = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop)[0]
    mask = np.repeat(energy > threshold * energy.max(), hop)
    mask = np.pad(mask, (0, max(0, waveform.shape[0] - mask.shape[0])), constant_values=True)
    return waveform[mask[: waveform.shape[0]]]


def enforce_length(waveform: np.ndarray, sr: int, min_seconds: float, max_seconds: float) -> np.ndarray:
    min_samples = int(min_seconds * sr)
    max_samples = int(max_seconds * sr)
    if waveform.shape[0] < min_samples:
        pad = np.zeros(min_samples - waveform.shape[0], dtype=waveform.dtype)
        waveform = np.concatenate([waveform, pad])
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]
    return waveform


def preprocess(path: str | Path, sample_rate: int, min_seconds: float, max_seconds: float, trim_db: float, vad_threshold: float, peak_norm: bool) -> np.ndarray:
    waveform, sr = load_audio(path, sample_rate)
    waveform = trim_silence(waveform, top_db=trim_db)
    waveform = energy_vad(waveform, threshold=vad_threshold)
    waveform = enforce_length(waveform, sr, min_seconds, max_seconds)
    if peak_norm:
        waveform = peak_normalize(waveform)
    return waveform


__all__ = ["preprocess"]
