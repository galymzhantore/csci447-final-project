from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np

from .noise import load_noise_clip, load_rir


def snr_mix(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = librosa.util.fix_length(noise, len(clean))
    rms_clean = np.sqrt(np.mean(clean**2) + 1e-9)
    rms_noise = np.sqrt(np.mean(noise**2) + 1e-9)
    target_noise_rms = rms_clean / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / (rms_noise + 1e-9))
    return clean + noise


def gaussian_noise(clean: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    noise = np.random.randn(clean.shape[0]).astype(clean.dtype)
    return snr_mix(clean, noise, snr_db)


def pink_noise(clean: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    uneven = clean.size % 2
    X = np.random.randn(clean.size // 2 + 1 + uneven) + 1j * np.random.randn(clean.size // 2 + 1 + uneven)
    S = np.arange(len(X)) + 1.0
    X = X / np.sqrt(S)
    y = np.fft.irfft(X).real
    if uneven:
        y = y[:-1]
    y = y / np.max(np.abs(y))
    return snr_mix(clean, y.astype(clean.dtype), snr_db)


def tempo_pitch(clean: np.ndarray, sr: int, tempo_range=(0.95, 1.05), pitch_semitones=(-1.0, 1.0)) -> np.ndarray:
    tempo = random.uniform(*tempo_range)
    pitched = librosa.effects.time_stretch(clean, rate=tempo)
    semitone = random.uniform(*pitch_semitones)
    pitched = librosa.effects.pitch_shift(pitched, sr=sr, n_steps=semitone)
    return librosa.util.fix_length(pitched, len(clean))


def rir_convolution(clean: np.ndarray, rir_dir: Optional[Path], sr: int, prob: float = 0.3) -> np.ndarray:
    if rir_dir is None or not rir_dir.exists() or random.random() > prob:
        return clean
    rir = load_rir(rir_dir)
    augmented = np.convolve(clean, rir)[: len(clean)]
    return augmented.astype(clean.dtype)


def background_mix(clean: np.ndarray, noise_dir: Optional[Path], snr_db: float = 10.0) -> np.ndarray:
    if noise_dir is None or not noise_dir.exists():
        return clean
    noise = load_noise_clip(noise_dir, target_len=len(clean))
    return snr_mix(clean, noise, snr_db)


def apply_augmentations(waveform: np.ndarray, sr: int, cfg: Dict) -> np.ndarray:
    aug = waveform
    if cfg.get("gaussian_snr_db"):
        snr = random.choice(cfg["gaussian_snr_db"])
        aug = gaussian_noise(aug, snr)
    if cfg.get("pink_snr_db"):
        snr = random.choice(cfg["pink_snr_db"])
        aug = pink_noise(aug, snr)
    if cfg.get("tempo_range"):
        aug = tempo_pitch(aug, sr, tuple(cfg["tempo_range"]), tuple(cfg.get("pitch_semitones", (-1, 1))))
    noise_dir = Path(cfg.get("noise_dir", "")) if cfg.get("noise_dir") else None
    if noise_dir and cfg.get("bg_mix_prob", 0) > random.random():
        snr = random.choice(cfg.get("snr_sweep_db", [10]))
        aug = background_mix(aug, noise_dir, snr)
    rir_dir = Path(cfg.get("rir_dir", "")) if cfg.get("rir_dir") else None
    aug = rir_convolution(aug, rir_dir, sr, cfg.get("rir_prob", 0.0))
    return librosa.util.fix_length(aug, len(waveform))


__all__ = ["apply_augmentations"]
