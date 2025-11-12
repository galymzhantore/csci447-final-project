from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf


def _list_audio_files(root: Path) -> list[Path]:
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def load_noise_clip(noise_dir: Path, target_len: int) -> np.ndarray:
    files = _list_audio_files(noise_dir)
    if not files:
        return np.zeros(target_len, dtype=np.float32)
    path = random.choice(files)
    audio, sr = sf.read(path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < target_len:
        reps = int(np.ceil(target_len / len(audio)))
        audio = np.tile(audio, reps)
    return audio[:target_len]


def load_rir(rir_dir: Path) -> np.ndarray:
    files = _list_audio_files(rir_dir)
    if not files:
        return np.array([1.0], dtype=np.float32)
    path = random.choice(files)
    rir, _ = sf.read(path)
    if rir.ndim > 1:
        rir = rir.mean(axis=1)
    return rir.astype(np.float32)


__all__ = ["load_noise_clip", "load_rir"]
