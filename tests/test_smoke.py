import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def _write_audio_clips(root: Path, count: int = 6, sr: int = 16000) -> list[Path]:
    paths = []
    for idx in range(count):
        waveform = np.sin(np.linspace(0, np.pi * 4, sr)).astype(np.float32) * (idx + 1)
        path = root / f"clip_{idx}.wav"
        sf.write(path, waveform, sr)
        paths.append(path)
    return paths


def _write_manifests(exp_dir: Path, clips: list[Path]) -> None:
    manifests = exp_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": clips[:4],
        "valid": clips[4:5],
        "test": clips[5:],
    }
    for split, split_clips in splits.items():
        rows = []
        for idx, clip in enumerate(split_clips):
            rows.append({
                "path": str(clip),
                "duration": 1.0,
                "text": "demo",
                "speaker_id": f"spk{idx}",
                "label": "poor" if idx % 2 == 0 else "good",
            })
        with (manifests / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "duration", "text", "speaker_id", "label"])
            writer.writeheader()
            writer.writerows(rows)


def test_tiny_training_pipeline(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    experiments = tmp_path / "exp"
    experiments.mkdir(parents=True, exist_ok=True)
    clips = _write_audio_clips(data_dir)
    _write_manifests(experiments, clips)
    cfg = {
        "seed": 1,
        "data": {
            "data_dir": str(data_dir),
            "output_dir": str(experiments),
            "sample_rate": 16000,
            "clip_seconds": [1, 1],
            "label_map": {"poor": 0, "good": 1, "moderate": 2},
            "stratify_by": ["label"],
        },
        "features": {
            "type": "mfcc",
            "mfcc": {"n_mfcc": 13},
            "frame_length_ms": 25,
            "frame_shift_ms": 10,
            "on_the_fly": True,
            "cmvn": "utterance",
            "cache_dir": str(tmp_path / "cache"),
        },
        "augmentation": {"enabled": False},
        "models": {
            "mlp_small": {"hidden_sizes": [8], "dropout": 0.0},
            "mlp_activation": "relu",
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "model": "mlp_small",
            "learning_rate": 1e-3,
            "scheduler": None,
            "patience": 1,
        },
        "evaluation": {"batch_size": 2},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "src.train", "--config", str(cfg_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (experiments / "results.json").exists()
