from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil

from utils.cli import parse_config
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def load_samples(cfg: Dict, batch_size: int = 8) -> np.ndarray:
    manifest = Path(cfg["data"].get("output_dir", "experiments")) / "manifests" / "test.csv"
    if not manifest.exists():
        raise FileNotFoundError("Test manifest missing, run download/features")
    import pandas as pd
    from audio.processing import preprocess
    from features.extractor import extract_features

    df = pd.read_csv(manifest).head(batch_size)
    feats = []
    max_frames = 0
    aug_cfg = cfg.get("augmentation", {})
    for path in df["path"]:
        waveform = preprocess(
            path,
            cfg["data"].get("sample_rate", 16000),
            cfg["data"].get("clip_seconds", [1, 5])[0],
            cfg["data"].get("clip_seconds", [1, 5])[1],
            aug_cfg.get("trim_db", 25),
            aug_cfg.get("vad_threshold", 0.5),
            aug_cfg.get("peak_normalize", True),
        )
        feat = extract_features(waveform, cfg["data"].get("sample_rate", 16000), cfg["features"])
        max_frames = max(max_frames, feat.shape[1])
        feats.append(feat)
    padded = []
    for feat in feats:
        if feat.shape[1] < max_frames:
            pad = np.zeros((feat.shape[0], max_frames - feat.shape[1]), dtype=feat.dtype)
            feat = np.concatenate([feat, pad], axis=1)
        padded.append(feat)
    return np.stack(padded)


def profile_onnx(cfg: Dict, samples: np.ndarray, reps: int) -> Dict:
    import onnxruntime as ort

    onnx_path = Path(cfg["data"].get("output_dir", "experiments")) / "exports" / "model.onnx"
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    latencies = []
    for _ in range(reps):
        start = time.perf_counter()
        session.run(None, {input_name: samples})
        latencies.append((time.perf_counter() - start) * 1000)
    return {"latency_ms": float(np.mean(latencies)), "std_ms": float(np.std(latencies)), "model": "onnx"}


def profile_tflite(cfg: Dict, samples: np.ndarray, reps: int) -> Dict:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    tflite_path = Path(cfg["data"].get("output_dir", "experiments")) / "exports" / "model.tflite"
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    latencies = []
    for _ in range(reps):
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], samples.reshape(samples.shape[0], -1).astype(np.float32))
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
        latencies.append((time.perf_counter() - start) * 1000)
    return {"latency_ms": float(np.mean(latencies)), "std_ms": float(np.std(latencies)), "model": "tflite"}


def energy_proxy(latency_ms: float, device: str) -> float:
    power = psutil.sensors_battery().power_plugged if hasattr(psutil, "sensors_battery") else 1.0
    factor = 0.8 if device == "pi4" else 1.2
    return float(latency_ms * factor * power)


def save_profile(results: List[Dict], cfg: Dict) -> None:
    out_dir = Path(cfg["data"].get("output_dir", "experiments")) / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "profile.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "latency_ms", "std_ms", "energy_proxy"])
        writer.writeheader()
        writer.writerows(results)
    summary_path = out_dir / "summary.json"
    import json

    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Profile saved to %s", csv_path)


def main() -> None:
    cfg = parse_config("Profile exports on device")
    samples = load_samples(cfg)
    reps = cfg.get("profile", {}).get("repetitions", 20)
    onnx_stats = profile_onnx(cfg, samples, reps)
    tflite_stats = profile_tflite(cfg, samples, reps)
    target = cfg.get("profile", {}).get("target_device", "pi4")
    for stat in (onnx_stats, tflite_stats):
        stat["energy_proxy"] = energy_proxy(stat["latency_ms"], target)
    save_profile([onnx_stats, tflite_stats], cfg)


if __name__ == "__main__":
    main()
