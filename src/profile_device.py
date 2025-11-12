from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

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


def process_memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


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
    return {
        "latency_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "model": "onnx",
        "memory_mb": process_memory_mb(),
    }


def _tflite_path(cfg: Dict) -> Path:
    export_dir = Path(cfg["data"].get("output_dir", "experiments")) / "exports"
    for name in ("model_int8.tflite", "model_fp32.tflite", "model.tflite"):
        path = export_dir / name
        if path.exists():
            return path
    raise FileNotFoundError("No TFLite artifact found. Run make export_tflite.")


def profile_tflite(cfg: Dict, samples: np.ndarray, reps: int) -> Dict:
    Interpreter = _interpreter()
    tflite_path = _tflite_path(cfg)
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    latencies = []
    flat = samples.reshape(samples.shape[0], -1).astype(np.float32)
    prepared = _quantize_input(flat, input_details)
    for _ in range(reps):
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], prepared)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "latency_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "model": f"tflite ({tflite_path.name})",
        "memory_mb": process_memory_mb(),
    }


def _interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:  # pragma: no cover
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    return Interpreter


def _quantize_input(feats: np.ndarray, details: Dict) -> np.ndarray:
    if details["dtype"] == np.int8:
        scale, zero_point = details["quantization"]
        return np.clip(np.round(feats / scale + zero_point), -128, 127).astype(np.int8)
    return feats.astype(details["dtype"])


def energy_proxy(latency_ms: float, device: str) -> float:
    power = psutil.sensors_battery().power_plugged if hasattr(psutil, "sensors_battery") else 1.0
    factor = 0.8 if device == "pi4" else 1.2
    return float(latency_ms * factor * power)


def pull_android_metrics(cfg: Dict) -> Optional[Dict]:
    target = cfg.get("profile", {}).get("target_device", "")
    if "android" not in target:
        return None
    adb_path = cfg.get("profile", {}).get("adb_path", "adb")
    package = cfg.get("android", {}).get("package_name", "com.example.fluencyscorer")
    cmd = [adb_path, "shell", "run-as", package, "cat", "files/metrics/latest.json"]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        logger.warning("Unable to pull Android metrics: %s", err)
        return None
    metrics = json.loads(result.stdout.strip() or "{}")
    metrics["model"] = "android_device"
    metrics["source"] = package
    metrics["timestamp"] = time.time()
    return metrics


def save_profile(results: List[Dict], cfg: Dict) -> Path:
    out_dir = Path(cfg["data"].get("output_dir", "experiments")) / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "profile.csv"
    fieldnames = ["model", "latency_ms", "std_ms", "memory_mb", "energy_proxy"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Profile saved to %s", csv_path)
    return summary_path


def mirror_results(results: List[Dict], cfg: Dict, android_metrics: Optional[Dict]) -> None:
    results_dir = Path(cfg.get("reporting", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {"host": results, "android": android_metrics}
    (results_dir / "profile_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    cfg = parse_config("Profile exports on device")
    samples = load_samples(cfg)
    reps = cfg.get("profile", {}).get("repetitions", 20)
    onnx_stats = profile_onnx(cfg, samples, reps)
    tflite_stats = profile_tflite(cfg, samples, reps)
    target = cfg.get("profile", {}).get("target_device", "pi4")
    entries = [onnx_stats, tflite_stats]
    for stat in entries:
        stat["energy_proxy"] = energy_proxy(stat["latency_ms"], target)
    summary_path = save_profile(entries, cfg)
    android_metrics = pull_android_metrics(cfg)
    if android_metrics:
        entries.append(android_metrics)
        (summary_path.parent / "android_metrics.json").write_text(json.dumps(android_metrics, indent=2), encoding="utf-8")
        logger.info("Pulled Android metrics from device")
    mirror_results(entries, cfg, android_metrics)


if __name__ == "__main__":
    main()
