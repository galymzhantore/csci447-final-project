from __future__ import annotations

import json
import shutil
from pathlib import Path

from utils.cli import parse_config
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def load_export_summary(export_dir: Path) -> dict:
    summary_path = export_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Export summary not found at {summary_path}. Run make export_tflite first.")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def ensure_label_map(summary: dict, export_dir: Path) -> Path:
    label_map_path = Path(summary.get("label_map_path", "")) if summary.get("label_map_path") else export_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError("Label map missing; rerun export_tflite.")
    return label_map_path


def copy_asset(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    logger.info("Copied %s -> %s", src, dst)


def main() -> None:
    cfg = parse_config("Copy TFLite assets into Android project")
    export_dir = Path(cfg["data"].get("output_dir", "experiments")) / "exports"
    summary = load_export_summary(export_dir)
    android_cfg = cfg.get("android", {})

    model_path = Path(summary.get("int8_tflite", {}).get("path") or summary.get("fp32_tflite", {}).get("path"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model asset missing at {model_path}. Run make export_tflite.")
    label_map_path = ensure_label_map(summary, export_dir)

    model_dest = Path(android_cfg.get("model_asset", "android_app/app/src/main/assets/model_fluency.tflite"))
    label_dest = Path(android_cfg.get("label_map_asset", "android_app/app/src/main/assets/label_map.json"))
    metadata_dest = Path(android_cfg.get("metrics_asset", "android_app/app/src/main/assets/metadata.json"))

    copy_asset(model_path, model_dest)
    copy_asset(label_map_path, label_dest)

    metadata = {
        "freq": summary.get("freq"),
        "frames": summary.get("frames"),
        "input_dim": summary.get("input_dim"),
        "sample_rate": cfg["data"].get("sample_rate", 16000),
        "clip_seconds": cfg["data"].get("clip_seconds", [1, 5])[1],
        "fp32_accuracy": summary.get("fp32_tflite", {}).get("accuracy"),
        "int8_accuracy": summary.get("int8_tflite", {}).get("accuracy"),
        "accuracy_drop": summary.get("int8_tflite", {}).get("accuracy_drop"),
        "model_path": model_dest.as_posix(),
        "label_map_path": label_dest.as_posix(),
    }
    metadata_dest.parent.mkdir(parents=True, exist_ok=True)
    metadata_dest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote Android metadata to %s", metadata_dest)


if __name__ == "__main__":
    main()
