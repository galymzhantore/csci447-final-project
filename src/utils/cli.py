from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .config import load_config, merge_overrides


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--data-dir", dest="data_dir", default=None, help="Override data directory")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Override experiment directory")
    parser.add_argument("--label-schema", dest="label_schema", default=None, help="Label schema string or JSON mapping")
    parser.add_argument("--model", default=None, help="Model name to train/evaluate")
    parser.add_argument("--snr-sweep", dest="snr_sweep", default=None, help="Comma-separated SNR values")
    parser.add_argument("--target-device", dest="target_device", default=None, help="Target device profile name")
    parser.add_argument("--teacher-checkpoint", dest="teacher_checkpoint", default=None, help="Teacher checkpoint path")
    parser.add_argument("--extra", nargs="*", default=None, help="Additional key=value overrides")
    return parser


def apply_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    overrides: Dict[str, Any] = {}
    if args.data_dir:
        overrides.setdefault("data", {})["data_dir"] = args.data_dir
    if args.output_dir:
        overrides.setdefault("data", {})["output_dir"] = args.output_dir
    if args.label_schema:
        overrides.setdefault("data", {})["label_schema"] = args.label_schema
    if args.model:
        overrides.setdefault("training", {})["model"] = args.model
    if args.snr_sweep:
        overrides.setdefault("evaluation", {})["snr_sweep_db"] = [int(x) for x in args.snr_sweep.split(",") if x]
    if args.target_device:
        overrides.setdefault("profile", {})["target_device"] = args.target_device
    if args.teacher_checkpoint:
        overrides.setdefault("training", {}).setdefault("distillation", {})["teacher_checkpoint"] = args.teacher_checkpoint
    if args.extra:
        for item in args.extra:
            key, value = item.split("=", maxsplit=1)
            section, _, subkey = key.partition(".")
            overrides.setdefault(section, {})[subkey] = value
    return merge_overrides(cfg, overrides)


def parse_config(description: str) -> Dict[str, Any]:
    parser = build_parser(description)
    args = parser.parse_args()
    cfg = apply_cli_overrides(args)
    cfg["config_path"] = str(Path(args.config).resolve())
    return cfg


__all__ = ["parse_config", "build_parser", "apply_cli_overrides"]
