from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from torch import nn

from models.mlp import build_mlp
from utils.cli import parse_config
from utils.config import parse_label_schema
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def load_model(cfg, input_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    model = build_mlp(cfg["training"].get("model", "mlp_small"), input_dim, num_classes, cfg["models"])
    ckpt = Path(cfg["data"].get("output_dir", "experiments")) / "checkpoints" / f"{cfg["training"].get("model", "mlp_small")}.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def dynamic_quantization(model: nn.Module, out_path: Path) -> Path:
    q_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(q_model.state_dict(), out_path)
    logger.info("Saved dynamic quantized model to %s", out_path)
    return out_path


def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")
    return model


def main() -> None:
    cfg = parse_config("Quantize and prune models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = cfg["data"].get("label_map") or parse_label_schema(cfg["data"].get("label_schema", "3class"))
    sample_frames = int(cfg["data"].get("clip_seconds", [1, 5])[1] * 1000 / cfg["features"].get("frame_shift_ms", 10))
    feat_type = cfg["features"].get("type", "mfcc")
    if feat_type == "fbank":
        freq_dim = cfg["features"].get("fbank", {}).get("n_mels", 40)
    else:
        base = cfg["features"].get("mfcc", {}).get("n_mfcc", 20)
        factor = 1
        if cfg["features"].get("mfcc", {}).get("add_deltas", True):
            factor += 1
        if cfg["features"].get("mfcc", {}).get("add_delta_deltas", True):
            factor += 1
        freq_dim = base * factor
    input_dim = freq_dim * sample_frames
    model = load_model(cfg, input_dim, len(label_map), device)

    artifacts_dir = Path(cfg["data"].get("output_dir", "experiments")) / "quantized"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dynamic_path = artifacts_dir / "model_dynamic.pt"
    dynamic_quantization(model.cpu(), dynamic_path)

    pruned_model = prune_model(build_mlp(cfg["training"].get("model", "mlp_small"), input_dim, len(label_map), cfg["models"]), cfg.get("quantization", {}).get("prune_amount", 0.3))
    pruned_path = artifacts_dir / "model_pruned.pt"
    torch.save(pruned_model.state_dict(), pruned_path)

    summary = {
        "fp32_size_mb": model_size_mb(Path(cfg["data"].get("output_dir", "experiments")) / "checkpoints" / f"{cfg["training"].get("model", "mlp_small")}.pt"),
        "dynamic_size_mb": model_size_mb(dynamic_path),
        "pruned_size_mb": model_size_mb(pruned_path),
    }
    (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Quantization summary saved to %s", artifacts_dir / "summary.json")


if __name__ == "__main__":
    main()
