from __future__ import annotations

import json
import copy
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from data.data_module import DataModule
from models.mlp import build_mlp
from utils.cli import parse_config
from utils.config import parse_label_schema
from utils.logging_utils import setup_logger
from utils.metrics import compute_metrics

logger = setup_logger(__name__)


def prepare_data(cfg):
    label_map = cfg["data"].get("label_map") or parse_label_schema(cfg["data"].get("label_schema", "3class"))
    data_module = DataModule(cfg["data"], cfg["features"], cfg.get("augmentation", {}), label_map)
    train_loader = data_module.train_dataloader(cfg["training"].get("batch_size", 32))
    valid_loader = data_module.valid_dataloader(cfg["evaluation"].get("batch_size", 64))
    sample_batch = next(iter(train_loader))
    feat = sample_batch["features"]
    input_dim = feat.shape[1] * feat.shape[2]
    return label_map, input_dim, train_loader, valid_loader


def load_model(cfg, input_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    model = build_mlp(cfg["training"].get("model", "mlp_small"), input_dim, num_classes, cfg["models"])
    ckpt = Path(cfg["data"].get("output_dir", "experiments")) / "checkpoints" / f"{cfg['training'].get('model', 'mlp_small')}.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def evaluate_model(model: nn.Module, loader, device: torch.device, class_names: list[str]):
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(feats)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
            probs.extend(torch.softmax(logits, dim=1).cpu().tolist())
    metrics = compute_metrics(targets, preds, probs, class_names)
    return metrics


def dynamic_quantization(model: nn.Module, out_path: Path) -> nn.Module:
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "qnnpack"
    q_model = torch.quantization.quantize_dynamic(copy.deepcopy(model).cpu(), {nn.Linear}, dtype=torch.qint8)
    torch.save(q_model.state_dict(), out_path)
    logger.info("Saved dynamic quantized model to %s", out_path)
    return q_model


def _select_backend(preferred=("fbgemm", "qnnpack")) -> str | None:
    supported = getattr(torch.backends.quantized, "supported_engines", [])
    for candidate in preferred:
        if candidate in supported:
            return candidate
    return None


def static_quantization(model: nn.Module, calibration_loader, max_batches: int | None = None) -> nn.Module | None:
    backend = _select_backend()
    if backend is None:
        logger.warning("No supported quantized backend found; skipping static quantization.")
        return None
    try:
        torch.backends.quantized.engine = backend
    except RuntimeError as exc:
        logger.warning("Unable to set quantized backend %s (%s); skipping static quantization.", backend, exc)
        return None
    qconfig_mapping = get_default_qconfig_mapping(backend)
    example_batch = next(iter(calibration_loader))["features"].cpu()
    prepared = prepare_fx(copy.deepcopy(model).cpu().eval(), qconfig_mapping, example_inputs=(example_batch,))
    with torch.no_grad():
        for idx, batch in enumerate(calibration_loader):
            prepared(batch["features"].cpu())
            if max_batches and idx + 1 >= max_batches:
                break
    quantized = convert_fx(prepared)
    return quantized


def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    cloned = copy.deepcopy(model)
    parameters_to_prune = []
    for module in cloned.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")
    return cloned


def main() -> None:
    cfg = parse_config("Quantize and prune models")
    device = torch.device("cpu")
    label_map, input_dim, train_loader, valid_loader = prepare_data(cfg)
    class_names = list(label_map.keys())
    model = load_model(cfg, input_dim, len(label_map), device)
    artifacts_dir = Path(cfg["data"].get("output_dir", "experiments")) / "quantized"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = Path(cfg["data"].get("output_dir", "experiments")) / "checkpoints" / f"{cfg['training'].get('model', 'mlp_small')}.pt"
    base_metrics = evaluate_model(model, valid_loader, device, class_names)

    summary = {
        "fp32": {
            "size_mb": model_size_mb(fp32_path),
            "accuracy": base_metrics.accuracy,
            "macro_f1": base_metrics.macro_f1,
        }
    }

    q_cfg = cfg.get("quantization", {})
    if q_cfg.get("dynamic", True):
        dynamic_path = artifacts_dir / "model_dynamic.pt"
        dyn_model = dynamic_quantization(model, dynamic_path)
        dyn_metrics = evaluate_model(dyn_model, valid_loader, torch.device("cpu"), class_names)
        summary["dynamic"] = {
            "size_mb": model_size_mb(dynamic_path),
            "accuracy": dyn_metrics.accuracy,
            "macro_f1": dyn_metrics.macro_f1,
            "accuracy_drop": float(base_metrics.accuracy - dyn_metrics.accuracy),
        }

    if q_cfg.get("static", True):
        calib_samples = q_cfg.get("int8_calibration_samples", 128)
        batch_size = cfg["training"].get("batch_size", 32)
        max_batches = max(1, calib_samples // max(batch_size, 1))
        static_model = static_quantization(model, train_loader, max_batches)
        if static_model is not None:
            static_path = artifacts_dir / "model_static.pt"
            torch.save(static_model.state_dict(), static_path)
            static_metrics = evaluate_model(static_model, valid_loader, torch.device("cpu"), class_names)
            summary["static"] = {
                "size_mb": model_size_mb(static_path),
                "accuracy": static_metrics.accuracy,
                "macro_f1": static_metrics.macro_f1,
                "accuracy_drop": float(base_metrics.accuracy - static_metrics.accuracy),
            }
        else:
            summary["static"] = {"skipped": True, "reason": "backend_unavailable"}

    prune_amount = q_cfg.get("prune_amount", 0.3)
    pruned_model = prune_model(model, prune_amount)
    pruned_path = artifacts_dir / "model_pruned.pt"
    torch.save(pruned_model.state_dict(), pruned_path)
    pruned_metrics = evaluate_model(pruned_model, valid_loader, device, class_names)
    summary["pruned"] = {
        "size_mb": model_size_mb(pruned_path),
        "accuracy": pruned_metrics.accuracy,
        "macro_f1": pruned_metrics.macro_f1,
        "accuracy_drop": float(base_metrics.accuracy - pruned_metrics.accuracy),
        "amount": prune_amount,
    }

    summary_path = artifacts_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Quantization summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
