from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.data_module import DataModule, load_cmvn
from data.dataset import SpeechDataset
from features.cache import FeatureCache
from models.mlp import build_mlp
from utils.cli import parse_config
from utils.config import parse_label_schema
from utils.logging_utils import setup_logger
from utils.metrics import compute_metrics
from utils.plotting import save_confusion_matrix, save_metric_curve

logger = setup_logger(__name__)


def load_model(cfg: Dict, input_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    model = build_mlp(cfg["training"].get("model", "mlp_small"), input_dim, num_classes, cfg["models"]).to(device)
    ckpt_path = Path(cfg["data"].get("output_dir", "experiments")) / "checkpoints" / f"{cfg["training"].get("model", "mlp_small")}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, class_names: List[str]) -> Dict:
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
    return {"metrics": metrics, "y_true": targets, "y_pred": preds}


def snr_sweep(cfg: Dict, model: nn.Module, label_map: Dict[str, int], device: torch.device, snr_levels: List[int]) -> Dict[int, float]:
    data_cfg = cfg["data"].copy()
    feature_cfg = cfg["features"].copy()
    feature_cfg["on_the_fly"] = True
    cmvn_path = Path(cfg["data"].get("output_dir", "experiments")) / "features" / "cmvn.npz"
    cmvn_stats = load_cmvn(cmvn_path)
    cache = FeatureCache(Path(feature_cfg.get("cache_dir", "data/features")), feature_cfg.get("type", "mfcc"))
    manifest = Path(cfg["data"].get("output_dir", "experiments")) / "manifests" / "test.csv"
    results = {}
    for snr in snr_levels:
        aug_cfg = cfg.get("augmentation", {}).copy()
        aug_cfg.update({"enabled": True, "gaussian_snr_db": [snr], "pink_snr_db": []})
        dataset = SpeechDataset(
            manifest,
            cache,
            feature_cfg,
            data_cfg,
            cmvn_stats,
            label_map,
            augmentation_cfg=aug_cfg,
            split="eval",
        )
        loader = DataLoader(dataset, batch_size=cfg["evaluation"].get("batch_size", 64), shuffle=False)
        metrics = evaluate_loader(model, loader, device, list(label_map.keys()))
        results[snr] = metrics["metrics"].accuracy
        logger.info("SNR %s dB -> accuracy %.3f", snr, metrics["metrics"].accuracy)
    return results


def main() -> None:
    cfg = parse_config("Evaluate edge fluency models")
    label_map = cfg["data"].get("label_map") or parse_label_schema(cfg["data"].get("label_schema", "3class"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = DataModule(cfg["data"], cfg["features"], cfg.get("augmentation", {}), label_map)
    train_loader = data_module.train_dataloader(cfg["training"].get("batch_size", 32))
    sample = train_loader.dataset[0]
    feat = sample["features"]
    input_dim = feat.shape[0] * feat.shape[1]
    num_classes = len(label_map)
    model = load_model(cfg, input_dim, num_classes, device)

    evals = {}
    for split, loader in {
        "train": train_loader,
        "valid": data_module.valid_dataloader(cfg["training"].get("batch_size", 32)),
        "test": data_module.test_dataloader(cfg["evaluation"].get("batch_size", 64)),
    }.items():
        result = evaluate_loader(model, loader, device, list(label_map.keys()))
        evals[split] = {
            "accuracy": result["metrics"].accuracy,
            "macro_f1": result["metrics"].macro_f1,
            "confusion": result["metrics"].confusion,
        }
        if split == "test":
            fig_path = Path(cfg["data"].get("output_dir", "experiments")) / "figures" / "confusion_matrix.png"
            save_confusion_matrix(result["metrics"].confusion, list(label_map.keys()), fig_path)

    snr_levels = cfg.get("evaluation", {}).get("snr_sweep_db", [0, 5, 10, 15, 20, 30])
    snr_results = snr_sweep(cfg, model, label_map, device, snr_levels)
    sweep_path = Path(cfg["data"].get("output_dir", "experiments")) / "figures" / "snr_curve.png"
    save_metric_curve(snr_levels, [snr_results[s] for s in snr_levels], "SNR (dB)", "Accuracy", "Robustness", sweep_path)

    report_path = Path(cfg["data"].get("output_dir", "experiments")) / "metrics" / "evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"splits": evals, "snr": snr_results}, indent=2), encoding="utf-8")
    logger.info("Evaluation artifacts saved to %s", report_path)


if __name__ == "__main__":
    main()
