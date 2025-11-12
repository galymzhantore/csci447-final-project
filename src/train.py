from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data.data_module import DataModule
from models.distillation import DistillationLoss
from models.mlp import build_mlp
from models.plda import PLDAConfig, SimplePLDA
from utils.cli import parse_config
from utils.config import parse_label_schema
from utils.logging_utils import setup_logger
from utils.metrics import compute_metrics
from utils.seed import set_seed

logger = setup_logger(__name__)


def _prepare_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    label_map = cfg["data"].get("label_map") or parse_label_schema(cfg["data"].get("label_schema", "3class"))
    data_module = DataModule(
        cfg["data"],
        cfg["features"],
        cfg.get("augmentation", {}),
        label_map,
    )
    train_loader = data_module.train_dataloader(cfg["training"].get("batch_size", 32))
    valid_loader = data_module.valid_dataloader(cfg["training"].get("batch_size", 32))
    test_loader = data_module.test_dataloader(cfg["evaluation"].get("batch_size", 64))
    return train_loader, valid_loader, test_loader, label_map


def _determine_dims(loader: DataLoader) -> int:
    batch = next(iter(loader))
    feat = batch["features"]
    return feat.shape[1] * feat.shape[2]


def _init_model(cfg: Dict, input_dim: int, num_classes: int) -> nn.Module:
    model_name = cfg["training"].get("model", "mlp_small")
    return build_mlp(model_name, input_dim, num_classes, cfg["models"])


def _load_teacher(cfg: Dict, input_dim: int, num_classes: int, device: torch.device) -> nn.Module | None:
    teacher_cfg = cfg["training"].get("distillation", {})
    checkpoint = teacher_cfg.get("teacher_checkpoint")
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.exists():
        logger.warning("Teacher checkpoint %s missing", path)
        return None
    teacher = build_mlp("mlp_teacher", input_dim, num_classes, cfg["models"])
    state = torch.load(path, map_location=device)
    teacher.load_state_dict(state["state_dict"])
    teacher.to(device)
    teacher.eval()
    return teacher


def _collect_numpy(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for batch in loader:
        feats = batch["features"].flatten(1).cpu().numpy()
        labs = batch["label"].cpu().numpy()
        features.append(feats)
        labels.append(labs)
    if not features:
        raise RuntimeError("Empty loader provided to PLDA trainer")
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def train_plda(cfg: Dict, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader, label_map: Dict[str, int]) -> None:
    output_dir = Path(cfg["data"].get("output_dir", "experiments"))
    plda_cfg_dict = cfg.get("models", {}).get("plda", {})
    plda_cfg = PLDAConfig(**plda_cfg_dict)
    X_train, y_train = _collect_numpy(train_loader)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = SimplePLDA(plda_cfg, len(label_map))
    model.fit(X_train, y_train)

    def _eval(loader: DataLoader) -> Dict:
        X, y = _collect_numpy(loader)
        Xn = scaler.transform(X)
        preds = model.predict(Xn)
        probs = model.predict_proba(Xn)
        metrics = compute_metrics(y, preds, probs, list(label_map.keys()))
        return {"metrics": metrics}

    evals = {
        "train": _eval(train_loader),
        "valid": _eval(valid_loader),
        "test": _eval(test_loader),
    }
    ckpt_path = output_dir / "checkpoints" / "plda.joblib"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "label_map": label_map, "config": plda_cfg_dict}, ckpt_path)
    summary = {
        "model": "plda",
        "train_macro_f1": evals["train"]["metrics"].macro_f1,
        "valid_macro_f1": evals["valid"]["metrics"].macro_f1,
        "test_macro_f1": evals["test"]["metrics"].macro_f1,
        "test_accuracy": evals["test"]["metrics"].accuracy,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("PLDA training complete; checkpoint at %s", ckpt_path)


def _build_optimizer(model: nn.Module, cfg: Dict):
    opt_name = cfg["training"].get("optimizer", "adam").lower()
    params = model.parameters()
    lr = cfg["training"].get("learning_rate", 1e-3)
    weight_decay = cfg["training"].get("weight_decay", 0.0)
    if opt_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    return Adam(params, lr=lr, weight_decay=weight_decay)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion: DistillationLoss, device: torch.device, scaler: GradScaler | None, teacher: nn.Module | None, mix_precision: bool) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        feats = batch["features"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        context = autocast() if mix_precision else nullcontext()
        with context:
            logits = model(feats)
            teacher_logits = teacher(feats).detach() if teacher is not None else None
            loss = criterion(logits, labels, teacher_logits)
        if scaler and mix_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> Dict:
    model.eval()
    preds, targets, probs = [], [], []
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            probs.extend(torch.softmax(logits, dim=1).cpu().tolist())
    metrics = compute_metrics(targets, preds, probs, class_names)
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "metrics": metrics,
    }


def save_checkpoint(model: nn.Module, path: Path, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(state, path)


def main() -> None:
    cfg = parse_config("Train pronunciation models")
    set_seed(cfg.get("seed", 1337))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader, label_map = _prepare_dataloaders(cfg)
    model_choice = cfg["training"].get("model", "mlp_small")
    if model_choice == "plda":
        train_plda(cfg, train_loader, valid_loader, test_loader, label_map)
        return
    input_dim = _determine_dims(train_loader)
    num_classes = len(label_map)
    model = _init_model(cfg, input_dim, num_classes).to(device)
    teacher = _load_teacher(cfg, input_dim, num_classes, device)
    class_weights = cfg["training"].get("class_weights")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    criterion = DistillationLoss(
        alpha=cfg["training"].get("distillation", {}).get("alpha", 0.5),
        temperature=cfg["training"].get("distillation", {}).get("temperature", 2.0),
        weight=weight_tensor,
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
    )
    optimizer = _build_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(optimizer, patience=cfg["training"].get("patience", 5)) if cfg["training"].get("scheduler") == "reduce_on_plateau" else None
    scaler = GradScaler(enabled=cfg["training"].get("mix_precision", False))

    best_metric = 0.0
    best_path = Path(cfg["data"].get("output_dir", "experiments")) / "checkpoints" / f"{model_choice}.pt"
    history = {"train_loss": [], "valid_macro_f1": []}

    for epoch in range(1, cfg["training"].get("epochs", 10) + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            teacher,
            cfg["training"].get("mix_precision", False),
        )
        eval_result = evaluate(model, valid_loader, device, list(label_map.keys()))
        macro_f1 = eval_result["metrics"].macro_f1
        history["train_loss"].append(train_loss)
        history["valid_macro_f1"].append(macro_f1)
        if scheduler:
            scheduler.step(1 - macro_f1)
        logger.info("Epoch %d: train_loss=%.4f valid_macro_f1=%.4f", epoch, train_loss, macro_f1)
        if macro_f1 > best_metric:
            best_metric = macro_f1
            save_checkpoint(model, best_path, {"macro_f1": macro_f1, "epoch": epoch})

    # final eval on test split
    test_result = evaluate(model, test_loader, device, list(label_map.keys()))
    summary = {
        "model": model_choice,
        "best_macro_f1": best_metric,
        "test_accuracy": test_result["metrics"].accuracy,
        "test_macro_f1": test_result["metrics"].macro_f1,
    }
    results_path = Path(cfg["data"].get("output_dir", "experiments")) / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Training complete. Results saved to %s", results_path)


if __name__ == "__main__":
    main()
