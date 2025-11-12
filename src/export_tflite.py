from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.onnx

from data.data_module import DataModule
from models.mlp import build_mlp
from utils.cli import parse_config
from utils.config import parse_label_schema
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def _prepare_data(cfg: Dict) -> Tuple[np.ndarray, int, int, Dict[str, int], torch.utils.data.DataLoader]:
    label_map = cfg["data"].get("label_map") or parse_label_schema(cfg["data"].get("label_schema", "3class"))
    data_module = DataModule(cfg["data"], cfg["features"], cfg.get("augmentation", {}), label_map)
    train_loader = data_module.train_dataloader(cfg["training"].get("batch_size", 32))
    valid_loader = data_module.valid_dataloader(cfg["evaluation"].get("batch_size", 64))
    sample = next(iter(train_loader))
    feat = sample["features"][:1]
    freq, frames = feat.shape[1:]
    return feat.numpy(), freq, frames, label_map, valid_loader


def _load_model(cfg: Dict, input_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    model = build_mlp(cfg["training"].get("model", "mlp_small"), input_dim, num_classes, cfg["models"])
    ckpt = Path(cfg["data"].get("output_dir", "experiments")) / "checkpoints" / f"{cfg['training'].get('model', 'mlp_small')}.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    model.to(device)
    return model


def export_onnx(model: nn.Module, dummy: np.ndarray, out_path: Path) -> np.ndarray:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_tensor = torch.from_numpy(dummy)
    torch.onnx.export(
        model,
        dummy_tensor,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    logger.info("Saved ONNX model to %s", out_path)
    return dummy_tensor.numpy()


def validate_onnx(onnx_path: Path, sample_input: np.ndarray, torch_out: np.ndarray) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: sample_input})[0]
    delta = float(np.max(np.abs(ort_out - torch_out)))
    logger.info("ONNX max deviation: %.6f", delta)
    return delta


def build_tf_model(cfg: Dict, input_dim: int, num_classes: int, state_dict: Dict[str, torch.Tensor]):
    import tensorflow as tf

    model_name = cfg["training"].get("model", "mlp_small")
    model_cfg = cfg["models"].get(model_name, cfg["models"].get("mlp_small", {}))
    activation = model_cfg.get("activation", cfg["models"].get("mlp_activation", "relu"))
    keras_layers = [tf.keras.layers.Input(shape=(input_dim,))]
    hidden_sizes = model_cfg.get("hidden_sizes", [128, 64])
    for hidden in hidden_sizes:
        keras_layers.append(tf.keras.layers.Dense(hidden, activation=activation))
    keras_layers.append(tf.keras.layers.Dense(num_classes, activation="linear"))
    tf_model = tf.keras.Sequential(keras_layers)
    dense_layers = [layer for layer in tf_model.layers if isinstance(layer, tf.keras.layers.Dense)]
    for idx, dense in enumerate(dense_layers[:-1]):
        weight_key = f"backbone.{idx * 3}.weight"
        bias_key = f"backbone.{idx * 3}.bias"
        weight = state_dict[weight_key].cpu().numpy().T
        bias = state_dict[bias_key].cpu().numpy()
        dense.set_weights([weight, bias])
    clf_weight = state_dict["classifier.weight"].cpu().numpy().T
    clf_bias = state_dict["classifier.bias"].cpu().numpy()
    dense_layers[-1].set_weights([clf_weight, clf_bias])
    return tf_model


def export_tflite(tf_model, out_path: Path, representative_dataset=None, int8: bool = False, optimizations=None) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    if optimizations:
        converter.optimizations = optimizations
    if int8:
        converter.optimizations = optimizations or [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    logger.info("Saved TFLite model to %s", out_path)
    return tflite_model


def representative_dataset(loader, sample_limit: int):
    def gen():
        count = 0
        for batch in loader:
            feats = batch["features"].numpy().reshape(batch["features"].shape[0], -1).astype(np.float32)
            for row in feats:
                yield [row[np.newaxis, :]]
                count += 1
                if sample_limit and count >= sample_limit:
                    return

    return gen


def _interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:  # pragma: no cover - fallback for dev machines
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    return Interpreter


def validate_tflite(tflite_bytes: bytes, sample_input: np.ndarray, torch_out: np.ndarray) -> float:
    Interpreter = _interpreter()
    interpreter = Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    flat = sample_input.reshape(sample_input.shape[0], -1).astype(np.float32)
    prepared = _quantize_input(flat, input_details)
    interpreter.set_tensor(input_details["index"], prepared)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    output = _dequantize_output(output, output_details)
    delta = float(np.max(np.abs(output - torch_out)))
    logger.info("TFLite max deviation: %.6f", delta)
    return delta


def _quantize_input(feats: np.ndarray, details: Dict) -> np.ndarray:
    if details["dtype"] == np.int8:
        scale, zero_point = details["quantization"]
        return np.clip(np.round(feats / scale + zero_point), -128, 127).astype(np.int8)
    return feats.astype(details["dtype"])


def _dequantize_output(output: np.ndarray, details: Dict) -> np.ndarray:
    if details["dtype"] == np.int8:
        scale, zero_point = details["quantization"]
        return scale * (output.astype(np.float32) - zero_point)
    return output.astype(np.float32)


def evaluate_tflite(tflite_bytes: bytes, loader, max_batches: int) -> float:
    Interpreter = _interpreter()
    interpreter = Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    total = 0
    correct = 0
    for idx, batch in enumerate(loader):
        feats = batch["features"].numpy().reshape(batch["features"].shape[0], -1).astype(np.float32)
        labels = batch["label"].numpy()
        prepared = _quantize_input(feats, input_details)
        interpreter.set_tensor(input_details["index"], prepared)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        logits = _dequantize_output(output, output_details)
        preds = np.argmax(logits, axis=1)
        correct += int((preds == labels).sum())
        total += labels.size
        if max_batches and idx + 1 >= max_batches:
            break
    return float(correct / max(total, 1))


def save_label_map(label_map: Dict[str, int], path: Path) -> None:
    path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")


def main() -> None:
    cfg = parse_config("Export models to edge runtimes")
    sample, freq, frames, label_map, valid_loader = _prepare_data(cfg)
    input_dim = freq * frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg, input_dim, len(label_map), device)
    torch_out = model(torch.from_numpy(sample).to(device)).detach().cpu().numpy()

    export_dir = Path(cfg["data"].get("output_dir", "experiments")) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_dir / "model.onnx"
    sample_input = export_onnx(model, sample, onnx_path)
    onnx_delta = validate_onnx(onnx_path, sample_input, torch_out)

    tf_model = build_tf_model(cfg, input_dim, len(label_map), model.state_dict())
    tflite_fp32 = export_tflite(tf_model, export_dir / "model_fp32.tflite")
    tflite_delta = validate_tflite(tflite_fp32, sample, torch_out)
    fp32_acc = evaluate_tflite(tflite_fp32, valid_loader, cfg.get("export", {}).get("max_eval_batches", 5))

    int8_metrics = {}
    if cfg.get("export", {}).get("tflite_int8", True):
        reps = representative_dataset(valid_loader, cfg.get("quantization", {}).get("int8_calibration_samples", 128))
        tflite_int8 = export_tflite(
            tf_model,
            export_dir / "model_int8.tflite",
            representative_dataset=reps,
            int8=True,
            optimizations=["DEFAULT"],
        )
        int8_acc = evaluate_tflite(tflite_int8, valid_loader, cfg.get("export", {}).get("max_eval_batches", 5))
        int8_metrics = {
            "path": str(export_dir / "model_int8.tflite"),
            "accuracy": int8_acc,
            "accuracy_drop": float(fp32_acc - int8_acc),
        }

    label_map_path = export_dir / "label_map.json"
    save_label_map(label_map, label_map_path)

    summary = {
        "freq": freq,
        "frames": frames,
        "input_dim": input_dim,
        "label_map_path": str(label_map_path),
        "onnx_path": str(onnx_path),
        "fp32_tflite": {"path": str(export_dir / "model_fp32.tflite"), "accuracy": fp32_acc, "delta": tflite_delta},
        "int8_tflite": int8_metrics,
        "onnx_delta": onnx_delta,
    }
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Export summary saved to %s", export_dir / "summary.json")


if __name__ == "__main__":
    main()
