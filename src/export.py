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


def _prepare_sample(cfg: Dict) -> Tuple[np.ndarray, int, int, Dict[str, int]]:
    label_map = cfg["data"].get("label_map") or parse_label_schema(cfg["data"].get("label_schema", "3class"))
    data_module = DataModule(cfg["data"], cfg["features"], cfg.get("augmentation", {}), label_map)
    train_loader = data_module.train_dataloader(cfg["training"].get("batch_size", 32))
    sample = next(iter(train_loader))
    feat = sample["features"][:1]
    freq, frames = feat.shape[1:]
    return feat.numpy(), freq, frames, label_map


def _load_model(cfg: Dict, input_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    model = build_mlp(cfg["training"].get("model", "mlp_small"), input_dim, num_classes, cfg["models"])
    ckpt = Path(cfg["data"].get("output_dir", "experiments")) / "checkpoints" / f"{cfg["training"].get("model", "mlp_small")}.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    model.to(device)
    return model


def export_onnx(model: nn.Module, dummy: np.ndarray, freq: int, frames: int, out_path: Path) -> np.ndarray:
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
    ort_out = session.run(None, {session.get_inputs()[0].name: sample_input})[0]
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
    for idx, hidden in enumerate(hidden_sizes):
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


def export_tflite(tf_model, out_path: Path) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    logger.info("Saved TFLite model to %s", out_path)
    return tflite_model


def validate_tflite(tflite_bytes: bytes, sample_input: np.ndarray, torch_out: np.ndarray) -> float:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], sample_input.reshape(sample_input.shape[0], -1).astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    delta = float(np.max(np.abs(output - torch_out)))
    logger.info("TFLite max deviation: %.6f", delta)
    return delta


def main() -> None:
    cfg = parse_config("Export models to edge runtimes")
    sample, freq, frames, label_map = _prepare_sample(cfg)
    input_dim = freq * frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg, input_dim, len(label_map), device)
    torch_out = model(torch.from_numpy(sample).to(device)).cpu().numpy()

    export_dir = Path(cfg["data"].get("output_dir", "experiments")) / "exports"
    onnx_path = export_dir / "model.onnx"
    sample_input = export_onnx(model, sample, freq, frames, onnx_path)
    onnx_delta = validate_onnx(onnx_path, sample_input, torch_out)

    tf_model = build_tf_model(cfg, input_dim, len(label_map), model.state_dict())
    tflite_bytes = export_tflite(tf_model, export_dir / "model.tflite")
    tflite_delta = validate_tflite(tflite_bytes, sample, torch_out)

    summary = {
        "onnx_path": str(onnx_path),
        "tflite_path": str(export_dir / "model.tflite"),
        "onnx_delta": onnx_delta,
        "tflite_delta": tflite_delta,
    }
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Export summary saved")


if __name__ == "__main__":
    main()
