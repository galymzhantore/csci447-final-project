import numpy as np
import pytest

from export import validate_tflite


def test_validate_tflite_roundtrip():
    tf = pytest.importorskip("tensorflow")
    sample = np.random.rand(1, 4, 5).astype(np.float32)
    weights = np.ones((20, 3), dtype=np.float32)
    bias = np.zeros(3, dtype=np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20,)),
        tf.keras.layers.Dense(3, use_bias=True),
    ])
    model.layers[1].set_weights([weights, bias])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    torch_out = sample.reshape(1, -1).dot(weights) + bias
    delta = validate_tflite(tflite_bytes, sample, torch_out)
    assert delta < 1e-3
