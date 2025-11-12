import numpy as np

from features.extractor import extract_features


def test_mfcc_feature_shape():
    waveform = np.random.randn(16000).astype(np.float32)
    cfg = {"type": "mfcc", "mfcc": {"n_mfcc": 13}, "frame_length_ms": 25, "frame_shift_ms": 10}
    feats = extract_features(waveform, 16000, cfg)
    assert feats.shape[0] >= 13
    assert feats.ndim == 2
