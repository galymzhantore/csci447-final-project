from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from audio.processing import preprocess
from audio.augment import apply_augmentations
from features.cache import FeatureCache
from features.cmvn import CMVN
from features.extractor import extract_features
from utils.config import load_config
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def process_manifest(manifest_path: Path, cfg: dict, cache: FeatureCache, cmvn: CMVN, aug_cfg: dict, sample_rate: int, clip_range: list[float]) -> None:
    df = pd.read_csv(manifest_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"features-{manifest_path.stem}"):
        audio_path = row["path"]
        feats = cache.load(audio_path)
        if feats is not None:
            cmvn.update(feats)
            continue
        waveform = preprocess(audio_path, sample_rate, clip_range[0], clip_range[1], aug_cfg.get("trim_db", 25), aug_cfg.get("vad_threshold", 0.5), aug_cfg.get("peak_normalize", True))
        if aug_cfg.get("enabled", False) and aug_cfg.get("for_features", False):
            waveform = apply_augmentations(waveform, sample_rate, aug_cfg)
        feats = extract_features(waveform, sample_rate, cfg)
        if cfg.get("cmvn") == "utterance":
            feats = (feats - feats.mean(axis=1, keepdims=True)) / (feats.std(axis=1, keepdims=True) + 1e-9)
        cmvn.update(feats)
        cache.save(audio_path, feats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MFCC/FBANK features with caching")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    feature_cfg = cfg.get("features", {})
    aug_cfg = cfg.get("augmentation", {})
    data_cfg = cfg.get("data", {})
    manifests_dir = Path(args.output_dir) / "manifests"
    if not manifests_dir.exists():
        raise FileNotFoundError("Manifests missing, run make download first")
    cache = FeatureCache(Path(feature_cfg.get("cache_dir", "data/features")), feature_cfg.get("type", "mfcc"))
    cmvn = CMVN()
    manifest_files = sorted(manifests_dir.glob("*.csv"))
    for manifest_path in manifest_files:
        process_manifest(manifest_path, feature_cfg, cache, cmvn, aug_cfg, data_cfg.get("sample_rate", 16000), data_cfg.get("clip_seconds", [1, 5]))
    cmvn_path = Path(args.output_dir) / "features" / "cmvn.npz"
    cmvn_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cmvn_path, mean=cmvn.mean, var=cmvn.var, count=cmvn.count)
    meta = {
        "feature_type": feature_cfg.get("type", "mfcc"),
        "cache_dir": feature_cfg.get("cache_dir", "data/features"),
        "cmvn_mode": feature_cfg.get("cmvn", "global"),
    }
    (Path(args.output_dir) / "features" / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Feature extraction completed; stats saved to %s", cmvn_path)


if __name__ == "__main__":
    main()
