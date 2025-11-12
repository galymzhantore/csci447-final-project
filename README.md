# Edge Fluency Classifier

Edge-ready pipeline for scoring pronunciation/fluency from 1–5 s speech clips, exporting tiny TensorFlow Lite models, and deploying them in an on-device Android app with full profiling hooks.

## Highlights

- **Data**: Pull Mozilla Common Voice (or Speechocean) splits with language filtering (`LANGS="en,es"`), optional locally recorded clips, deterministic manifests, and augmentation-ready metadata.
- **Features**: MFCC/FBANK extraction with CMVN, augmentation (noise, pitch/tempo, RIR), caching, and configurable clip windows.
- **Models**: Compact MLPs (teacher/student), classical PLDA scoring, optional distillation, multi-metric evaluation, and SNR robustness sweeps.
- **Quantization**: Dynamic + static INT8 quantization (PyTorch) with accuracy tracking, ONNX export, FP32/INT8 TFLite generation, and metadata packs for Android assets.
- **Deployment**: Jetpack Compose app with on-device MFCC extraction, AudioRecord capture, selectable clips, live latency/memory/battery logging, and `adb`-retrievable metrics for profiling.

## End-to-End Workflow

```bash
python -m venv .venv && source .venv/bin/activate
make setup

# 1) Data + features
make download LANGS="en"
make features

# 2) Training
make train_teacher
make train_student
make train_plda           # optional classical baseline
make evaluate_all

# 3) Edge readiness
make quantize
make export_tflite
make convert_android      # copies assets into android_app/

# 4) Android + profiling
make android_build        # requires Android SDK
make android_profile DEVICE="android_tablet"

# 5) Reporting
make report
```

Core parameters (override via env or CLI): `LANGS`, `DEVICE/TARGET_DEVICE`, `CLIP_SECONDS`, `LABEL_SCHEMA`, `DATA_DIR`, `OUTPUT_DIR`.

## Repository Layout

```
repo_root/
├── android_app/                   # Android Studio project (Jetpack Compose + TFLite)
├── config/default.yaml            # Global configuration
├── data/                          # Raw audio + caches (gitignored)
├── docker/Dockerfile.cpu          # Repro training environment
├── experiments/                   # Artifacts (checkpoints, metrics, exports)
├── notebooks/demo_android_pipeline.ipynb
├── results/                       # Final report + profiler exports
├── src/                           # Python library + entrypoints
├── Makefile                       # Automation shortcuts
└── README.md
```

## Make Targets (Automation)

| Target | Description |
| ------ | ----------- |
| `make download` | Stream Common Voice/Speechocean splits, merge optional local recordings, stratify manifests. |
| `make features` | Run preprocessing, augmentations, MFCC/FBANK extraction, and CMVN stats. |
| `make train_teacher` / `make train_student` | Train teacher MLP and distilled student. |
| `make train_plda` | Fit a classical PLDA baseline and persist as `plda.joblib`. |
| `make evaluate_all` | Compute per-split metrics + SNR sweeps with figures under `experiments/`. |
| `make quantize` | Run dynamic/static INT8 quantization + pruning with accuracy tracking. |
| `make export_tflite` | Produce ONNX + FP32/INT8 TFLite, label maps, and metadata. |
| `make convert_android` | Copy latest model + metadata into `android_app/app/src/main/assets/`. |
| `make android_build` | Assemble the Android app (requires local Android SDK). |
| `make android_profile` | Pull profiler metrics (host + Android device via `adb`). |
| `make report` | Summarize metrics/latency/size into `results.md` and `results/report.md`. |

## Android App (Iguana-ready)

- Kotlin + Jetpack Compose UI (`MainActivity.kt`) with record/select audio flows, prediction cards, and telemetry view.
- `FeatureExtractor.kt` implements MFCC(+Δ/+ΔΔ) to keep preprocessing on-device.
- `InferenceHelper.kt` loads the quantized TFLite model, label map, and metadata, returning latency-aware predictions.
- `MetricsLogger.kt` writes latency/memory/battery snapshots to `files/metrics/latest.json`, allowing `make android_profile` to pull them via `adb shell run-as`.
- Assets (`model_fluency.tflite`, `label_map.json`, `metadata.json`) are refreshed after `make convert_android`.

Open `android_app/` in Android Studio Igauana+, plug in a tablet, and run to capture profiler traces. Exported screenshots/JSON can be stored under `results/`.

## Data Notes

- Default dataset: `mozilla-foundation/common_voice_17_0`. Labels are derived from up/down vote ratios (100-point pseudo-score mapped to `poor`, `moderate`, `good` thresholds).
- To switch to Speechocean, set `data.dataset: speechocean` or run `make download DATA_DIR=... LANGS="en"`.
- Custom label schemas: pass `LABEL_SCHEMA='{"bad":0,"ok":1,"great":2}'` or edit `config/default.yaml`.
- Local bilingual snippets: run `python -m src.data.local_recorder --output-dir data/local --label moderate` and re-run `make download`.

## Results & Profiling

- Quantization summary (`experiments/quantized/summary.json`) tracks size + accuracy deltas for FP32, dynamic, static, and pruned models.
- Android + host profiling writes CSV/JSON to `experiments/profiles/` and mirrors summaries to `results/profile_summary.json`.
- `make report` stitches everything into `results.md` + `results/report.md`, including INT8 accuracy drop (goal <5 %), latency (<100 ms/clip), and model size (<10 MB).

## Docker

`docker/Dockerfile.cpu` provisions Python 3.11 with all dependencies, enabling reproducible training/export runs:

```bash
docker build -t edge-fluency -f docker/Dockerfile.cpu .
docker run --rm -v $PWD:/workspace edge-fluency bash -lc "make download && make features && make train_teacher"
```

## License & Attribution

Released under the MIT License. Ensure compliance with Common Voice/Speechocean dataset terms when downloading or redistributing audio samples.
