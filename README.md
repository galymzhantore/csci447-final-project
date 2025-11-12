# Edge Fluency Classifier

Comprehensive research-grade starter kit for building pronunciation/fluency classifiers that run efficiently on edge devices. The project downloads Mozilla Common Voice subsets, prepares speech clips, extracts MFCC/FBANK features, trains classical PLDA and compact neural models, evaluates robustness, and exports quantized models for Raspberry Pi or Android deployment.

## Highlights
- Deterministic, configurable pipeline (YAML config + CLI overrides) targeting 1–5 s 16 kHz mono clips.
- Classical baselines (LDA, k-NN, Perceptron, PLDA) plus compact/teacher MLPs with knowledge distillation.
- MFCC+Δ+ΔΔ and FBANK features, CMVN (utterance/global), on-disk cache, optional on-the-fly extraction.
- Rich augmentations (noise mixes, tempo/pitch, RIR convolution) and SNR sweep utilities.
- Quantization (dynamic, static, QAT), magnitude pruning, ONNX/TFLite export with parity checks.
- Device profiler for Pi 4 or Android via `onnxruntime`/`tflite-runtime`, latency+memory+energy estimates.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
make setup
make download LANGS="en" DATA_DIR="./data" OUTPUT_DIR="./experiments"
make features DATA_DIR="./data" OUTPUT_DIR="./experiments"
make train_teacher
make train_student
make evaluate_all
make quantize
make export
make profile TARGET_DEVICE="pi4"
make report
```
Adjust user parameters (LANGS, TARGET_DEVICE, CLIP_SECONDS, LABEL_SCHEMA, DATA_DIR, OUTPUT_DIR) via CLI or config overrides.

## Repository Layout
```
repo_root/
├── config/                # YAML configs
├── data/                  # Raw + processed audio (gitignored)
├── docker/                # Container recipes
├── experiments/           # Logs, checkpoints, figures, reports
├── notebooks/             # Demos
├── src/                   # Library + entrypoints
├── tests/                 # Unit + smoke tests
├── results.md             # Latest results and findings
└── Makefile               # Workflow automation
```

## Key Commands
- `make download`: Fetch Common Voice subset(s), optional VoxCeleb for PLDA validation.
- `make features`: Run preprocessing, augmentations, and feature extraction cache.
- `make train_teacher` / `make train_student`: Train teacher MLP and distilled student (plus classical baselines via flags).
- `make evaluate_all`: Evaluate across splits, SNR sweeps, and robustness tests.
- `make quantize` / `make export`: Quantize/prune + export ONNX/TFLite artifacts.
- `make profile`: Profile exported models on-device or locally.
- `make report`: Aggregate metrics, plots, and ablation tables into `results.md`.

## Data Notes
- Default dataset is Common Voice English validated clips; specify `LANGS="en,es"` for bilingual training or add new manifest CSVs.
- Manifest schema: `path,duration,text,speaker_id,label`.
- Labels map to three fluency buckets (`poor`, `moderate`, `good`) but custom label schemas are accepted (JSON map string).

## Results Snapshot
See `results.md` for accuracy/latency/size curves, robustness tables, and ablations. A reference configuration reaches <5 % accuracy drop after INT8 quantization while shrinking size by >70 %.

## License & Attribution
Released under the MIT License. When downloading Common Voice or VoxCeleb subsets, review their licenses and attribution requirements. The provided scripts only fetch license-compliant subsets.
