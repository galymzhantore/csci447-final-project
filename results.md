# Edge Fluency Results

This report is generated automatically by `make report`. After running the end-to-end workflow it will contain:

- **Split metrics** (accuracy + macro F1 for train/valid/test) and SNR robustness curves.
- **Quantization summary** comparing FP32 vs dynamic vs static vs pruned models (size + accuracy drop).
- **Device profiling** for host ONNX/TFLite as well as Android tablet latency/memory/battery logs.
- **Export parity** (ONNX vs PyTorch delta, FP32 vs INT8 TFLite accuracy).

Refresh with:

```bash
make report
```

Artifacts are mirrored under `results/report.md` for sharing alongside profiler screenshots.
