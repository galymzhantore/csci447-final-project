from __future__ import annotations

import json
from pathlib import Path

from utils.cli import parse_config
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    cfg = parse_config("Aggregate reports")
    root = Path(cfg["data"].get("output_dir", "experiments"))
    eval_metrics = read_json(root / "metrics" / "evaluation.json")
    quant_summary = read_json(root / "quantized" / "summary.json")
    profile = read_json(root / "profiles" / "summary.json")
    export_summary = read_json(root / "exports" / "summary.json")

    report_lines = [
        "# Edge Fluency Results",
        "",
        "## Split Metrics",
    ]
    for split, stats in (eval_metrics.get("splits", {})).items():
        report_lines.append(f"- **{split}**: acc={stats.get('accuracy', 0):.3f}, macro_f1={stats.get('macro_f1', 0):.3f}")
    report_lines.append("")
    report_lines.append("## SNR Robustness")
    for snr, acc in (eval_metrics.get("snr", {})).items():
        report_lines.append(f"- {snr} dB: {acc:.3f} acc")
    report_lines.append("")
    report_lines.append("## Quantization & Pruning")
    for name, stats in quant_summary.items():
        if not isinstance(stats, dict):
            continue
        line = f"- {name}: {stats.get('size_mb', 0):.2f} MB"
        if stats.get("accuracy") is not None:
            line += f", acc={stats['accuracy']:.3f}"
        if stats.get("accuracy_drop") is not None:
            line += f" (drop {stats['accuracy_drop']:.3f})"
        report_lines.append(line)
    report_lines.append("")
    report_lines.append("## Device Profiling")
    for entry in (profile if isinstance(profile, list) else []):
        model = entry.get("model", "unknown")
        if model == "android_device":
            report_lines.append(f"- Android app: {entry.get('latency_ms')} ms avg, {entry.get('memory_mb')} MB, battery {entry.get('battery_drop_mwh', 'n/a')}")
            continue
        report_lines.append(
            f"- {model}: {entry.get('latency_ms', 0):.2f}±{entry.get('std_ms', 0):.2f} ms, mem={entry.get('memory_mb', 0):.1f} MB, energy proxy {entry.get('energy_proxy', 0):.2f}"
        )
    report_lines.append("")
    report_lines.append("## Export Parity")
    report_lines.append(f"- ONNX delta {export_summary.get('onnx_delta', float('nan')):.6f}")
    if export_summary.get("fp32_tflite"):
        report_lines.append(f"- FP32 TFLite acc={export_summary['fp32_tflite'].get('accuracy', float('nan')):.3f}")
    if export_summary.get("int8_tflite"):
        int8_info = export_summary["int8_tflite"]
        drop = int8_info.get("accuracy_drop")
        drop_txt = f"{drop:.3f}" if isinstance(drop, (float, int)) else "n/a"
        report_lines.append(
            f"- INT8 TFLite acc={int8_info.get('accuracy', float('nan')):.3f} (drop {drop_txt})"
        )

    results_path = Path("results.md")
    results_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Updated %s", results_path)

    results_dir = Path(cfg.get("reporting", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
