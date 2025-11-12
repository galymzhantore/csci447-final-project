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
    for key, value in quant_summary.items():
        report_lines.append(f"- {key}: {value:.3f} MB")
    report_lines.append("")
    report_lines.append("## Device Profiling")
    for entry in profile:
        report_lines.append(f"- {entry['model']}: {entry['latency_ms']:.2f}±{entry['std_ms']:.2f} ms, energy proxy {entry['energy_proxy']:.2f}")
    report_lines.append("")
    report_lines.append("## Export Parity")
    report_lines.append(f"- ONNX delta {export_summary.get('onnx_delta', float('nan')):.6f}")
    report_lines.append(f"- TFLite delta {export_summary.get('tflite_delta', float('nan')):.6f}")

    results_path = Path("results.md")
    results_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Updated %s", results_path)


if __name__ == "__main__":
    main()
