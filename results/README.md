# Results

Artifacts in this folder mirror the latest end-to-end run:

- `profile_summary.json` – host + Android latency/memory stats from `make profile`/ADB.
- `report.md` – rendered summary (same content as the root `results.md`).
- `jupyter_profile_summary.json` – combined export + profiling snapshot emitted by the device notebook.

Re-run `make quantize && make export_onnx && make report` (plus `make profile`/ADB if you need fresh device numbers) to refresh everything here.
