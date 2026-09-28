# Script index

Small development and validation tools. Production entry points remain the `jaull` CLI/TUI.

| Script | Purpose | Notes |
|---|---|---|
| [`bake_shark_frames.py`](bake_shark_frames.py) | Convert source GIF frames to compact Python sprite data for the TUI. | Requires Pillow; writes under `src/jaull/tui/widgets/` unless `--output` is set. |
| [`capture_screenshots.py`](capture_screenshots.py) | Regenerate deterministic TUI SVG screenshots with fake services. | Offline; defaults to `docs/assets/`. |
| [`check_dist.py`](check_dist.py) | Check wheel and sdist contents after `uv build`. | Reads `dist/`; no writes. |
| [`hardware_fit_matrix.py`](hardware_fit_matrix.py) | Render the checked-in hardware-fit scenario catalogue. | Offline; `--write-snapshot` explicitly updates its snapshot. |
| [`hardware_fit_vs_llmfit.py`](hardware_fit_vs_llmfit.py) | Compare the placement answers from Jaull and `llmfit`. | Requires `llmfit` for the external side; comparison is diagnostic, not ground truth. |
| [`validate_hardware_fit_against_llama_cpp.py`](validate_hardware_fit_against_llama_cpp.py) | Freeze a hardware-fit prediction, run llama.cpp, and report observed memory/placement. | Requires a local GGUF, llama.cpp and compatible GPU; does not calibrate the estimator. |
| [`validate_local_tensor_launch.py`](validate_local_tensor_launch.py) | Replay a recorded tensor launch budget, optionally running llama.cpp/llama-bench controls. | `--execute` runs workloads; output directory must be new. |
| [`validate_prediction_comparison.py`](validate_prediction_comparison.py) | Run one CPU TinyLlama prediction/observation/comparison check. | May download the model unless already cached; requires `llama-cli`. |

For the controlled cross-machine procedure and evidence rules, see
[`docs/experimental-validation.md`](../docs/experimental-validation.md). Preserve generated
reports and raw logs as evidence; do not treat a script run as calibration by itself.
