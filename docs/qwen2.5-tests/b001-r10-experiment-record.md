# B001-R10: persisted runtime allocation observation

## Purpose

Run the current `ExperimentRequest` path with the local Qwen2.5-7B Q4_K_M
artifact and verify that a successful execution persists llama.cpp's device
buffer report in an `ExperimentRecord`. This is a single plumbing and
observation run; it is not used for calibration or throughput conclusions.

## Method

- Hardware: RTX 2060 6 GiB, Ryzen 5 3600, WSL2.
- Artifact: `bartowski/Qwen2.5-7B-Instruct-GGUF`,
  `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, SHA-256
  `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`.
- Runtime: llama.cpp `689e227db` / version `10357`.
- Prediction inputs: frozen analysis and inference configuration from B001-R4;
  hardware freshly scanned and estimate recalculated with current code.
- Workload: context 4096, one user, fixed validation prompt
  `Explain briefly what artificial intelligence is.`
- Execution: one persisted `ExperimentRequest`, through
  `AdvisorService.run_experiment`.

The full persisted record is [experiment.json](../../validation/b001-r10-experiment-record/experiment.json);
the compact machine-readable summary is [summary.json](../../validation/b001-r10-experiment-record/summary.json).
The store copy is under `~/.local/share/jaull/experiments/` and its `experiment`
payload matches the repository evidence copy.

## Result

The request was ready and completed successfully in 16.05 seconds, exit code 0.
At selection time, 3.39 GiB VRAM and 6.05 GiB RAM were available. The current
estimate selected `gpu_offload` with compatibility `offloading_required` and
`--n-gpu-layers 14`.

| CUDA0 buffer | Measured |
|---|---:|
| Model | 2172.17 MiB |
| KV cache | 104.00 MiB |
| Compute | 183.44 MiB |
| **Total device allocation** | **2459.61 MiB** |

The record contains all eight parsed host/device buffers, `prediction_input`,
the executed command and the experiment environment (`d2ec8cf`). As expected on
this WDDM setup, `peak_vram_bytes` is `null`; the runtime-reported allocation is
the available device observation.

`backend_trace.observed_backend` is also `null`, although
`runtime_allocation.device` and its device buffer labels identify `CUDA0`. A
separate diagnostic rerun emitted `ggml_backend_cuda_graph_compute` and the
CUDA0 allocation lines. The detector recognized `ggml_cuda` markers but not
this equivalent CUDA backend marker; it now recognizes both. The immutable R10
record remains unchanged and accurately preserves the original `null` value.

## Comparison and limits

The VRAM comparison is `methodologically_unavailable`: HFA's block placement
does not map to this run's llama.cpp offload units in the comparison layer. No
error percentage should be inferred from the predicted and observed totals. The
run confirms the persistent observation path and adds one data point; it does
not calibrate memory estimates, reserves, margins or launch policy.
