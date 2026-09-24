# B001-R8: RTX 2060 context matrix

## Scope

Controlled re-execution of the local `bartowski/Qwen2.5-7B-Instruct-GGUF`
Q4_K_M artifact on the RTX 2060. The purpose is to check that the current
estimator, launch path and observation contract behave consistently as context
changes. No formula, reserve, safety margin or calibration constant was changed.

Artifact: `Qwen2.5-7B-Instruct-Q4_K_M.gguf` from the local model directory.
Runtime: llama.cpp build `689e227db` / version `10357`.
Prompt: `Explain briefly what a local language model is.`
Tokens: 32. Concurrent users: 1.

The raw reports and logs are stored under
`validation/b001-r8-2060-context-matrix/`.

## Results

| Context | HFA placement | HFA blocks | llama.cpp maximum started | CUDA buffers at HFA probe | Peak RSS |
|---:|---|---:|---:|---:|---:|
| 512 | `gpu_offload` | 19/28 | 29/29 | 3032.1 MiB | 4722.3 MiB |
| 2048 | `gpu_offload` | 18/28 | 29/29 | 2957.0 MiB | 4722.4 MiB |
| 4096 | `gpu_offload` | 15/28 | 29/29 | 2609.8 MiB | 4722.1 MiB |

A repeat at context 4096 also completed: HFA selected 18/28 blocks with a
different instantaneous VRAM budget, while the HFA probe reported 3026.1 MiB
of CUDA buffers and peak RSS 4722.1 MiB. The different selected block count
demonstrates why available memory and the exact prediction input must be
recorded for every run.

All three launches completed successfully. The model did not reach a fully
resident HFA placement on this machine: the Q4_K_M weights alone are about
4466 MiB, while the available VRAM varied between about 4439 and 4993 MiB and
the policy also reserves runtime memory and headroom.

## Interpretation

The context trend is coherent: increasing context reduces the predicted GPU
placement because the KV cache grows from 28 MiB to 224 MiB. The measured
runtime buffer is not presented as a prediction error because HFA counts
runtime-agnostic transformer blocks while llama.cpp reports backend-specific
`--n-gpu-layers` units. The probe therefore compares neither placement nor
component totals and does not calibrate any estimator constant.

RSS remains around 4.7 GiB across placements. This is expected to include the
memory-mapped model file and is not the same quantity as HFA's `ram_required`.

## Throughput, and what the sweep does not say

This matrix recorded that every offload level started; it did not record how
fast any of them ran. Read without that, the table above invites a wrong
conclusion — HFA predicts 15–20 blocks while llama.cpp starts with 29/29, so the
gap looks like a 3–4x performance loss.

The HFA block count is not what Jaull launches. Measured in
[B001-R9](b001-r9-launch-policy-throughput.md), the launch policy emits
`--n-gpu-layers` 25–26 for these same contexts. In one run per cell, full offload
was **1.4x–2.1x** faster than the selected level, not the 3–4x inferred by
mistaking HFA blocks for launch units. Full offload left 196–366 MiB free after
baseline occupancy; the typical performance gap and failure threshold remain
unmeasured.

## Limitations

- NVML cannot attribute process VRAM under WDDM on this GeForce, so the runtime
  buffer report is the usable observation source.
- A validated mapping between HFA transformer blocks and llama.cpp offload units
  is still required before component-wise prediction error can be reported.
- This matrix validates execution and observation plumbing; it is not a basis
  for changing runtime overhead, device reserve, safety margin or launch policy.
