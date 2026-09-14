# B001-R6: local GGUF tensor launch policy

## Scope

This validates local tensor-based launch refinement, not a change to Hardware Fit,
its transformer-block placement, ranking or memory formulas. It does **not** declare
the offload-performance issue solved. No measured throughput or tensor size was
used as a calibration constant.

The rule is limited to llama.cpp `689e227db`, dense `qwen2`, separate input/output
tensors and one confirmed discrete CUDA device. The source for that build places
input weights on CPU, the output tensors on GPU for positive runtime units, and
the last `u - 1` repeating blocks for `u` units (capped at the block count).
Unknown builds/layouts retain the conservative aggregate fallback.

## Artifact and method

Artifact: `bartowski/Qwen2.5-7B-Instruct-GGUF`,
`Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4,683,074,240 bytes.
Recorded SHA-256:
`65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`.
The experiment carries that identity; this descriptor-only reader does not rehash
the payload. The record uses revision `main`, not a pinned Hub commit.

The reader found 339 tensors, 28 repeating blocks and a data offset of 5,954,240
bytes. The stored tensor payload totals 4,677,120,000 bytes:

| Tensor group | Stored bytes | Basis |
|---|---:|---|
| `token_embd.weight` | 306,561,024 | Q4_K descriptor |
| `output.weight` | 447,068,160 | Q6_K descriptor |
| `output_norm.weight` | 14,336 | F32 descriptor |
| Repeating blocks | 3,923,476,480 | Sum of individual descriptors |

These are stored sizes, not process-attributed CUDA allocations. Block suffixes
use their individual sizes, not their average. The original device reserve
(512 MiB), runtime headroom (256 MiB) and estimated KV input are unchanged.

## Frozen inputs versus live inputs

The historical source is
[B001-R4 experiment](../../validation/bundles/b001-r4-launch-policy/records/experiment.json).
Neither its prediction nor its observation was overwritten.

| Launch calculation | Historical VRAM budget | Live VRAM budget |
|---|---:|---:|
| Available VRAM, bytes | 5,197,422,592 | 4,910,895,104 |
| Aggregate fallback, runtime units | 24 | 22 |
| Tensor refinement, runtime units | 27 | 25 |
| Repeating blocks on GPU | 26 | 24 |
| GPU block weight budget, bytes | 3,625,250,816 | 3,345,002,496 |
| GPU output weight budget, bytes | 447,082,496 | 447,082,496 |
| GPU KV budget, bytes | 218,103,808 | 201,326,592 |
| GPU required including reserve/headroom, bytes | 5,095,743,488 | 4,798,717,952 |
| Remaining planning headroom, bytes | 101,679,104 | 112,177,152 |
| Next runtime unit exceeds budget by, bytes | 55,822,336 | 27,346,944 |

The next candidate is rejected by the unchanged launch-budget inequality, not by
a throughput preference. The historical column is an offline re-evaluation, not
a second physical run or exact replay of the original code.

Frozen-input output:
[prediction](../../validation/bundles/b001-r6-tensor-policy-frozen/prediction.json).
Live output:
[prediction and commands](../../validation/bundles/b001-r6-tensor-policy/prediction.json).
Each records its UTC timestamp, source-record hash, script hash and policy-source
hashes. A Git commit alone would not identify these uncommitted sources.

## Recorded execution

Live timestamp: `2026-09-13T22:36:30.887656+00:00` (00:36 on September 14 in Madrid).
RTX 2060, 6 GiB, driver `616.92`; both CLI and benchmark identify build `689e227db`.
`llama-bench --version` is unsupported, so a zero-task probe captured its build
footer before measurement. That probe supplies a nonexistent local model and
does not load weights or perform inference.

The automatic CLI smoke at context 4096, `-ngl25`, generated 32 tokens and exited
successfully. Peak process RSS was 4,940,369,920 bytes; process-attributed VRAM was
unavailable. It is not a VRAM accuracy test. Raw output:
[smoke](../../validation/bundles/b001-r6-tensor-policy/launch-smoke-stdout.log).
The CLI's normal output does not enumerate placed tensors, so no new observed
block-count or buffer-size assertion is made from this smoke.

Sequential controls used the same local artifact, `-dev CUDA0`, pp512/tg128,
three repetitions and llama-bench's default warmup:

| Requested units | pp512 tok/s, mean +/- sd | tg128 tok/s, mean +/- sd |
|---|---:|---:|
| 25 (live automatic) | 1163.28 +/- 100.42 | 20.20 +/- 4.93 |
| 24 (historical-policy control) | 1159.45 +/- 39.76 | 19.41 +/- 3.42 |
| 28 | 1472.07 +/- 68.58 | 45.22 +/- 0.62 |
| -1 (full offload) | 1558.16 +/- 44.68 | 60.85 +/- 0.17 |

Raw logs and parsed measurements remain under
[`validation/bundles/b001-r6-tensor-policy`](../../validation/bundles/b001-r6-tensor-policy).
All commands exited successfully. The live aggregate fallback would have requested
22, which was **not measured** in this control series. The 25 versus 24 difference
is small relative to the observed standard deviations; this is not evidence of a
statistically established throughput improvement.

## Conclusion and limits

The structural correction works: the verified local artifact/build can drive a
different automatic launch without changing HFA or using a fitted safety factor.
However, manually requested 28/full offload still outperform the automatic choice.
The remaining gap is explicit, not resolved by this milestone.

These microbenchmarks do not measure sustained context-4096 generation, production
concurrency or a safe maximum under changing occupancy. The display shares this
GPU, the order was not randomized, only one session was measured, and thermal/CPU
effects may differ across controls. The 512+256 MiB policy allowance remains in the
automatic budget; successful manual launches do not establish a general reason
to remove it. No new memory-policy tuning or cross-build mapping was made.
