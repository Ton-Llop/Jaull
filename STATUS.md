# Status

Updated 2026-11-17. Last verified remote CI checkpoint: [`ba5b5da`](https://github.com/Ton-Llop/Jaull/actions/runs/44152938519).

## What works

Jaull detects hardware, recommends models and prepares an execution plan.
It verifies the artifact, runs the model, saves experiments and benchmarks,
and compares predictions with measurements when the evidence allows it.

- CLI and TUI use the same services.
- Recommendations and execution readiness are separate: missing dependencies
  block execution, not the recommendation.
- Task and workload mode are separate. Workload requirements and optional
  throughput/TTFT targets can be recorded; they are not measured results.
- Artifact verification rejects incorrect downloads and removes stale SHA
  sidecars before replacing files.
- **Serve** sizes context and slots, starts a local `llama-server` without Docker,
  and distinguishes a live process from one that responds to its health check.
  It checks the server binary and device before launch and rehashes cached models.
  Older published GGUFs are still accepted when the bytes match their accepted SHA.
- **Measured ranking policies are active.** After the base order, stored
  measurements may reorder plans:
  - Quality reads a complete audited IFEval run of the plan's exact GGUF.
  - Fastest reads the median `llama_bench_v2` tg128 of at least three calls of
    the plan's own configuration (pp512 + tg128, 5 repetitions, `-fa off`,
    threads = physical cores), on the same machine, GPU driver/CUDA, device and
    offload. A pair moves only when the observed ranges do not overlap.
  - Balanced compares each pair directly: first when no axis is clearly worse
    and one is clearly better (quality by 95% Wilson interval). Overlapping
    speed lets clearly better quality decide; a real trade-off never moves.
  - Memory has no measured policy.
  Measured plans only swap places with each other inside one comparable cohort
  and stratum; without applicable evidence the order is the base order.
  `llama_bench_v1` records remain references only.
- **Evaluate candidates follows the search priority**: IFEval for Quality,
  Speed (no Docker) for Fastest, both for Balanced, running only the axis each
  candidate is missing. Evaluator setup is automatic; downloads need explicit
  permission. Recompare proposes, from stored quality and speed, what a search
  would change. Free disk space deletes downloaded models and run folders while
  keeping the records.
- The quality store preserves repeated runs; identical repeats count as one
  result and differing repeats are a conflict, never an applicable score.

## Checks and evidence

- The CI checkpoint includes Linux and Windows tests, static checks, and packaging.
  It does not substitute for GPU tests.
- On Intel Arc Pro B60, the KV and recurrent-state buffer predictions agreed with
  llama.cpp's reported allocations in the tested 1/2/4-slot configuration.
  A separate native Windows/Vulkan run exercised Serve, idle sleep and wake-up.
  These results are specific to the tested model, build and workload; they do not
  validate general VRAM availability or sustainable concurrency.
  See [the B60 experiments](docs/TFG/concurrencia-b60.md).
- RTX 2060 campaigns (2026-10-10/11), all three plans ranking-ready, tg128
  median over all stored calls: Qwen2.5-0.5B Q5_K_M 246.1 tok/s (6 calls),
  IFEval 25.9 %; Qwen2.5-1.5B 150.7 tok/s (4), 41.0 %; LFM2.5-1.2B 230.5 tok/s
  (4), 80.2 %. Balanced keeps the Qwen pair as a trade-off. See
  [the plan](docs/fastest-balanced-plan.md) and
  [the noise campaign](docs/qwen2.5-tests/rtx2060-tg128-noise-20261011.md).
- An offline tg128 predictor (not in the product) was frozen before measuring four
  new RTX 2060 artifacts. Its observed errors were −6.1 % to +10.5 %, but the audit
  found a WSL kernel change between calibration and measurement: this is exploratory
  cross-environment evidence, not validated accuracy within a comparable cohort.
  The corrected scorer rejects the environment mismatch without rewriting the originals. See
  [the report](docs/qwen2.5-tests/rtx2060-speed-predictor-20261014.md).
- Results distinguishes applicable measured speed from the latest metadata-matched
  Transformers benchmark (historical, not ranking evidence). Without either, there
  is no speed label; the Speed shortcut remains available. Predictions were withdrawn
  after a Transformers case measured 18.36 tok/s against roughly 170 estimated.
  The offline predictor and original experiments remain available for the TFG;
  other-condition benchmarks stay historical references in the details.
  See `docs/recommendation.md`.
- The RTX 4060 campaign has a successful Validate and a complete pp512/tg128
  benchmark. [The comparison report](docs/qwen2.5-tests/rtx4060-campaign-comparison.md)
  explains the RTX 2060/4060 results and their limits.
- Public RTX 4060 evidence is anonymized. Original bytes are kept privately;
  see [the evidence index](validation/README.md). Old failed runs remain failures.

## What is still missing

- Speed noise between calls is 4-14 % on the RTX 2060
  ([campaign](docs/qwen2.5-tests/rtx2060-tg128-noise-20261011.md)). The min-max
  range only grows, so one atypical call can keep two plans indistinguishable
  indefinitely; a robust range is a later improvement.
- Capability estimation without measurements still uses a scale prior.
  Published catalog results are references only.
- No automated qualification verdict or sustainable capacity estimate. The
  concurrency experiments on the RTX 2060 and B60 are evidence, not a product gate.
- NVML process VRAM is unavailable on the measured WDDM machines.
- Host RSS is not the host share of GPU offload because llama.cpp uses mmap.
- Memory overhead, reserve and headroom are not calibrated from these runs.

## Next steps

1. Repeat the native Windows B60 Serve run and compare prompt cache on/off with the
   same workload, recording effective flags and minimum free system RAM.
2. Consolidate the TFG evidence by separating tested code, physical measurements,
   and unsupported or unvalidated configurations.

Historical audits remain in [`docs/history/`](docs/history/).
