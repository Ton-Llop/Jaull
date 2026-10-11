# Status

Updated 2026-10-11. Code checkpoint: `330bf0f` on `master`; CI passes.

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

- CI passes on `330bf0f`, including ruff, mypy, the full test suite and the
  architecture checks.
- RTX 2060 campaigns (2026-10-10/11), all three plans ranking-ready, tg128
  median over all stored calls: Qwen2.5-0.5B Q5_K_M 246.1 tok/s (6 calls),
  IFEval 25.9 %; Qwen2.5-1.5B 150.7 tok/s (4), 41.0 %; LFM2.5-1.2B 230.5 tok/s
  (4), 80.2 %. Balanced keeps the Qwen pair as a trade-off. See
  [the plan](docs/fastest-balanced-plan.md) and
  [the noise campaign](docs/qwen2.5-tests/rtx2060-tg128-noise-20261011.md).
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
- No qualification verdict yet. The 1/2/4 concurrent-user experiment is measured
  under `validation/2060-1vs2vs4-users/` and is deliberately not a priority.
- NVML process VRAM is unavailable on the measured WDDM machines.
- Host RSS is not the host share of GPU offload because llama.cpp uses mmap.
- Memory overhead, reserve and headroom are not calibrated from these runs.

## Next steps

1. Measure more plans on the RTX 2060 to strengthen the TFG evidence.
2. Broaden the catalog cautiously, with primary sources and exact provenance.

Historical audits remain in [`docs/history/`](docs/history/).
