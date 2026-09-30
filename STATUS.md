# Status

Updated 2026-09-30. Code checkpoint: `8b128c6` on `master`.
PR [#25](https://github.com/Ton-Llop/Jaull/pull/25) is merged.

## What works

Jaull detects hardware, recommends models and prepares an execution plan.
It verifies the artifact, runs the model, saves experiments and benchmarks,
and compares predictions with measurements when the evidence allows it.

- CLI and TUI use the same services.
- Recommendations and execution readiness are separate: missing dependencies
  block execution, not the recommendation.
- Task and workload mode are separate. Workload requirements and optional
  throughput/TTFT targets can be recorded; they are not measured results.
- New llama.cpp plans no longer translate KV sequence batch size into
  `--batch-size`. Token batching uses runtime defaults; explicit old flags remain.
- Artifact verification rejects incorrect downloads and removes stale SHA
  sidecars before replacing files.

## Checks and evidence

- [CI for `8b128c6`](https://github.com/Ton-Llop/Jaull/actions/runs/36748809534)
  completed successfully. CI includes tests, Ruff, mypy, packaging checks and an
  installed-wheel smoke test.
- Local cleanup validation: **1,923 tests passed**. Ruff, mypy, compileall and
  architecture checks also passed.
- The RTX 4060 campaign has a successful Validate and a complete
  pp512/tg128 benchmark with three repetitions.
- [The comparison report](docs/qwen2.5-tests/rtx4060-campaign-comparison.md)
  explains the RTX 2060/4060 results and their limits. Different builds and
  environments mean this is not a controlled GPU comparison.
- Public RTX 4060 evidence is anonymized. Original bytes are kept privately;
  see [the evidence index](validation/README.md). Old failed runs remain failures.

## What is still missing

- No real multiuser workload experiment or qualification verdict yet.
- NVML process VRAM is unavailable on the measured WDDM machines. Runtime
  buffer comparisons work where the methodology gates allow them, but do not
  measure every process allocation.
- Host RSS is not the host share of GPU offload because llama.cpp uses mmap.
- Memory overhead, reserve and headroom are not calibrated from these runs.

## Next steps

1. Review this documentation checkpoint. `v0.2.0-alpha` is a proposed release,
   not a published tag.
2. Define a small llama.cpp workload experiment: one loaded `llama-server`,
   then 1, 2 and 4 simultaneous users. Separate loading, warm-up and measurement;
   record per-request TTFT, generation speed, failures and memory sources.
3. Only after that, consider evidence-based qualification and `jaull.lock`.

No HFA, ranking, scoring or calibration changes are part of this checkpoint.
Historical audits remain in [`docs/history/`](docs/history/).
