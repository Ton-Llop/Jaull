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

- Capability is still estimated from size and publisher-reported catalog claims.
  All 11 entries lack a precision and a revision, so they describe a model
  rather than the artifact that runs.
- No qualification verdict yet. The 1/2/4 concurrent-user experiment is measured
  under `validation/2060-1vs2vs4-users/` and is deliberately not a priority.
- NVML process VRAM is unavailable on the measured WDDM machines. Runtime
  buffer comparisons work where the methodology gates allow them, but do not
  measure every process allocation.
- Host RSS is not the host share of GPU offload because llama.cpp uses mmap.
- Memory overhead, reserve and headroom are not calibrated from these runs.

## Next steps

The priority is how Jaull estimates quality, not multiuser throughput.

1. Review this documentation checkpoint. `v0.2.0-alpha` is a proposed release,
   not a published tag.
2. Review the [quality-evaluation pilot](docs/quality-evaluation-pilot.md)
   on `docker-models-comp`: two exact GGUFs completed sequential three-example
   smokes. Records remain plumbing diagnostics, not reusable quality evidence.
3. Go from a three-example smoke to a limited pilot: enough samples for a
   confidence interval, and how often a placement change moves an answer. A
   `--limit 20` must never become "this model is better".
4. Only then turn that evidence into a Quality signal, kept separate from
   Fastest. Prefer a gate on measured throughput over a weighted score, which
   would reintroduce the global score the ranking dropped.

No HFA, ranking, scoring or calibration changes are part of this checkpoint.
Historical audits remain in [`docs/history/`](docs/history/).
